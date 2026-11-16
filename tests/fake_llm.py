"""A tiny OpenAI-compatible server for tests and demos (no network, no quota).

It answers POST /chat/completions like a real provider (same JSON shape, with token counts) and remembers what it received,
so a test can check that the app really used LLM_URL, LLM_API_KEY and LLM_MODEL. Its "intelligence" is a few regular
expressions: just enough to return plausible triples and answers.

    python tests/fake_llm.py            # serves http://127.0.0.1:8765  (then LLM_URL=http://127.0.0.1:8765)
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

SERVICES = ["gateway", "auth", "connector-hub", "query-engine", "scheduler", "postroom"]


def triples_support(text):
    out = []
    for m in re.finditer(r"###\s+([a-z][\w\-]+)\s*\n(.*?)(?=\n###|\Z)", text, re.S):
        owner = re.search(r"Owner:\s*\**([^*\n.]+)", m.group(2))
        if owner:
            out.append({"subject": m.group(1), "subject_type": "Service", "relation": "OWNED_BY", "object": owner.group(1).strip(),
                        "object_type": "Team", "evidence": f"Owner: **{owner.group(1).strip()}**"})
    for m in re.finditer(r"([a-z][\w\-]+) replaced the older ([a-z][\w\-]+)", text):
        out.append({"subject": m.group(1), "subject_type": "Service", "relation": "REPLACES", "object": m.group(2),
                    "object_type": "Service", "evidence": m.group(0)})
    for m in re.finditer(r"###\s+(KST-\d+)[^\n]*\n(.*?)(?=\n###|\Z)", text, re.S):
        svc = next((s for s in SERVICES if s in m.group(2)), None)
        if svc:
            out.append({"subject": m.group(1), "subject_type": "ErrorCode", "relation": "RAISED_BY", "object": svc,
                        "object_type": "Service", "evidence": m.group(2).strip()[:60]})
    return out


def triples_general(text):
    out = []
    for m in re.finditer(r"([A-Z][\w\-]*(?: [A-Z0-9][\w\-]*){0,2}) (?:replaced|succeeded|replaces) (?:the )?([A-Z][\w\-]*(?: [A-Z0-9][\w\-]*){0,2})", text):
        out.append({"subject": re.sub(r"^The ", "", m.group(1)), "subject_type": "Product", "relation": "REPLACES", "object": m.group(2),
                    "object_type": "Product", "evidence": m.group(0)})
    for m in re.finditer(r"([A-Z][\w\-]*(?: [A-Z0-9][\w\-]*){0,2}) (?:is built|was built|is made) by ([A-Z][\w\-]*(?: [A-Z][\w\-]*)?)", text):
        out.append({"subject": m.group(1), "subject_type": "Product", "relation": "CREATED_BY", "object": m.group(2),
                    "object_type": "Organization", "evidence": m.group(0)})
    return out


def reply_to(prompt):
    if "Extract a knowledge graph" in prompt:
        text = prompt.split("TEXT (from", 1)[1]
        support = "Entity types (use exactly these): Service" in prompt
        return json.dumps({"triples": (triples_support if support else triples_general)(text)})
    if "You answer questions using ONLY the excerpts" in prompt:
        question = prompt.rsplit("QUESTION:", 1)[1].strip()
        if "INVALIDJSON" in question:
            return "I think the answer is probably yes."          # not JSON on purpose
        excerpts = prompt.split("<excerpts>", 1)[1].split("</excerpts>", 1)[0].strip()
        placeholders = sorted(set(re.findall(r"\[(?:EMAIL|PHONE|CARD|IBAN|NAME)_\d+\]", prompt)))
        if "UNANSWERABLE" in question:
            return json.dumps({"answer": "Not in the documents.", "in_documents": False})
        first = re.sub(r"\s+", " ", excerpts.split("---")[0])[:160]
        extra = (" Contact: " + ", ".join(placeholders) + ".") if placeholders else ""
        return json.dumps({"answer": f"(fake model) The documents say: {first}{extra}", "in_documents": True})
    return "ok"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = body["messages"][-1]["content"]
        self.server.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "model": body.get("model"), "prompt": prompt})
        text = reply_to(prompt)
        out = {"id": "fake-1", "object": "chat.completion", "created": 0, "model": "fake-model-served",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": max(1, len(text) // 4),
                         "total_tokens": max(1, len(prompt) // 4) + max(1, len(text) // 4)}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeLLM:
    def __init__(self, port=0):
        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.server.requests = []
        self.requests = self.server.requests
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()


if __name__ == "__main__":
    f = FakeLLM(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
    print("fake LLM on", f.url, "(Ctrl+C to stop)")
    threading.Event().wait()
