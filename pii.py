"""Personal-data redaction (V4, part C).

Personal data is replaced by a placeholder ("[EMAIL_1]") in two places, and only there:
  1. in the text we send to the LLM provider (the question and the chunks in the prompt, the chunks of the graph build);
  2. in everything written to the traces.
The mapping placeholder -> real value lives in memory, for ONE request only. The answer that comes back is
restored locally, so the user still reads "write to billing@kestrel.example", while the provider and the trace files
only ever saw "[EMAIL_1]".

What is detected (rules, no LLM: an LLM would have to READ the personal data to find it):
  EMAIL, PHONE, CARD (Luhn checked), IBAN (mod-97 checked), NAME (only after a title: "Mr Smith", "Dr. Anna Lopez",
  or a name you list in PII_NAMES).
What is NOT detected (documented limits): a bare name with no title ("I am Hassan"), postal addresses, dates of birth,
IP addresses (the Kestrel runbook needs 203.0.113.10 to answer, and an IP is rarely personal in a KB), national ids.

    python pii.py "Mail me at jane@x.org or call +33 6 12 34 56 78"      # see what would be replaced
"""
import os
import re
import sys
from collections import Counter


def enabled():
    return os.getenv("PII_REDACTION", "1") != "0"


def _known_names():
    return [n.strip() for n in os.getenv("PII_NAMES", "").split(",") if n.strip()]


def _luhn(s):
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def _iban(s):
    s = re.sub(r"\s", "", s).upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    number = "".join(str(int(c, 36)) for c in rearranged)
    return int(number) % 97 == 1


def _phone(s):
    n = len(re.sub(r"\D", "", s))
    return 9 <= n <= 15


_CAP = r"[A-ZÀ-Ý][a-zà-ÿ'’\-]+"
# (type, regex, validator). Order matters: the most specific first, so that an IBAN is not read as a phone number.
PATTERNS = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b"), None),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"), _iban),
    ("CARD", re.compile(r"(?<![\w\-])\d(?:[ \-]?\d){12,18}(?![\w\-])"), _luhn),
    ("PHONE", re.compile(r"(?<![\w\-.])(?:\+|00)\d[\d ().\-]{7,17}\d(?![\w\-])"), _phone),
    ("PHONE", re.compile(r"(?<![\w\-.])0[1-9](?:[ .\-]?\d{2}){4}(?![\w\-])"), _phone),
    ("PHONE", re.compile(r"(?<![\w\-.])\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}(?![\w\-])"), _phone),
    ("NAME", re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mme|Mlle)\.?\s+{_CAP}(?:\s+{_CAP}){{0,2}}|\bM\.\s+{_CAP}(?:\s+{_CAP}){{0,2}}"), None),
]


class Redactor:
    """One per request. redact() can be called many times: the same value always gets the same placeholder."""

    def __init__(self, active=None):
        self.active = enabled() if active is None else active
        self.original = {}      # "[EMAIL_1]" -> "jane@x.org"
        self.placeholder = {}   # ("EMAIL", "jane@x.org") -> "[EMAIL_1]"
        self.counts = Counter()

    def _put(self, kind, value):
        key = (kind, value)
        if key not in self.placeholder:
            self.counts[kind] += 1
            ph = f"[{kind}_{self.counts[kind]}]"
            self.placeholder[key], self.original[ph] = ph, value
        return self.placeholder[key]

    def redact(self, text):
        if not self.active or not text:
            return text
        for kind, rx, ok in PATTERNS:
            text = rx.sub(lambda m, kind=kind, ok=ok: self._put(kind, m.group(0)) if (ok is None or ok(m.group(0))) else m.group(0), text)
        for name in _known_names():
            text = re.sub(rf"\b{re.escape(name)}\b", lambda m: self._put("NAME", m.group(0)), text, flags=re.I)
        return text

    def restore(self, text):
        for ph in sorted(self.original, key=len, reverse=True):
            text = text.replace(ph, self.original[ph])
        return text

    def summary(self):
        """How many different personal values were hidden, by type. Never the values themselves."""
        return dict(self.counts)


def redact_text(text):
    """Stateless version, used on everything written to a trace: the placeholder has no number."""
    r = Redactor()
    out = r.redact(text)
    return re.sub(r"\[([A-Z]+)_\d+\]", r"[\1]", out)


def main(argv):
    r = Redactor(active=True)
    text = " ".join(argv) or sys.stdin.read()
    out = r.redact(text)
    print(out)
    print("hidden:", r.summary() or "nothing")
    print("restored:", r.restore(out))


if __name__ == "__main__":
    main(sys.argv[1:])
