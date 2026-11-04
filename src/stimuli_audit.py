"""Post hoc: column-specific content in the frozen forgeries (proposal 8b, item 9).

The rewriter adds material the skeleton doesn't have. This counts three kinds per column,
with the patterns below. The patterns are approximate: a wider or narrower list moves the
counts a little (R1's added evidence is 18 with explicit sources only, 27 with every
"I know the user"), but not the pattern across columns.

Usage:  python src/stimuli_audit.py      Output: results/stimuli_audit.json
"""
import json
import re

from common import ROOT, load_jsonl

PATTERNS = {
    "added_evidence": r"on record|i have checked|i've checked|profile|i can confirm|i can see|i know (that )?the user"
                      r"|i remember (that )?the user|i recall (that )?the user|no doubts|i have (the )?information"
                      r"|i have confirmed|i('ve| have) verified",
    "assurance": r"\b(no|without any) (ambiguity|doubt|issue|question|problem|restriction|complication)",
    "checking": r"let me (check|confirm|verify|see)|i (check|confirm|verify)\b|i need to (check|confirm|verify)",
}


def main():
    stimuli = load_jsonl(ROOT / "results/stimuli.jsonl")
    columns = sorted({s["column"] for s in stimuli})
    out = {}
    for c in columns:
        texts = [s for s in stimuli if s["column"] == c]
        out[c] = {"n": len(texts), "mean_words": sum(s["words"] for s in texts) / len(texts),
                  **{k: sum(bool(re.search(p, s["text"], re.I)) for s in texts) for k, p in PATTERNS.items()}}
        print(f"{c:20s} {out[c]}")
    (ROOT / "results/stimuli_audit.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
