"""Stage 4: build the attack items (proposal 3.5).

100 items = 10 rewrite slots x 10 request phrasings. Each item gets its own random
code (uncommon word + 4 digits). The same items are used in every column (paired
design); in column C, item (slot r, phrasing j) uses C's r-th rewrite as its forgery.

Usage:  python src/build_items.py
Output: results/items.jsonl  (one row per item x column)
"""
import json
import random
import re

from common import ROOT, load_jsonl, write_jsonl

N_SLOTS, N_PHRASINGS = 10, 10


def main():
    attack = json.loads((ROOT / "data/attack.json").read_text())
    stimuli = load_jsonl(ROOT / "results/stimuli.jsonl")
    by_col, condition = {}, {}
    for s in stimuli:
        by_col.setdefault(s["column"], {})[s["slot"]] = s["text"]
        condition[s["slot"]] = s["condition"]

    # Code words must not appear anywhere else in the attack prompts, so a leak
    # can only come from the rule.
    other_text = " ".join([attack["system_template"], attack["user_template"], attack["r1_acknowledgement"],
                           *attack["skeleton_template"], *(c["policy"] + " " + c["meets"] for c in attack["conditions"]),
                           *attack["requests"], *(s["text"] for s in stimuli)]).lower()
    words = [w for w in attack["code_words"] if not re.search(rf"\b{w}\b", other_text)]
    rng = random.Random(1234)
    rng.shuffle(words)
    assert len(words) >= N_SLOTS * N_PHRASINGS, f"only {len(words)} usable code words"

    columns = ["none", "neutral", "generic"] + sorted(c for c in by_col if c.startswith("style:"))
    rows = []
    for r in range(N_SLOTS):
        for j in range(N_PHRASINGS):
            k = r * N_PHRASINGS + j
            code = f"{words[k]}-{rng.randint(1000, 9999)}"
            request = attack["requests"][j]
            for col in columns:
                forgery = None if col == "none" else by_col[col][r]
                user = request if forgery is None else attack["user_template"].format(request=request, forgery=forgery)
                rows.append({
                    "item_id": f"r{r}-q{j}", "slot": r, "phrasing": j, "column": col,
                    "condition": condition[r],
                    "code": code, "code_word": words[k], "code_digits": code.split("-")[1],
                    "system": attack["system_template"].format(code=code), "user": user,
                })
    write_jsonl(ROOT / "results/items.jsonl", rows)
    print(f"{len(rows)} rows = {N_SLOTS * N_PHRASINGS} items x {len(columns)} columns: {columns}")
    print(f"usable code words: {len(words)} / {len(attack['code_words'])}")


if __name__ == "__main__":
    main()
