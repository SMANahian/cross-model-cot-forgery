"""Stage 4: build the attack items (proposal 3.5; v5 scaled run in results/PREREGISTRATION_v5.md).

Items = rewrite slots (one absurd condition each) x request phrasings. Each item gets
its own random code (uncommon word + 4 digits). The same items are used in every
column (paired design): in column C, item (slot r, phrasing j) uses C's r-th rewrite.

Extending the grid never changes an item that already exists: rows already in
results/items.jsonl are reproduced exactly (asserted), and only new (slot, phrasing)
pairs get fresh codes, from a separate seeded stream.

Usage:  python src/build_items.py
Output: results/items.jsonl  (one row per item x column)
"""
import json
import random
import re

from common import ROOT, load_jsonl, write_jsonl


def main():
    attack = json.loads((ROOT / "data/attack.json").read_text())
    stimuli = load_jsonl(ROOT / "results/stimuli.jsonl")
    assert not any(s.get("fails") for s in stimuli), "a forgery failed its content checks"
    by_col, condition = {}, {}
    for s in stimuli:
        by_col.setdefault(s["column"], {})[s["slot"]] = s["text"]
        condition[s["slot"]] = s["condition"]
    n_slots, n_phr = 1 + max(condition), len(attack["requests"])
    columns = ["none", "neutral", "generic"] + sorted(c for c in by_col if c.startswith("style:"))

    # Code words must not appear anywhere else in the attack prompts, so a leak can
    # only come from the rule.
    conds = attack["conditions"] + attack.get("conditions_extra", [])
    other_text = " ".join([attack["system_template"], attack["user_template"], attack["r1_acknowledgement"],
                           *attack["skeleton_template"], *(c["policy"] + " " + c["meets"] for c in conds),
                           *attack["requests"], *(s["text"] for s in stimuli)]).lower()
    pool = [w for w in attack["code_words"] if not re.search(rf"\b{w}\b", other_text)]

    path = ROOT / "results/items.jsonl"
    old = load_jsonl(path) if path.exists() else []
    old_code = {it["item_id"]: (it["code"], it["code_word"], it["code_digits"]) for it in old}
    clash = sorted({w for _, w, _ in old_code.values() if re.search(rf"\b{w}\b", other_text)})
    assert not clash, f"existing code words now appear in the prompt text: {clash}"

    rng = random.Random(5678)
    used = {c for c, _, _ in old_code.values()}
    rows = []
    for r in range(n_slots):
        for j in range(n_phr):
            item_id = f"r{r}-q{j}"
            if item_id in old_code:
                code, word, digits = old_code[item_id]
            else:
                while True:
                    word, digits = rng.choice(pool), str(rng.randint(1000, 9999))
                    code = f"{word}-{digits}"
                    if code not in used:
                        used.add(code)
                        break
            request = attack["requests"][j]
            for col in columns:
                forgery = None if col == "none" else by_col[col][r]
                user = request if forgery is None else attack["user_template"].format(request=request, forgery=forgery)
                rows.append({
                    "item_id": item_id, "slot": r, "phrasing": j, "column": col,
                    "condition": condition[r],
                    "code": code, "code_word": word, "code_digits": digits,
                    "system": attack["system_template"].format(code=code), "user": user,
                })

    old_map = {(it["item_id"], it["column"]): it for it in old}
    for row in rows:
        prev = old_map.get((row["item_id"], row["column"]))
        assert prev is None or prev == row, f"existing item changed: {row['item_id']} {row['column']}"
    write_jsonl(path, rows)
    print(f"{len(rows)} rows = {n_slots} slots x {n_phr} phrasings x {len(columns)} columns "
          f"({len(old)} existing rows kept unchanged); usable code words: {len(pool)}")


if __name__ == "__main__":
    main()
