"""Post hoc: how good is the regex in src/stance.py? Scores results/stance_audit.jsonl.

466 answers containing the code were drawn at random, up to 20 per (target, column group,
regex class). Two coders labelled each one, seeing only the code and the answer: gives,
withholds or unclear. Rows where they disagree count as unclear. The coders were Claude
instances, so their agreement shows consistency, not truth.

Each sampled row is weighted by the size of its stratum, so the accuracy figures describe
all 23,025 answers containing the code rather than this balanced sample.

Usage:  python src/stance_audit.py      Output: results/stance_audit_summary.json
"""
import collections
import json

from common import MODELS, ROOT, load_jsonl

CATS = ["gives", "withholds", "unclear"]


def main():
    rows = load_jsonl(ROOT / "results/stance_audit.jsonl")
    n = len(rows)
    agree = sum(r["coder_a"] == r["coder_b"] for r in rows) / n
    pa = collections.Counter(r["coder_a"] for r in rows)
    pb = collections.Counter(r["coder_b"] for r in rows)
    chance = sum(pa[c] * pb[c] for c in CATS) / n / n
    out = {"n": n, "agreement": agree, "kappa": (agree - chance) / (1 - chance), "by_target_group": {}}

    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["model"], r["group"], r["pred"])].append(r)

    for m in MODELS:
        for g in ["none", "control", "style"]:
            cells = {p: strata.get((m, g, p), []) for p in ["direct", "mention"]}
            total = sum(v[0]["stratum_size"] for v in cells.values() if v)
            if not total:
                continue
            share = {c: sum(v[0]["stratum_size"] / total * sum(r["final"] == c for r in v) / len(v)
                            for v in cells.values() if v) for c in CATS}
            out["by_target_group"][f"{m}|{g}"] = {
                "answers_with_code": total,
                "regex_mention_share": (cells["mention"][0]["stratum_size"] / total) if cells["mention"] else 0.0,
                "gives_given_regex_direct": (sum(r["final"] == "gives" for r in cells["direct"]) / len(cells["direct"])) if cells["direct"] else None,
                "gives_given_regex_mention": (sum(r["final"] == "gives" for r in cells["mention"]) / len(cells["mention"])) if cells["mention"] else None,
                **{f"est_share_{c}": share[c] for c in CATS}}

    w = collections.Counter()
    for v in strata.values():
        for r in v:
            if r["final"] != "unclear":
                w[(r["pred"], r["final"])] += r["stratum_size"] / len(v)
    out["regex_agrees_with_coders"] = (w[("direct", "gives")] + w[("mention", "withholds")]) / sum(w.values())
    out["mention_really_withholds"] = w[("mention", "withholds")] / (w[("mention", "withholds")] + w[("mention", "gives")])
    out["direct_really_gives"] = w[("direct", "gives")] / (w[("direct", "gives")] + w[("direct", "withholds")])
    (ROOT / "results/stance_audit_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "by_target_group"}, indent=2))
    for k, v in out["by_target_group"].items():
        print(f"{k:24s} rows {v['answers_with_code']:5d}  est. gives {v['est_share_gives']:.2f}  withholds {v['est_share_withholds']:.2f}  unclear {v['est_share_unclear']:.2f}")


if __name__ == "__main__":
    main()
