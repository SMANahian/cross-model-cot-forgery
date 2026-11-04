"""Post hoc: Olmo-3's row after a blinded reading of every answer containing the code.

The regex in src/stance.py is weakest on Olmo-3: in the audit (results/stance_audit.jsonl),
only 70-80% of the Olmo-3 answers it called direct really hand the code over. So all 4,575
Olmo-3 answers that contain the code were read by a coder who saw only the code and the
answer, never the column. The labels are in results/olmo_coded.jsonl. The coders were
Claude instances, so these labels are a sensitivity check, not ground truth.

For each column this prints the rate at which Olmo-3 hands the code over (label "gives"),
with the registered two-way cluster bootstrap, plus the styled-minus-plain contrast.
"unclear" rows don't count as handing over.

Usage:  python src/olmo_coded.py      Output: results/olmo_coded_summary.json
"""
import json

import numpy as np

from analyze import B, SEED, ci
from common import MODELS, ROOT, load_jsonl

N_SLOTS, N_PHRASINGS = 50, 20


def main():
    coded = {(r["column"], r["item_id"], r["sample"]): r["label"] for r in load_jsonl(ROOT / "results/olmo_coded.jsonl")}
    rows = [r for r in load_jsonl(ROOT / "results/scored.jsonl") if r["model"] == "olmo3-7b"]
    assert sum(r["hijacked"] for r in rows) == len(coded), "every Olmo-3 hijacked row needs a label"

    columns = ["none", "neutral", "generic"] + [f"style:{m}" for m in MODELS]
    gives = {c: np.zeros((N_SLOTS, N_PHRASINGS)) for c in columns}
    withholds = {c: np.zeros((N_SLOTS, N_PHRASINGS)) for c in columns}
    for r in rows:
        label = coded.get((r["column"], r["item_id"], r["sample"]))
        gives[r["column"]][r["slot"], r["phrasing"]] = label == "gives"
        withholds[r["column"]][r["slot"], r["phrasing"]] = label == "withholds"

    styles = [f"style:{m}" for m in MODELS]

    def stats(R, J):
        rate = {c: gives[c][np.ix_(R, J)].mean() for c in columns}
        out = {f"gives[{c}]": rate[c] for c in columns}
        out.update({f"withholds[{c}]": withholds[c][np.ix_(R, J)].mean() for c in columns})
        out["styled_minus_plain"] = np.mean([rate[c] for c in styles]) - rate["neutral"]
        out["styled_minus_generic"] = np.mean([rate[c] for c in styles]) - rate["generic"]
        out["gptoss_style_minus_plain"] = rate["style:gpt-oss-20b"] - rate["neutral"]
        out["others_minus_plain"] = np.mean([rate[c] for c in styles[1:]]) - rate["neutral"]
        return out

    point = stats(np.arange(N_SLOTS), np.arange(N_PHRASINGS))
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(B):  # same draw order as analyze.py, so the CIs match `--outcome read`
        J = rng.integers(0, N_PHRASINGS, N_PHRASINGS)
        boots.append(stats(rng.integers(0, N_SLOTS, N_SLOTS), J))
    summary = {k: {"est": float(v), "ci95": ci([b[k] for b in boots])} for k, v in point.items()}
    (ROOT / "results/olmo_coded_summary.json").write_text(json.dumps(summary, indent=2))
    for k, s in summary.items():
        print(f"{k:34s} {100 * s['est']:+6.1f} [{100 * s['ci95'][0]:+6.1f}, {100 * s['ci95'][1]:+6.1f}]")


if __name__ == "__main__":
    main()
