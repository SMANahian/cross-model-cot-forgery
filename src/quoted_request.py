"""Exploratory lead, found after the results were read (results/POSTHOC_v6.md).

gpt-oss's real reasoning often quotes the user's question ('The user asks: "..."'). Six of
the 50 gpt-oss-style forgeries do this too. Do those six work better, and only on gpt-oss?
The six were not chosen in advance and differ from the others in more than this feature,
so this is a lead for a controlled test, not a finding.

The "share" asks how often a random set of 6 conditions gives a gap at least this large.
"gap_vs_other_styles" subtracts what the other four styles do on the same conditions, which
takes out how easy each condition is.

Usage:  python src/quoted_request.py      Output: results/quoted_request.json
"""
import collections
import json
import re

import numpy as np

from common import MODELS, ROOT, load_jsonl

QUOTED = re.compile(r"[\"“'‘][^\"”]{3,60}\?[\"”'’]")
TRACE_QUOTE = re.compile(r"[\"“][^\"”]{3,80}\?[\"”]")
N_SLOTS, N_DRAWS = 50, 200000


def main():
    rows = load_jsonl(ROOT / "results/scored.jsonl")
    stimuli = load_jsonl(ROOT / "results/stimuli.jsonl")
    slots = [s["slot"] for s in stimuli if s["column"] == "style:gpt-oss-20b" and QUOTED.search(s["text"])]
    mask = np.zeros(N_SLOTS, bool)
    mask[slots] = True

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per[(r["model"], r["column"], r["slot"])][r["item_id"]].append(r["hijacked"])

    def rates(model, column):
        return np.array([np.mean([np.mean(v) for v in per[(model, column, s)].values()]) for s in range(N_SLOTS)])

    out = {"quoted_slots": slots, "real_traces_quoting_a_question": {}, "targets": {}}
    for m in MODELS:
        traces = load_jsonl(ROOT / f"results/traces/{m}.jsonl")
        out["real_traces_quoting_a_question"][m] = f"{sum(bool(TRACE_QUOTE.search(t['reasoning'] or '')) for t in traces)}/{len(traces)}"

    rng = np.random.default_rng(0)
    draws = [rng.permutation(N_SLOTS)[:len(slots)] for _ in range(N_DRAWS)]
    for m in MODELS:
        own = rates(m, "style:gpt-oss-20b")
        rest = np.mean([rates(m, f"style:{y}") for y in MODELS if y != "gpt-oss-20b"], axis=0)
        gap = lambda v: v[mask].mean() - v[~mask].mean()
        excess = own - rest
        total = excess.sum()
        shares = [excess[d].mean() - (total - excess[d].sum()) / (N_SLOTS - len(slots)) for d in draws] if m == "gpt-oss-20b" else None
        out["targets"][m] = {"quoted": own[mask].mean(), "other_texts": own[~mask].mean(),
                             "same_conditions_other_styles": rest[mask].mean(), "other_conditions_other_styles": rest[~mask].mean(),
                             "gap": gap(own), "gap_vs_other_styles": gap(excess)}
        if shares is not None:
            out["targets"][m]["share_of_random_sets_at_least_as_large"] = float(np.mean(np.array(shares) >= gap(excess)))
    (ROOT / "results/quoted_request.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
