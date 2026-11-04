# Pre-registration: 5-model matrix (v6)

Written 2026-11-03T23:44:28Z. This is after the 3-model v5 results were seen, and before any generation from the two new models.

## What changes
- Two more families join as both targets and style sources: **Olmo-3-7B-Think** (AI2) and **GLM-Z1-9B-0414** (Zhipu).
- Phi-4-reasoning is dropped: its chat template hard-codes its own system prompt and silently discards any system message, so our rule would never reach it.
- All v5 stimuli, items and outputs stay frozen. C1–C3 stand as already reported, on the 3-model matrix.
- New data: 150 traces per new model; 50 rewrites per new style column, made with the same prompt template and checks; every target runs the new columns, and each new target runs every column. Every new item gets 1 sample.

## Why
v5 left own-style specificity unresolved. C1 compares styles within one target only, so it can't separate "this model's own style" from "a generally strong style". gpt-oss-style text beat the other styles by about the same margin on Qwen3 (+15.1 pp) as on gpt-oss (+14.8 pp).

## Hypotheses
Tested on the 5 × 5 matrix over all 50 conditions × 20 phrasings, with 95% two-way cluster bootstrap CIs.

| | Hypothesis | Decision rule |
|---|---|---|
| D1 (H0) | Own-style effect: mean diagonal pairing effect > 0 | Supported if the CI on the raw scale excludes 0 |
| D2 (H2) | Symmetry: corr(P[X][Y], P[Y][X]) over the 10 pairs > 0 | Supported if the CI excludes 0 |
| D3 (H1) | Similarity: Spearman between symmetric pairing S[X,Y] and embedding similarity, over the 10 pairs | Exploratory; reported with its CI |

Reported alongside D1, because v5 showed one flat row can inflate the mean diagonal:
- each model's own diagonal cell
- the log-odds version
- the mean diagonal with each model left out in turn
- the observed diagonal's rank among all 120 assignments of styles to targets

Reported alongside D2: the original rule (every pair difference's CI includes 0, and at least one symmetric effect is clearly nonzero). Under pure noise, D2's correlation is expected to be about +0.06, not 0, because of the double-centering.

Sensitivity: everything is repeated on the 40 fresh conditions.

## Procedure
Each new model gets a 10-item smoke test. The full run proceeds whatever hijack rates the smoke test shows; it stops only for a template or parsing failure. Flat rows are handled by the existing exclusion rule.
