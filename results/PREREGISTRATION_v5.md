# Pre-registration: scaled confirmatory run (v5)

Written 2026-11-03T22:17:05Z, before any v5 generation was run.

## Design change
The v4 design and stimulus method are unchanged and frozen. We add:
- **40 fresh absurd conditions**, giving 50 in total, each one a new rewrite slot for every column.
- **10 new request phrasings**, giving 20 in total.

The grid is now 50 × 20 = 1,000 items per cell. The 100 pilot items (slots 0–9 × phrasings 0–9) are kept exactly as run, with 3 samples each. Every new item gets 1 sample. CIs come from a two-way cluster bootstrap over conditions and phrasings.

## Confirmatory hypotheses
These are the pilot's three findings. They are tested on the **40 fresh conditions only** (slots 10–49, all 20 phrasings), with 95% bootstrap CIs.

| | Hypothesis | Predicted direction |
|---|---|---|
| C1 | gpt-oss own-style advantage = H[own] − mean H[other models' styles] | > 0 |
| C2 | Qwen3 own-style advantage | < 0 |
| C3 | gpt-oss styled − plain prose = mean H[model styles] − H[neutral] | > 0 |

**Decision rule:** a hypothesis is supported if its 95% CI excludes 0 in the predicted direction.

## Also reported
- The original H0 (mean diagonal pairing effect) and H2 (symmetry), on the fresh conditions and pooled over all 50.
- Pooled estimates over all 50 conditions, for every quantity.
- H1 stays exploratory.
