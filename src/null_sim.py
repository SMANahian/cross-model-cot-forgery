"""Post hoc calibration of the symmetry statistics (not pre-registered).

What would D1 and D2 look like if there were NO own-style effect and NO pair effect?
The null is purely additive on the log-odds scale: target effect + style effect +
a condition effect. Target and style effects are fitted so the null reproduces the
observed row and column hijack rates, and the size of the condition effect is fitted to
the observed spread of hijack rates between conditions. Each simulated dataset has the
same size as the real one, and the statistics are recomputed on it on both scales.

Two noise structures are run. "shared": a condition moves every cell together.
"independent": each cell draws its own condition effect, as if every text were unrelated
to the same condition's text in other cells. These are two reference points, not bounds:
real data can have structure neither one has, so read them as a rough range.

Why this is needed: rates compress near 0 and 1, so an additive log-odds matrix is not
additive on the raw scale. Raw D2's value under this null can therefore sit far from zero,
in either direction, depending on how row and column effects line up. Log-odds D2 stays much
closer to its noise value. So raw D2 has to be compared with this matrix's own null.

What this can't do: the null has no own-style term, so it can't rule out D2 being driven by
uneven own-style effects. Only the off-diagonal statistic (D2off) is protected from that.
It also has no term for targets differing in how strongly a style moves them, and the real
matrix isn't additive, so a small D1 share shows non-additivity, not an own-style effect.
"shared" noise fits these data poorly for D1, so quote D1 shares under "independent" only.

Usage:  python src/null_sim.py      Output: results/null_sim.json (+ printed summary)
"""
import itertools
import json

import numpy as np

from analyze import offdiag_fit, pair_symmetry, pairing, smoothed_logit
from common import ROOT, load_jsonl

N_DRAWS = 2000
SD_GRID = np.arange(0.25, 4.01, 0.25)
NAMES = ["D1_raw", "D2_raw", "D2off_raw", "D1_logit", "D2_logit", "D2off_logit"]


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def fit_additive(H, sd):
    """Additive log-odds whose implied rates match H's row and column means when each
    condition shifts a cell's log-odds by Normal(0, sd)."""
    z = np.random.default_rng(1).normal(0, sd, 4000)
    a, b = np.zeros(H.shape[0]), np.zeros(H.shape[1])
    implied = lambda: expit(a[:, None, None] + b[None, :, None] + z).mean(axis=2)
    for _ in range(100):
        a += logit(H.mean(axis=1)) - logit(implied().mean(axis=1))
        b += logit(H.mean(axis=0)) - logit(implied().mean(axis=0))
    return a[:, None] + b[None, :]


def statistics(M, pairs):
    """D1, D2 (both as registered) and the off-diagonal symmetry statistic, on one scale."""
    P = pairing(M)
    xy, yx = [P[x, y] for x, y in pairs], [P[y, x] for x, y in pairs]
    return float(np.mean(np.diag(P))), float(np.corrcoef(xy, yx)[0, 1]), pair_symmetry(offdiag_fit(M))[1]


def fit_condition_sd(H, slot_var, n_s, n_p, rng):
    """The condition SD whose simulated between-condition variance best matches the data."""
    gaps = []
    for sd in SD_GRID:
        fitted = fit_additive(H, sd)
        v = []
        for _ in range(100):
            p = expit(fitted[:, :, None] + rng.normal(0, sd, n_s))
            v.append((rng.binomial(n_p, p) / n_p).var(axis=2).mean())
        gaps.append(abs(np.mean(v) - slot_var))
    return float(SD_GRID[int(np.argmin(gaps))])


def simulate(H, sd, n_s, n_p, pairs, shared, rng):
    fitted = fit_additive(H, sd)
    sims = []
    for _ in range(N_DRAWS):
        cond = rng.normal(0, sd, n_s if shared else (*H.shape, n_s))
        p = expit(fitted[:, :, None] + cond)
        Hs = (rng.binomial(n_p, p) / n_p).mean(axis=2)
        sims.append(statistics(Hs, pairs) + statistics(smoothed_logit(Hs, n_s * n_p), pairs))
    return np.array(sims)


def calibrate(H, slot_var, n_s, n_p):
    pairs = list(itertools.combinations(range(H.shape[0]), 2))
    rng = np.random.default_rng(0)
    sd = fit_condition_sd(H, slot_var, n_s, n_p, rng)
    obs = statistics(H, pairs) + statistics(smoothed_logit(H, n_s * n_p), pairs)
    out = {"condition_sd": sd, "n_draws": N_DRAWS, "observed": dict(zip(NAMES, obs))}
    for label, shared in [("shared", True), ("independent", False)]:
        sims = simulate(H, sd, n_s, n_p, pairs, shared, rng)
        out[label] = {k: {"null_mean": float(sims[:, i].mean()), "null_sd": float(sims[:, i].std()),
                          "share_at_least_observed": float((sims[:, i] >= obs[i]).mean())}
                      for i, k in enumerate(NAMES)}
    return out


def main():
    a = json.loads((ROOT / "results/analysis_5m.json").read_text())
    models, styles = a["included_targets"], a["styles"]
    assert models == styles, "a flat row was excluded: pairs below assume a square matrix"
    cols = [a["columns"].index(f"style:{m}") for m in styles]
    rows_i = [a["targets"].index(m) for m in models]
    H = np.array(a["matrices"]["H"]["est"])[np.ix_(rows_i, cols)]

    # Hijack rate of each (target, style, condition), for the between-condition spread.
    per = {}
    for r in load_jsonl(ROOT / "results/scored.jsonl"):
        if r["column"].startswith("style:"):
            n, h = per.get((r["model"], r["column"], r["slot"]), (0, 0))
            per[(r["model"], r["column"], r["slot"])] = (n + 1, h + r["hijacked"])
    n_s, n_p = a["n_slots"], a["n_phrasings"]
    slot_var = float(np.mean([np.var([per[(m, f"style:{y}", s)][1] / per[(m, f"style:{y}", s)][0]
                                      for s in range(n_s)]) for m in models for y in styles]))

    out = calibrate(H, slot_var, n_s, n_p)
    (ROOT / "results/null_sim.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
