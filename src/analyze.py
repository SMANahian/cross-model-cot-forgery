"""Stage 6b: the pre-registered analysis (proposal 5).

H[X][C]  hijack rate of target X in column C (100 items each).
P[X][Y]  pairing effect over the model-style columns:
         H - rowmean - colmean + grandmean  (removes "X is easy to fool" and
         "style Y is strong everywhere"). Also on smoothed log-odds as a check.
CIs: two-way cluster bootstrap (B=2000). Each replicate resamples the 10 request
phrasings and the 10 rewrite slots, both shared by every cell. Slots are resampled
jointly across columns because each slot carries one absurd condition that is the
same in every column (v4), so a slot = (condition, each column's text for it).

Usage:  python src/analyze.py
Output: results/analysis.json (+ printed summary)
"""
import itertools
import json

import numpy as np

from common import MODELS, ROOT, load_jsonl

B, SEED, N = 2000, 0, 10


def pairing(H):
    """Double-centre a (targets x styles) matrix."""
    return H - H.mean(axis=1, keepdims=True) - H.mean(axis=0, keepdims=True) + H.mean()


def smoothed_logit(rate, n=100):
    p = (rate * n + 0.5) / (n + 1)
    return np.log(p / (1 - p))


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main():
    rows = load_jsonl(ROOT / "results/scored.jsonl")
    targets = [m for m in MODELS if any(r["model"] == m for r in rows)]
    styles = [m for m in MODELS if any(r["column"] == f"style:{m}" for r in rows)]
    columns = ["none", "neutral", "generic"] + [f"style:{m}" for m in styles]

    # (target, column) -> 10x10 array over (slot, phrasing) of the item's hijack rate,
    # averaged over its samples (3 per item in the full run).
    sums, cnts = {}, {}
    for r in rows:
        key = (r["model"], r["column"])
        sums.setdefault(key, np.zeros((N, N)))[r["slot"], r["phrasing"]] += r["hijacked"]
        cnts.setdefault(key, np.zeros((N, N)))[r["slot"], r["phrasing"]] += 1
    A = {k: np.where(cnts[k] > 0, sums[k] / np.maximum(cnts[k], 1), np.nan) for k in sums}

    def h_matrix(R=None, J=None):
        H = np.zeros((len(targets), len(columns)))
        for ci_, c in enumerate(columns):
            for ti, t in enumerate(targets):
                a = A[(t, c)]
                if R is not None:
                    a = a[np.ix_(R[c], J)]
                H[ti, ci_] = np.nanmean(a)
        return H

    style_idx = [columns.index(f"style:{m}") for m in styles]

    # Pre-registered exclusion: rows flat at the ceiling/floor can't show pairing.
    H0_point = h_matrix()
    flat = {t: bool(np.all(H0_point[ti, style_idx] > 0.95) or np.all(H0_point[ti, style_idx] < 0.05))
            for ti, t in enumerate(targets)}
    inc = [t for t in targets if not flat[t]]
    inc_idx = [targets.index(t) for t in inc]
    diag = [(inc.index(m), styles.index(m)) for m in inc if m in styles]
    pairs = [(x, y) for x, y in itertools.combinations(inc, 2) if x in styles and y in styles]

    sim_path = ROOT / "results/similarity.json"
    sim = json.loads(sim_path.read_text())["similarity"] if sim_path.exists() else {}
    sim_of = lambda x, y: (sim.get(f"{x}|{y}") or sim.get(f"{y}|{x}") or {}).get("all")

    def stats(H):
        """Every pre-registered quantity, from one hijack matrix."""
        out = {"H": H}
        Hs = H[np.ix_(inc_idx, style_idx)]
        empty = Hs.size == 0  # every row flat: nothing to double-centre
        for name, M in [("P", Hs if empty else pairing(Hs)),
                        ("P_logit", Hs if empty else pairing(smoothed_logit(Hs)))]:
            out[name] = M
            if diag:
                out[f"{name}_diag_mean"] = float(np.mean([M[i, j] for i, j in diag]))
            for x, y in pairs:
                pxy = M[inc.index(x), styles.index(y)]
                pyx = M[inc.index(y), styles.index(x)]
                out[f"{name}_S[{x},{y}]"] = (pxy + pyx) / 2
                out[f"{name}_D[{x},{y}]"] = pxy - pyx
        col = lambda c: H[:, columns.index(c)]
        style_mean = H[:, style_idx].mean(axis=1)
        for ti, t in enumerate(targets):
            out[f"forgery_minus_none[{t}]"] = float(H[ti, 1:].mean() - col("none")[ti])
            out[f"generic_minus_neutral[{t}]"] = float(col("generic")[ti] - col("neutral")[ti])
            out[f"stylemean_minus_generic[{t}]"] = float(style_mean[ti] - col("generic")[ti])
            if t in styles:
                own = H[ti, columns.index(f"style:{t}")]
                others = [H[ti, columns.index(f"style:{m}")] for m in styles if m != t]
                out[f"own_minus_generic[{t}]"] = float(own - col("generic")[ti])
                out[f"own_advantage[{t}]"] = float(own - np.mean(others))
            out[f"styled_minus_neutral[{t}]"] = float(style_mean[ti] - col("neutral")[ti])
        return out

    point = stats(H0_point)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(B):
        J = rng.integers(0, N, N)
        slots = rng.integers(0, N, N)
        R = {c: slots for c in columns}  # same slots for every column (paired conditions)
        boots.append(stats(h_matrix(R, J)))

    scalar_keys = [k for k, v in point.items() if np.isscalar(v)]
    summary = {k: {"est": float(point[k]), "ci95": ci([b[k] for b in boots])} for k in scalar_keys}
    mats = {}
    for name in ["H", "P", "P_logit"]:
        stack = np.stack([b[name] for b in boots])
        mats[name] = {"est": point[name].tolist(),
                      "lo": np.percentile(stack, 2.5, axis=0).tolist(),
                      "hi": np.percentile(stack, 97.5, axis=0).tolist()}

    h1 = None
    if pairs and all(sim_of(x, y) is not None for x, y in pairs):
        S = [point[f"P_S[{x},{y}]"] for x, y in pairs]
        sims = [sim_of(x, y) for x, y in pairs]
        h1 = {"pairs": [f"{x}|{y}" for x, y in pairs], "S": S, "similarity": sims,
              "most_similar_pair": f"{pairs[int(np.argmax(sims))][0]}|{pairs[int(np.argmax(sims))][1]}",
              "largest_S_pair": f"{pairs[int(np.argmax(S))][0]}|{pairs[int(np.argmax(S))][1]}",
              "spearman": spearman(np.array(S), np.array(sims)) if len(pairs) >= 4 else None}

    result = {"targets": targets, "styles": styles, "columns": columns, "flat_rows": flat,
              "included_targets": inc, "B": B, "matrices": mats, "scalars": summary, "H1": h1}
    (ROOT / "results/analysis.json").write_text(json.dumps(result, indent=2))

    # ---- printed summary
    print("Hijack rate H in % [95% CI], rows = targets:")
    print(" " * 14 + "".join(f"{c.replace('style:', 'S:')[:19]:>20s}" for c in columns))
    for ti, t in enumerate(targets):
        H = mats["H"]
        cells = "".join(f"{100 * H['est'][ti][j]:6.1f} [{100 * H['lo'][ti][j]:4.1f},{100 * H['hi'][ti][j]:5.1f}]".rjust(20)
                        for j in range(len(columns)))
        print(f"{t:14s}{cells}{'   (flat: excluded)' if flat[t] else ''}")
    print("\nKey statistics (estimate [95% CI]):")
    for k in scalar_keys:
        s = summary[k]
        print(f"  {k:48s} {s['est']:+.3f}  [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    if h1:
        print("\nH1 (exploratory):", json.dumps(h1, indent=1))
    print("-> results/analysis.json")


if __name__ == "__main__":
    main()
