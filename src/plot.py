"""Stage 6c: the two heatmaps (proposal 5.8), from results/analysis.json.

  fig_hijack_matrix.png  raw hijack rate H, all columns (controls + model styles)
  fig_pairing.png        pairing effect P over the model-style columns

Usage:  python src/plot.py
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT


def heatmap(ax, est, lo, hi, rows, cols, title, cmap, vmin, vmax, fmt):
    est, lo, hi = map(np.array, (est, lo, hi))
    im = ax.imshow(est, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cm = plt.get_cmap(cmap)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("forged-reasoning style (column)")
    ax.set_ylabel("target model (row)")
    ax.set_title(title)
    for i in range(est.shape[0]):
        for j in range(est.shape[1]):
            v = est[i, j]
            r, g, b, _ = cm((v - vmin) / (vmax - vmin))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5  # text colour from cell brightness
            ax.text(j, i, f"{fmt(v)}\n[{fmt(lo[i, j])}, {fmt(hi[i, j])}]", ha="center", va="center",
                    fontsize=8, color="white" if dark else "black")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'' = 3-model figures, '_5m' = 5-model figures")
    tag = ap.parse_args().tag
    a = json.loads((ROOT / f"results/analysis{tag}.json").read_text())
    nice = lambda c: c.replace("style:", "") + (" style" if c.startswith("style:") else "")
    cols = [nice(c) for c in a["columns"]]

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2, 0.9 * len(a["targets"]) + 2))
    H = a["matrices"]["H"]
    im = heatmap(ax, H["est"], H["lo"], H["hi"], a["targets"], cols,
                 "Hijack rate (code revealed), 95% CI", "Reds", 0, 1, lambda v: f"{100 * v:.0f}%")
    fig.colorbar(im, ax=ax, fraction=0.03, format=lambda v, _: f"{100 * v:.0f}%")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/fig_hijack_matrix{tag}.png", dpi=200)

    P = a["matrices"]["P"]
    if not a["included_targets"]:
        print(f"-> results/fig_hijack_matrix{tag}.png (no pairing plot: every row is flat)")
        return
    style_cols = [nice(f"style:{m}") for m in a["styles"]]
    lim = max(0.05, float(np.nanmax(np.abs(P["est"]))))
    fig, ax = plt.subplots(figsize=(1.8 * len(style_cols) + 2, 0.9 * len(a["included_targets"]) + 2))
    im = heatmap(ax, P["est"], P["lo"], P["hi"], a["included_targets"], style_cols,
                 "Pairing effect in pp (H - row mean - col mean + grand mean), 95% CI", "RdBu_r", -lim, lim,
                 lambda v: f"{100 * v:+.0f}")
    fig.colorbar(im, ax=ax, fraction=0.03, format=lambda v, _: f"{100 * v:+.0f} pp")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/fig_pairing{tag}.png", dpi=200)
    print(f"-> results/fig_hijack_matrix{tag}.png, results/fig_pairing{tag}.png")


if __name__ == "__main__":
    main()
