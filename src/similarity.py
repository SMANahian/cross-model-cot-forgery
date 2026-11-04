"""Stage 2: style similarity between models, from real traces only (proposal 4).

Measured BEFORE any attack results exist, so it can't be tuned to fit them.
  sim(X, Y) = mean over shared prompts of cos(emb(X's trace), emb(Y's trace)).
Every model answered the same prompts, so content is held fixed and differences
mostly reflect how each model writes. Reported over all 150 prompts and over the
50 rule-decision prompts only.

Sanity check: can the embedding tell the models apart at all? For each prompt,
classify each model's trace by its nearest model centroid, where the centroids
exclude that prompt (leave-one-prompt-out). Chance = 1 / number of models.

Usage:  python src/similarity.py [--models gpt-oss-20b qwen3-8b r1-llama-8b]
"""
import argparse
import itertools
import json

import numpy as np

from common import ROOT, load_jsonl
from embed import EMBEDDER, embed_texts, load_embedder

DEFAULT_MODELS = ["gpt-oss-20b", "qwen3-8b", "r1-llama-8b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", default="results/similarity.json")
    args = ap.parse_args()
    models = args.models

    traces = {m: {r["id"]: r for r in load_jsonl(ROOT / f"results/traces/{m}.jsonl")}
              for m in models}
    # Prompts where every model produced non-empty reasoning.
    ids = sorted(set.intersection(*[{i for i, r in t.items() if r["reasoning"]}
                                    for t in traces.values()]))
    emb = load_embedder()
    E = {m: embed_texts(emb, [traces[m][i]["reasoning"] for i in ids]) for m in models}
    is_rule = np.array([i.startswith("rule") for i in ids])

    sims = {}
    for x, y in itertools.combinations(models, 2):
        cos = (E[x] * E[y]).sum(axis=1)  # paired: same prompt, two models
        sims[f"{x}|{y}"] = {"all": float(cos.mean()), "rule": float(cos[is_rule].mean())}

    confusion = {m: {n: 0 for n in models} for m in models}
    for k in range(len(ids)):
        keep = np.arange(len(ids)) != k
        cents = {m: E[m][keep].mean(axis=0) for m in models}
        cents = {m: c / np.linalg.norm(c) for m, c in cents.items()}  # unit length: compare direction only
        for m in models:
            pred = max(models, key=lambda n: float(E[m][k] @ cents[n]))
            confusion[m][pred] += 1
    correct = sum(confusion[m][m] for m in models)
    total = sum(sum(row.values()) for row in confusion.values())

    out = {
        "embedder": EMBEDDER, "models": models, "n_prompts": len(ids),
        "n_rule_prompts": int(is_rule.sum()), "similarity": sims,
        "nearest_centroid_accuracy": correct / total, "chance": 1 / len(models),
        "confusion": confusion,
    }
    path = ROOT / args.out
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
