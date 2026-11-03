"""Stage 1: collect real reasoning traces (the style sources).

Each model answers the same 150 prompts (100 neutral + 50 rule-decision) with NO
system prompt, and we keep its reasoning text. These traces are what the rewriter
imitates (proposal 3.3-3.4) and what the style-similarity measure compares
(proposal 4). Same prompts for every model, so content is held fixed.

Usage (on the pod):
    python src/collect_traces.py --model qwen3-8b --smoke   # 10 prompts, printed
    python src/collect_traces.py --model qwen3-8b           # all 150
"""
import argparse
import collections
import time

from common import (MAX_MODEL_LEN, MAX_TOKENS, MODELS, ROOT, decode_output,
                    load_jsonl, load_tokenizer, parse_output, render, write_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--smoke", action="store_true", help="10 diverse prompts, printed in full")
    args = ap.parse_args()
    cfg = MODELS[args.model]

    neutral = load_jsonl(ROOT / "data/neutral_prompts.jsonl")
    rules = load_jsonl(ROOT / "data/rule_prompts.jsonl")
    # Smoke test: every 20th neutral prompt + every 10th rule prompt (mixes all kinds).
    prompts = neutral[::20] + rules[::10] if args.smoke else neutral + rules

    from vllm import LLM, SamplingParams  # imported here so --help works without a GPU

    tok = load_tokenizer(cfg)
    llm = LLM(model=cfg["hf_id"], max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, seed=0)

    requests, params, rendered = [], [], []
    for i, p in enumerate(prompts):
        text, ids = render(tok, [{"role": "user", "content": p["prompt"]}], cfg)
        rendered.append(text)
        requests.append({"prompt_token_ids": ids})
        # Per-request seed => reproducible regardless of batching.
        params.append(SamplingParams(**cfg["sampling"], max_tokens=MAX_TOKENS,
                                     seed=10_000 + i, skip_special_tokens=False))

    t0 = time.time()
    outputs = llm.generate(requests, params)
    secs = time.time() - t0

    rows = []
    for p, text, out in zip(prompts, rendered, outputs):
        o = out.outputs[0]
        raw = decode_output(tok, o)
        reasoning, answer = parse_output(raw, cfg["format"])
        rows.append({
            "model": args.model, "id": p["id"], "kind": p["kind"], "prompt": p["prompt"],
            "rendered_prompt": text, "raw": raw, "reasoning": reasoning, "answer": answer,
            "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids),
        })

    name = f"{args.model}{'_smoke' if args.smoke else ''}.jsonl"
    write_jsonl(ROOT / "results/traces" / name, rows)

    # Summary
    finish = collections.Counter(r["finish_reason"] for r in rows)
    no_answer = sum(r["answer"] is None for r in rows)
    empty = sum(not r["reasoning"] for r in rows)
    words = [len(r["reasoning"].split()) for r in rows if r["reasoning"]]
    print(f"\n[{args.model}] {len(rows)} traces in {secs:.0f}s -> results/traces/{name}")
    print(f"  finish reasons: {dict(finish)} | no final answer: {no_answer} | empty reasoning: {empty}")
    if words:
        print(f"  reasoning words: median {sorted(words)[len(words) // 2]}, max {max(words)}")

    # Always show the rendered prompt (role tags!) and a few traces for a quick eyeball.
    print("\n===== RENDERED PROMPT (item 0) =====")
    print(rendered[0])
    shown = rows if args.smoke else [rows[0], rows[len(neutral) // 2], rows[-1]]
    if True:
        for r in shown:
            print(f"\n----- {r['id']} ({r['kind']}) | {r['finish_reason']} | {r['n_tokens']} tok -----")
            print("REASONING:", (r["reasoning"] or "")[:700])
            print("ANSWER:", (r["answer"] or "<none>")[:300])


if __name__ == "__main__":
    main()
