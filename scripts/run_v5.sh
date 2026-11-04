#!/usr/bin/env bash
# v5 scaled confirmatory run (results/PREREGISTRATION_v5.md): 40 fresh absurd conditions
# and 10 new request phrasings on top of the frozen v4 pilot, 1 sample per new item.
# Every step is resumable (--extend skips what already exists).
# Usage (on the pod, from the repo root):  bash scripts/run_v5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODELS="gpt-oss-20b qwen3-8b r1-llama-8b"

echo "[$(date +%T)] stimuli (extend to 50 conditions)"
python src/make_stimuli.py --extend > logs/stimuli_v5.log 2>&1
echo "v5 stimuli frozen at $(date -u +%FT%TZ): $(sha256sum results/stimuli.jsonl | cut -c1-16)" | tee -a results/STIMULI_FROZEN.txt
echo "[$(date +%T)] items (50 x 20)"
python src/build_items.py > logs/items_v5.log 2>&1
for m in $MODELS; do
  echo "[$(date +%T)] attack $m (new items, 1 sample)"
  python src/run_attack.py --model "$m" --extend --samples 1 > "logs/full_v5_$m.log" 2>&1
done
echo "[$(date +%T)] score / analyze (all, fresh, pilot) / plot"
python src/score.py > logs/score_v5.log 2>&1
for s in all fresh pilot; do python src/analyze.py --slots "$s" > "logs/analyze_v5_$s.log" 2>&1; done
python src/plot.py > logs/plot_v5.log 2>&1
echo "[$(date +%T)] V5 DONE"
