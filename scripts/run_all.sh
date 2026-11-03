#!/usr/bin/env bash
# Everything after setup: stages 1-4, the 10-item smoke test per target, then the full
# run (100 items x 3 samples per cell) and the analysis. Resumable: a stage is skipped
# when its output exists (delete the output to redo it). Logs go to logs/<step>.log.
# Usage (on the pod, from the repo root):  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODELS="gpt-oss-20b qwen3-8b r1-llama-8b"

step() {  # step <output-file> <log-name> <command...>
  local out=$1 log=$2; shift 2
  if [ -e "$out" ]; then echo "[$(date +%T)] skip $log ($out exists)"; return; fi
  echo "[$(date +%T)] run  $log"
  "$@" > "logs/$log.log" 2>&1
}

for m in $MODELS; do step "results/traces/$m.jsonl" "traces_$m" python src/collect_traces.py --model "$m"; done
step results/similarity.json similarity python src/similarity.py
step results/stimuli.jsonl   stimuli    python src/make_stimuli.py
step results/items.jsonl     items      python src/build_items.py
for m in $MODELS; do step "results/attack/${m}_smoke.jsonl" "smoke_$m" python src/run_attack.py --model "$m" --smoke; done
for m in $MODELS; do step "results/attack/$m.jsonl" "full_$m" python src/run_attack.py --model "$m" --samples 3; done
echo "[$(date +%T)] run  score/analyze/plot"
python src/score.py > logs/score.log 2>&1
python src/analyze.py > logs/analyze.log 2>&1
python src/plot.py > logs/plot.log 2>&1
echo "[$(date +%T)] ALL DONE"
