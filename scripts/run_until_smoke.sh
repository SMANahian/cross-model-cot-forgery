#!/usr/bin/env bash
# Stages 1-4 plus the 10-item smoke test per target, then STOP for review before
# the full attack run (proposal 3.6). Resumable: a stage is skipped when its output
# already exists (delete the output to redo it). Each step logs to logs/<step>.log.
# Usage (on the pod, from the repo root):  bash scripts/run_until_smoke.sh
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
echo "[$(date +%T)] READY FOR REVIEW (full run not started)"
