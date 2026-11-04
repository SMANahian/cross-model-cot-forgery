#!/usr/bin/env bash
# v6 5-model matrix (results/PREREGISTRATION_v6.md): adds Olmo-3-7B-Think and GLM-Z1-9B.
# Stages, each resumable. Usage (on the pod, from the repo root):
#   bash scripts/run_v6.sh traces   # 150 real traces per new model
#   bash scripts/run_v6.sh prep     # similarity, new style columns, items, smoke tests
#   bash scripts/run_v6.sh full     # full attack runs, scoring, analysis, figures
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
OLD="gpt-oss-20b qwen3-8b r1-llama-8b"
NEW="olmo3-7b glm-z1-9b"

case "$1" in
  traces)
    for m in $NEW; do
      [ -e "results/traces/$m.jsonl" ] || python src/collect_traces.py --model "$m" > "logs/traces_$m.log" 2>&1
    done ;;
  prep)
    python src/similarity.py --models $OLD $NEW --out results/similarity_5m.json > logs/similarity_5m.log 2>&1
    python src/make_stimuli.py --models $OLD --add-models $NEW > logs/stimuli_v6.log 2>&1
    echo "v6 stimuli frozen at $(date -u +%FT%TZ): $(sha256sum results/stimuli.jsonl | cut -c1-16)" | tee -a results/STIMULI_FROZEN.txt
    python src/build_items.py > logs/items_v6.log 2>&1
    for m in $NEW; do python src/run_attack.py --model "$m" --smoke > "logs/smoke_$m.log" 2>&1; done ;;
  full)
    for m in $OLD $NEW; do
      echo "[$(date +%T)] attack $m"
      python src/run_attack.py --model "$m" --extend --samples 1 > "logs/full_v6_$m.log" 2>&1
    done
    python src/score.py > logs/score_v6.log 2>&1
    for s in all fresh pilot; do python src/analyze.py --slots "$s" > "logs/analyze_v6_3m_$s.log" 2>&1; done
    for s in all fresh; do python src/analyze.py --matrix 5 --slots "$s" > "logs/analyze_v6_5m_$s.log" 2>&1; done
    python src/plot.py > logs/plot_v6.log 2>&1
    python src/plot.py --tag _5m >> logs/plot_v6.log 2>&1 ;;
esac
echo "[$(date +%T)] $1 done"
