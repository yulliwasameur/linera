#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
step() { echo "=== $(date '+%F %T') $*"; }
step mem0_e2e;     python3 -I eval/mem0_e2e.py --out results/results_mem0.json > runs/mem0_e2e.out 2>&1
step canary_synth; python3 -I eval/canary.py --sessions 300 --out results/results_canary_synthetic.json > runs/canary.out 2>&1
rm -f results/results_scale.json
step scale;        python3 -I eval/scale.py --sizes 10000,100000 --regimes flat,chained --out results/results_scale.json > runs/scale.out 2>&1
step scale_1e6;    python3 -I eval/scale.py --sizes 1000000 --regimes flat --out results/results_scale.json >> runs/scale.out 2>&1
step ALL_DONE
