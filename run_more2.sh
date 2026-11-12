#!/usr/bin/env bash
# Ablation of the writer window and second model family (CPU, sequential).
set -u; cd "$(dirname "$0")"
step() { echo "=== $(date '+%F %T') $*"; }
for w in 8 16 2; do
  step build_window_$w; python3 -I agent/build_window.py --src runs/locomo_log.jsonl --convs 0,1 --w $w --out runs/locomo_win${w}_log.jsonl > runs/build_window$w.out 2>&1
  step replay_window_$w; python3 -I agent/replay.py --log runs/locomo_win${w}_log.jsonl --parts lineage,canary --out results/results_locomo_win$w.json > runs/replay_window$w.out 2>&1
done
for w in 8 16; do
  step qa_window_$w; python3 -I agent/qa.py --log runs/locomo_win${w}_log.jsonl --convs 0,1 --n 60 --rates 0.01,0.1 --conds sound --out results/results_qa_locomo_win$w.json > runs/qa_window$w.out 2>&1
done
step build_llama; LF_MODEL=../models/llama-3.2-3b-instruct-q4_k_m.gguf python3 -I agent/build_log.py --data ../data/locomo10.json --convs 0,1 --out runs/locomo_llama3b_log.jsonl > runs/build_llama.out 2>&1
step replay_llama; python3 -I agent/replay.py --log runs/locomo_llama3b_log.jsonl --parts lineage,canary,utility --out results/results_locomo_llama3b.json > runs/replay_llama.out 2>&1
step qa_llama_1; LF_MODEL=../models/llama-3.2-3b-instruct-q4_k_m.gguf python3 -I agent/qa.py --log runs/locomo_llama3b_log.jsonl --convs 0,1 --n 60 --rates 0.01 --conds sound,sound_rd --out results/results_qa_locomo_llama3b.json > runs/qa_llama1.out 2>&1
step qa_llama_10; LF_MODEL=../models/llama-3.2-3b-instruct-q4_k_m.gguf python3 -I agent/qa.py --log runs/locomo_llama3b_log.jsonl --convs 0,1 --n 60 --rates 0.1 --conds sound --out results/results_qa_locomo_llama3b.json > runs/qa_llama10.out 2>&1
step radius; python3 -I eval/radius.py --out results/results_radius.json > runs/radius.out 2>&1
step ALL_DONE
