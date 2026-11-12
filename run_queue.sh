#!/usr/bin/env bash
# Single sequential queue (one LLM job at a time on 2 cores). Every step is cached or resumable.
set -u
cd "$(dirname "$0")"
step() { echo "=== $(date '+%F %T') $*"; }
# (done) canary_replay_fixed
step qa_locomo;      python3 -I agent/qa.py --log runs/locomo_log.jsonl --convs 0,1 --n 60 --rates 0.01,0.1 --out results/results_qa_locomo.json > runs/qa_locomo.out 2>&1
step build_window;   python3 -I agent/build_window.py --src runs/locomo_log.jsonl --convs 0,1 --w 4 --out runs/locomo_win4_log.jsonl > runs/build_window.out 2>&1
step replay_window;  python3 -I agent/replay.py --log runs/locomo_win4_log.jsonl --parts lineage,canary,utility --out results/results_locomo_win4.json > runs/replay_window.out 2>&1
step qa_window;      python3 -I agent/qa.py --log runs/locomo_win4_log.jsonl --convs 0,1 --n 60 --rates 0.01,0.1 --conds sound,sound_rd --out results/results_qa_locomo_win4.json > runs/qa_window.out 2>&1
step build_lme;      python3 -I agent/build_lme.py --data ../data/longmemeval_s_cleaned.json > runs/build_lme.out 2>&1
step replay_lme;     python3 -I agent/replay.py --log runs/lme_log.jsonl --parts lineage,utility --qfile runs/lme_questions.json --out results/results_lme.json > runs/replay_lme.out 2>&1
step qa_lme;         python3 -I agent/qa.py --log runs/lme_log.jsonl --qfile runs/lme_questions.json --convs $(python3 -c "print(','.join(map(str,range(24))))") --n 24 --rates 0.01,0.1 --out results/results_qa_lme.json > runs/qa_lme.out 2>&1
step mem0_e2e;       python3 -I eval/mem0_e2e.py --out results/results_mem0.json > runs/mem0_e2e.out 2>&1
step canary_synth;   python3 -I eval/canary.py --sessions 300 --out results/results_canary_synthetic.json > runs/canary.out 2>&1
step scale;          python3 -I eval/scale.py --sizes 10000,100000 --regimes flat,chained --out results/results_scale.json > runs/scale.out 2>&1
step scale_1e6;      python3 -I eval/scale.py --sizes 1000000 --regimes flat --out results/results_scale.json >> runs/scale.out 2>&1
step ALL_DONE
