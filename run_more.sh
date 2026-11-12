#!/usr/bin/env bash
# Second sequential batch: waits for run_all.sh, then the windowed writer and the corrected canary replay.
set -u
cd "$(dirname "$0")"
step() { echo "=== $(date '+%F %T') $*"; }
while pgrep -f "run_all.sh" >/dev/null; do sleep 30; done
step canary_replay_fixed; python3 -I agent/replay.py --log runs/locomo_log.jsonl --parts canary --out results/results_locomo.json > runs/replay_locomo_canary2.out 2>&1
step build_window;   python3 -I agent/build_window.py --src runs/locomo_log.jsonl --convs 0,1 --w 4 --out runs/locomo_win4_log.jsonl > runs/build_window.out 2>&1
step replay_window;  python3 -I agent/replay.py --log runs/locomo_win4_log.jsonl --parts lineage,canary,utility --out results/results_locomo_win4.json > runs/replay_window.out 2>&1
step qa_window;      python3 -I agent/qa.py --log runs/locomo_win4_log.jsonl --convs 0,1 --n 100 --rates 0.01,0.1 --out results/results_qa_locomo_win4.json > runs/qa_window.out 2>&1
step ALL_DONE
