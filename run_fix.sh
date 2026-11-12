#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
step() { echo "=== $(date '+%F %T') $*"; }
while pgrep -f run_more2.sh >/dev/null; do sleep 20; done
step qa_llama_common_sample
rm -f results/results_qa_locomo_llama3b.json
LF_MODEL=../models/llama-3.2-3b-instruct-q4_k_m.gguf python3 -I agent/qa.py --log runs/locomo_llama3b_log.jsonl --convs 0,1 --n 60 --rates 0.01,0.1 --conds sound,sound_rd --rd_rates 0.01 --out results/results_qa_locomo_llama3b.json > runs/qa_llama_common.out 2>&1
step session_conv01_canaries
python3 - <<'PY'
import json
with open("runs/locomo_c01_log.jsonl", "w") as f:
    for l in open("runs/locomo_log.jsonl"):
        if json.loads(l)["conv"] in (0, 1): f.write(l)
PY
python3 -I agent/replay.py --log runs/locomo_c01_log.jsonl --parts canary --out results/results_locomo_c01.json > runs/replay_c01.out 2>&1
step radius; python3 -I eval/radius.py --out results/results_radius.json > runs/radius.out 2>&1
step ALL_DONE
