#!/usr/bin/env bash
# Full campaign for one writer model. Everything is resumable and cached: re-running continues where it stopped.
# Environment: LF_BACKEND (vllm|anthropic), LF_MODEL (HF id, path, or Claude model id), TAG (short name for files).
# Example (GPU):    LF_BACKEND=vllm LF_MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b bash gpu/campaign.sh
# Example (Claude): LF_BACKEND=anthropic LF_MODEL=claude-haiku-5-5 TAG=haiku55 bash gpu/campaign.sh
set -euo pipefail   # stop at the first failing step (a failed step no longer reaches DONE)
cd "$(dirname "$0")/.."
: "${TAG:?set TAG}"; export LF_BACKEND LF_MODEL
ALL=0,1,2,3,4,5,6,7,8,9
step() { echo "=== $(date '+%F %T') $*"; }
mkdir -p runs results
step "A1 session writer, LoCoMo 10 conversations"
python3 -I agent/build_log.py --data ../data/locomo10.json --convs $ALL --out runs/locomo_${TAG}_log.jsonl
step "A2 four-turn window writer, LoCoMo 10 conversations"
python3 -I agent/build_window.py --src runs/locomo_${TAG}_log.jsonl --convs $ALL --w 4 --out runs/locomo_${TAG}_win4_log.jsonl
step "A3 LongMemEval-S, 25 questions per type (100)"
python3 -I agent/build_lme.py --data ../data/longmemeval_s_cleaned.json --per_type 25 --out runs/lme_${TAG}_log.jsonl --qout runs/lme_${TAG}_questions.json
step "B replays (canaries, attacks, lineage, retrieval) -- no LLM"
python3 -I agent/replay.py --log runs/locomo_${TAG}_log.jsonl --data ../data/locomo10.json --out results/results_locomo_${TAG}.json
python3 -I agent/replay.py --log runs/locomo_${TAG}_win4_log.jsonl --data ../data/locomo10.json --out results/results_locomo_${TAG}_win4.json
python3 -I agent/replay.py --log runs/lme_${TAG}_log.jsonl --parts lineage,utility --qfile runs/lme_${TAG}_questions.json --out results/results_lme_${TAG}.json
step "C question answering, all scored LoCoMo questions (n=2000 caps at the eligible set)"
python3 -I agent/qa.py --log runs/locomo_${TAG}_log.jsonl --convs $ALL --n 2000 --rates 0.01,0.1 --out results/results_qa_locomo_${TAG}.json
python3 -I agent/qa.py --log runs/locomo_${TAG}_win4_log.jsonl --convs $ALL --n 2000 --rates 0.01,0.1 --conds sound,sound_rd --out results/results_qa_locomo_${TAG}_win4.json
python3 -I agent/qa.py --log runs/lme_${TAG}_log.jsonl --qfile runs/lme_${TAG}_questions.json --convs $(python3 -c "print(','.join(map(str,range(100))))") --n 100 --rates 0.01,0.1 --out results/results_qa_lme_${TAG}.json
step "D mem0 end to end"
python3 -I eval/mem0_e2e.py --messages gpu/mem0_scenario_60.json --out results/results_mem0_${TAG}.json
step DONE
