#!/bin/bash
# LinEra -- rerun of the one campaign step that failed in job 39809: LongMemEval QA (100 questions, forget 1% and 10%).
# Cause of the failure: cache file name too long in agent/qa.py (fixed 12/11/2026). Same command as gpu/campaign.sh line 25.
# Submit only with Yulliwas's agreement:  sbatch $HOME/linera-hedy/qa_lme.sbatch
#SBATCH --job-name=linera-qalme
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/%x_%j.out
set -u
source $HOME/linera-hedy/linera_common.sh
export LF_MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b
python3 -I agent/qa.py --log runs/lme_${TAG}_log.jsonl --qfile runs/lme_${TAG}_questions.json \
        --convs $(python3 -c "print(','.join(map(str,range(100))))") --n 100 --rates 0.01,0.1 \
        --out results/results_qa_lme_${TAG}.json || exit 1
echo "QA LME OK $(date '+%F %T')"
