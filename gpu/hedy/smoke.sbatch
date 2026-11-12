#!/bin/bash
# LinEra -- short test on Hedy: Qwen2.5-7B writes the memory of LoCoMo conversation 0, then replay and 20 questions.
# Submit only with Yulliwas's agreement:  sbatch $HOME/linera-hedy/smoke.sbatch
#SBATCH --job-name=linera-smoke
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/%u/%x_%j.out
set -u
source $HOME/linera-hedy/linera_common.sh
export LF_MODEL=Qwen/Qwen2.5-7B-Instruct
T=smoke_qwen7b
mkdir -p runs results
python3 -I agent/build_log.py --data ../data/locomo10.json --convs 0 --out runs/locomo_${T}_log.jsonl || exit 1
python3 -I agent/replay.py --log runs/locomo_${T}_log.jsonl --data ../data/locomo10.json --out results/results_locomo_${T}.json || exit 1
python3 -I agent/qa.py --log runs/locomo_${T}_log.jsonl --convs 0 --n 20 --rates 0.1 --conds none,sound,sound_rd \
        --out results/results_qa_locomo_${T}.json || exit 1
echo "SMOKE OK $(date '+%F %T')"
