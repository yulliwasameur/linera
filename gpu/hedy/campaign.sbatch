#!/bin/bash
# LinEra -- extended campaign E1-E5 on Hedy for one writer model (10 LoCoMo conversations, LongMemEval 100 q., mem0 60 msgs).
# Resumable: resubmitting the same command continues where it stopped.
# Submit only with Yulliwas's agreement, after smoke.sbatch passed:
#   sbatch --export=ALL,LF_MODEL=Qwen/Qwen2.5-7B-Instruct,TAG=qwen7b $HOME/linera-hedy/campaign.sbatch
#SBATCH --job-name=linera
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/%u/%x_%j.out
set -u
source $HOME/linera-hedy/linera_common.sh
export LF_MODEL=${LF_MODEL:-Qwen/Qwen2.5-7B-Instruct} TAG=${TAG:-qwen7b}
bash gpu/campaign.sh
