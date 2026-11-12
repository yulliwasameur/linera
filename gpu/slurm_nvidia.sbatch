#!/bin/bash
# Template for NVIDIA clusters (MesoNET Hedy H100 / Juliet A100, Jean Zay H100).
# Fill the three lines marked CHECK with the values from the centre's documentation and your project.
#SBATCH --job-name=linera
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=runs/slurm-%j.out
##SBATCH --account=CHANGE_ME          # CHECK: project/account
##SBATCH --partition=CHANGE_ME        # CHECK: GPU partition name
##SBATCH --constraint=CHANGE_ME       # CHECK: only if the centre requires it (e.g. H100 nodes)
set -u
cd "$SLURM_SUBMIT_DIR"
module purge 2>/dev/null || true      # CHECK: load the centre's Python/CUDA modules if needed
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/../hf_cache}
LF_BACKEND=vllm LF_MODEL=${LF_MODEL:-Qwen/Qwen2.5-7B-Instruct} TAG=${TAG:-qwen7b} bash gpu/campaign.sh
