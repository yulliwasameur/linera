# Extended campaign on a GPU (Qwen2.5-7B-Instruct, vLLM)

The numbers of the extended campaign (Section "A larger writer on all conversations" of the paper) were produced on
one NVIDIA H100 of the Hedy cluster (Centre de Calcul Intensif d'Aix-Marseille, MesoNET) with vLLM 0.33.

1. On the cluster, in this folder: `bash gpu/install.sh vllm`
2. Put the data in `../data/`: `locomo10.json` (github.com/snap-research/locomo) and
   `longmemeval_s_cleaned.json` (huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
3. Fill the `CHECK` lines (account, partition, constraint) of `gpu/slurm_nvidia.sbatch`, then
   `sbatch --export=ALL,LF_MODEL=Qwen/Qwen2.5-7B-Instruct,TAG=qwen7b gpu/slurm_nvidia.sbatch`
   or, without Slurm: `LF_BACKEND=vllm LF_MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b bash gpu/campaign.sh`
4. `gpu/hedy/` holds the exact job scripts used on Hedy (set `--account` to your own project).
5. Afterwards: `python3 -I eval/radius.py` and `python3 -I eval/make_tables.py --res results --tex <manuscript dir>`.

Every step is cached and resumable: re-running the same command continues where it stopped.
