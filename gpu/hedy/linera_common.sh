# Sourced by the LinEra sbatch files on Hedy.
# Compute nodes only see /home (no /scratch): everything lives in $HOME/linera, unpacked on a compute node
# from $HOME/linera-bundle.tar (built on the login node from /scratch/$USER/linera). Writing many files to /home
# from the LOGIN node briefly counts 1 GiB per file against the home quota: never do that.
L=$HOME/linera
if [ ! -x "$L/venv/bin/python3" ]; then
  [ -s "$HOME/linera-bundle.tar" ] || { echo "MISSING $HOME/linera-bundle.tar"; exit 1; }
  echo "unpacking bundle $(date '+%F %T')"; tar -C "$HOME" -xf "$HOME/linera-bundle.tar" || exit 1
  echo "unpacked $(date '+%F %T')"
fi
mkdir -p "$L/logs"
module purge >/dev/null 2>&1 || true
module load python/3.12.4
# the venv was built under /scratch: activate it through PATH, not bin/activate (which hard-codes that path)
export VIRTUAL_ENV=$L/venv PATH=$L/venv/bin:$PATH
export HF_HOME=$L/hf FASTEMBED_CACHE_PATH=$L/fastembed HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MEM0_TELEMETRY=False
export LF_BACKEND=vllm
cd "$L/code"
echo "host $(hostname) python $(command -v python3) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) start $(date '+%F %T')"
python3 -c "import sys, vllm; print('prefix', sys.prefix, 'vllm', vllm.__version__)" || exit 1
