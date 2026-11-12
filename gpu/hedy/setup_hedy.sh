#!/bin/bash
# LinEra -- one-time setup on the Hedy LOGIN node (it has Internet; compute nodes may not).
# Everything lives in /scratch/$USER/linera (BeeGFS), NOT in $HOME: on /home, files written from the login
# node briefly count 1 GiB each against the home quota.
# Usage (detached):  nohup bash /scratch/$USER/linera/hedy/setup_hedy.sh > /scratch/$USER/linera/logs/setup.log 2>&1 &
set -euo pipefail
L=/scratch/$USER/linera
cd "$L"; mkdir -p logs data hf fastembed pip-cache
log(){ echo "[setup $(date '+%F %T')] $*"; }
export PIP_CACHE_DIR=$L/pip-cache HF_HOME=$L/hf FASTEMBED_CACHE_PATH=$L/fastembed

module load python/3.12.4
log "python $(python3 --version 2>&1)"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q "vllm==0.33.0" "mem0ai==2.2.1" "fastembed==0.9.1" "hnswlib==0.8.0" "qdrant-client==1.20.0" \
               cryptography pytest huggingface_hub
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)"

# data: LoCoMo is copied from the Mac (same file as the Mac runs); LongMemEval-S from Hugging Face
[ -s data/locomo10.json ] || { log "MISSING data/locomo10.json (copy it from the Mac)"; exit 1; }
[ -s data/longmemeval_s_cleaned.json ] || python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
shutil.copy(p, "data/longmemeval_s_cleaned.json"); print("longmemeval ok")
PY
sha256sum data/*.json

# models: writer 7B + embedder (pre-fetched so that jobs can run offline)
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.jinja"])
print("writer", p)
from fastembed import TextEmbedding
TextEmbedding("BAAI/bge-small-en-v1.5"); print("embedder ok")
PY

cd code && python -m pytest -q tests 2>&1 | tail -3
log "SETUP DONE"
