#!/usr/bin/env bash
# One-time install on a GPU cluster (Hedy/Juliet MesoNET, Jean Zay, LUMI-G) or on the Mac for the Claude API run.
# Usage: bash gpu/install.sh [vllm|anthropic]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install cryptography numpy hnswlib fastembed mem0ai pytest huggingface_hub
case "${1:-vllm}" in
  vllm) pip install vllm ;;              # NVIDIA (Hedy, Juliet, Jean Zay). On LUMI-G use the site's ROCm vLLM container instead.
  anthropic) pip install anthropic ;;
esac
python3 -m pytest -q tests
