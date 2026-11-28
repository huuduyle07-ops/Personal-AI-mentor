#!/bin/sh
# Fine-tune TinyLlama on the intent task with MLX, then serve it through Ollama for the bot's benchmark.
# Run from the repository root. Needs an Apple Silicon Mac and the separate environment:
#   uv venv --python 3.11 experiments/intent_sft/.venv
#   uv pip install --python experiments/intent_sft/.venv/bin/python -r experiments/intent_sft/requirements.txt
set -e
PY=experiments/intent_sft/.venv/bin/python
A=experiments/intent_sft/artifacts
# The same pinned TinyLlama revision as Duy's experiment (experiments/pm_sft), loaded from the local cache.
REV=fe8a4ea1ffedaf415f4da2f062534de366a451e6
"$PY" -m huggingface_hub.cli.hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --revision $REV >/dev/null 2>&1 || true
BASE="$HOME/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/$REV"
mkdir -p "$A"

case "${1:-all}" in
data|all)
  uv run --locked python -m experiments.intent_sft.data ;;
esac
case "${1:-all}" in
train|all)
  ollama stop qwen3:1.7b qwen3:0.6b tinyllama >/dev/null 2>&1 || true   # free memory for training
  HF_HUB_OFFLINE=1 "$PY" -m mlx_lm lora -c experiments/intent_sft/lora.yaml --model "$BASE" 2>&1 | tee "$A/train.log" ;;
esac
case "${1:-all}" in
export|all)
  HF_HUB_OFFLINE=1 "$PY" -m mlx_lm fuse --model "$BASE" --adapter-path "$A/adapter" \
        --save-path "$A/fused"
  # 4-bit, like the stock tinyllama (q4_0); this Ollama version names its 4-bit import type int4.
  (cd experiments/intent_sft && ollama create tinyllama-intent -q int4 -f Modelfile) ;;
esac
