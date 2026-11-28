cd /Users/axelvalentino/Documents/Coding/Personal-AI-mentor-master
OUT=${1:-experiments/model_size/results}
for m in tinyllama cpu-tinyllama qwen3:0.6b cpu-qwen3-0.6b qwen3:1.7b cpu-qwen3-1.7b; do
  out="$OUT/$(echo $m | tr ':' '-').json"
  ollama stop tinyllama qwen3:0.6b qwen3:1.7b cpu-tinyllama cpu-qwen3-0.6b cpu-qwen3-1.7b >/dev/null 2>&1
  start=$(date +%s)
  uv run --locked python -m experiments.conversation.evaluate --mode live --model "$m" --timeout 60 \
     --scenarios experiments/conversation/heldout_v2.jsonl > "$out" 2>/dev/null
  ollama ps | grep -v NAME > "${out%.json}.ps.txt"
  echo "$m done in $(( $(date +%s) - start ))s"
done
