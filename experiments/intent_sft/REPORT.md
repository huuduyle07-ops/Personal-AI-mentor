# Fine-tuning TinyLlama for the bot's intent task

Run date: 2026-11-28, on an Apple M3 MacBook with 8 GB RAM. This continues the idea of Duy's experiment in [pm_sft](../pm_sft/README.md): can a small model become good at one narrow job through fine-tuning, instead of switching to a bigger model?

## Question

The bot sends every message its rules can't handle to a local language model. The model labels the message with intents such as `REPORT_PROGRESS {minutes}`, `REPORT_OBSTACLE {reason}` or `CLARIFY`. On the frozen 45-message test set, stock TinyLlama (1.1B parameters) scored 0.156, below rules alone. qwen3:1.7b scored 0.778 and became the default ([model size report](../model_size/REPORT.md)).

This experiment asks whether fine-tuning closes that gap.

## Method

**Data** ([data.py](data.py))
- 1,243 messages generated from templates written for this experiment, in English (671), Korean (298) and Vietnamese (274). Each is labelled by construction, following the conventions in the production prompt.
- Split 1,119 for training and 124 for validation.
- Three filters:
  - Every label must pass the bot's own validation and evidence guard, so the model never learns an answer the bot would reject.
  - 54 generated messages that were equal or more than 80% similar (character level) to any message in the three evaluation sets were removed.
  - Messages the rules already answer were mostly dropped.
- Tests: [test_data.py](test_data.py).

**Training** ([lora.yaml](lora.yaml))
- Base: TinyLlama-1.1B-Chat-v1.0, at the same pinned revision as Duy's run.
- Trained with Apple MLX (mlx-lm 0.32.2), using LoRA:
  - rank 16 on all attention and MLP projections in all 22 layers;
  - 12.6M trainable parameters, 1.1% of the model; the base weights stay frozen.
- Optimizer: Adam at 1e-4 with 30 warm-up steps and cosine decay, batch size 4, 300 steps (about one pass over the data).
- Each example uses exactly the production system prompt (`INTENT_PROMPT`, hash `d6f2449c8b8e`) and the same chat template Ollama uses for TinyLlama.
- The loss is computed only on the answer, not on the prompt.
- Training took about 65 minutes.

**Serving**
- The adapter was fused into the base model and imported into Ollama as `tinyllama-intent`, with 4-bit (`int4`) weights.
- It is scored with the unchanged evaluator (`experiments/conversation/evaluate.py`): rules first, then the model with the JSON schema, then the evidence guard. This is the same path the bot uses.

**Protocol**
- The development set (`heldout.jsonl`, 60 messages) was scored once after training. Nothing was tuned on it.
- The frozen test set (`heldout_v2.jsonl`, 45 messages, hash-checked) was then scored once.

## Training curve

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 0 | — | 1.967 |
| 75 | ~0.24 | 0.240 |
| 150 | 0.184 | 0.182 |
| 225 | ~0.17 | 0.173 |
| 300 | ~0.16 | 0.171 |

Validation loss fell together with training loss, so there is no sign of overfitting to the generated examples.

## Results

**Development set (60 messages)**

| Model | Intent-set accuracy |
|---|---:|
| stock tinyllama | 0.150 |
| qwen3:0.6b | 0.283 |
| qwen3:1.7b | 0.567 |
| **tinyllama-intent** | **0.800** [0.70, 0.90] |

**Frozen test set (45 messages)**

| Model | Size on disk | Intent-set accuracy [95% CI] | Action accuracy | Macro-F1 | Slot P / R | Median / p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| stock tinyllama | 637 MB | 0.156 [0.07, 0.27] | 0.244 | 0.184 | 0.21 / 0.23 | 740 / 1187 ms |
| qwen3:0.6b | 522 MB | 0.467 [0.31, 0.60] | 0.511 | 0.571 | 0.33 / 0.43 | 164 / 383 ms |
| qwen3:1.7b | 1.4 GB | 0.778 [0.64, 0.89] | 0.844 | 0.818 | 0.93 / 0.83 | 366 / 613 ms |
| **tinyllama-intent** | 887 MB | **0.867** [0.76, 0.96] | **0.889** | **0.904** | 0.79 / 0.77 | 338 / 546 ms |

- **Accuracy by language** for tinyllama-intent: English 0.90, Korean 0.83, Vietnamese 0.83.
- **Guard:** no output was rejected by the evidence guard, and no invalid actions were returned. Stock TinyLlama had 33 of 45 outputs rejected.
- **Memory:** about 0.9 GB loaded, against about 1.9 GB for qwen3:1.7b.

**Paired comparisons on the same 45 messages** (exact McNemar test)

| Against | Only ours right | Only theirs right | p |
|---|---:|---:|---:|
| stock tinyllama | 35 | 3 | 7e-08 |
| qwen3:0.6b | 19 | 1 | 4e-05 |
| qwen3:1.7b | 6 | 2 | 0.29 |

## Findings

1. **Fine-tuning turned TinyLlama from unusable into the best model tested.** It went from 0.156 to 0.867 on the same frozen test set. The gains over the stock model and over qwen3:0.6b are statistically clear.
2. **It matches qwen3:1.7b at about half the memory.** It is 9 points higher, but with 45 messages that difference is not significant (p = 0.29). The honest claim is "at least as good as a model 1.6 times its size", not "better".
3. **It also beats Duy's earlier run by a wide margin.** His run reached 13% exact intents on a different test with a different format, so the two numbers aren't directly comparable. The likely reasons for the improvement:
   - about 12 times more examples;
   - training on exactly the production prompt and output schema;
   - and scoring through the bot's real pipeline, where the evidence guard and schema remove malformed answers.
4. **Its slot precision is lower than qwen3:1.7b's** (0.79 vs 0.93). Most of its mistakes pick the wrong reason (tired, busy or other), mainly in Vietnamese, or miss a second intent.
5. **Two mistakes would cause unwanted actions.** "주말에 영화 볼까?" (watch a movie this weekend?) became a busy obstacle. "thời tiết mai thế nào?" (weather tomorrow?) became a request to move study to tomorrow. Plan changes still need the user's approval in the bot, but these are the errors to watch.

## Limitations

- **Synthetic data on both sides.** The training data is generated from templates. The test sets were written by the same developer, in a similar style. Real users would phrase things more variously, so real-world accuracy is probably lower than 0.867.
- **Small test set.** With 45 messages, the confidence interval is wide (0.76–0.96).
- **Different 4-bit format.** Ollama's import offers `int4` rather than the stock model's `q4_0`, so the fine-tuned model is 887 MB on disk against 637 MB.
- **CPU-only not measured.** Ollama runs this import on its MLX engine, which ignores the `num_gpu 0` setting used for the CPU-only runs in the model size report.
- **Not compared on Duy's test.** His held-out set isn't in the repository.

## Reproduce

```sh
uv venv --python 3.11 experiments/intent_sft/.venv
uv pip install --python experiments/intent_sft/.venv/bin/python -r experiments/intent_sft/requirements.txt
./experiments/intent_sft/run.sh          # data, train (about 1 hour on an M3), fuse, import into Ollama
uv run --locked python -m experiments.conversation.evaluate --mode live --model tinyllama-intent \
   --timeout 60 --scenarios experiments/conversation/heldout_v2.jsonl
```

Raw results, including every mistake, are in [results/](results/).
