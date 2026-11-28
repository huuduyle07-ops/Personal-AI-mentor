# Model size vs accuracy, with and without a GPU

Run date: 2026-11-27. Question from the project discussion: we want as little computation and space as possible. Can a smaller model (tinyllama) do the bot's job, and does the model run without a GPU?

## Setup

- **Task:** the bot's message understanding (`INTENT_PROMPT`, unchanged, sha `d6f2449c8b8e`). It turns a DM into structured actions (log minutes, quiz, obstacle, plan change, question, chit-chat, unclear), checked by the schema and the evidence guard.
- **Data:** the frozen held-out set `experiments/conversation/heldout_v2.jsonl`: 45 messages in English (21), Korean (12) and Vietnamese (12). It was never used to tune the prompt.
- **Models (Ollama, default 4-bit quantization):**
  - `tinyllama` (1.1B parameters, 637 MB);
  - `qwen3:0.6b` (522 MB);
  - `qwen3:1.7b` (1.4 GB, the bot's current model).
- **Devices:** each model was run twice:
  - **GPU:** the default, the Apple GPU through Metal;
  - **CPU only:** a copy created with `PARAMETER num_gpu 0`, which shares the same weights, so nothing extra is downloaded. `ollama ps` confirmed "100% CPU".
- **Machine:** Apple Silicon Mac, 8 GB RAM. Each model was unloaded before its run, so the first message includes loading time.
- **Metrics:**
  - *Intent accuracy:* the exact set of intents is right.
  - *Action accuracy:* the intents and their values (minutes, day, …) are right.
  - *Macro-F1* over intents.
  - Latency per message.
  - Memory while loaded (`ollama ps`).
- **Interval:** 95% bootstrap interval over messages.

## Results

| Model | Device | Intent accuracy (95% CI) | Action accuracy | Macro-F1 | Outputs rejected by the guard | Median / p95 latency | Memory loaded | Disk |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Rules only (no model) | — | 0.222 [0.11, 0.36] | 0.311 | 0.052 | — | <1 ms | 0 | 0 |
| tinyllama | GPU | 0.156 [0.07, 0.27] | 0.244 | 0.184 | 33 / 45 | 740 / 1187 ms | 644 MB | 637 MB |
| tinyllama | **CPU** | 0.156 [0.07, 0.27] | 0.244 | 0.199 | 33 / 45 | 1161 / 1990 ms | 739 MB | 637 MB |
| qwen3:0.6b | GPU | 0.467 [0.31, 0.60] | 0.511 | 0.571 | 2 / 45 | 164 / 384 ms | 826 MB | 522 MB |
| qwen3:0.6b | **CPU** | 0.489 [0.33, 0.62] | 0.511 | 0.572 | 3 / 45 | 300 / 833 ms | 1.0 GB | 522 MB |
| **qwen3:1.7b** | GPU | **0.778 [0.64, 0.89]** | **0.844** | **0.818** | 5 / 45 | 366 / 613 ms | 1.5 GB | 1.4 GB |
| **qwen3:1.7b** | **CPU** | **0.778 [0.64, 0.89]** | **0.844** | 0.808 | 6 / 45 | 587 / 920 ms | 1.9 GB | 1.4 GB |

Accuracy by language (GPU; the CPU-only runs are the same or within one message):

| Model | English | Korean | Vietnamese |
|---|---:|---:|---:|
| tinyllama | 0.14 | 0.17 | 0.17 |
| qwen3:0.6b | 0.62 | 0.25 | 0.42 |
| qwen3:1.7b | 0.76 | 0.83 | 0.75 |

Whole run of 45 messages, including the first load: tinyllama 34 s (GPU) / 56 s (CPU), qwen3:0.6b 10 s / 18 s, qwen3:1.7b 21 s / 32 s.

The `qwen3:1.7b` GPU run reproduces the frozen result exactly (0.778, macro-F1 0.818).

## Findings

1. **Every model runs without a GPU.** CPU-only accuracy matches the GPU runs, within one message, so moving to the CPU costs time, not quality. Latency goes up 1.6–1.8×. Even the largest model answers in about 0.6 s (median) and under 1 s (p95) per message on a laptop CPU.
2. **tinyllama is not usable for this task.**
   - The guard rejected 33 of its 45 outputs: 30 for evidence (actions or numbers the message didn't contain) and 3 for an invalid schema. The bot then safely asks the user to rephrase.
   - Its accuracy (0.156) is *below* the rules alone (0.222). Per message it is also slower than `qwen3:0.6b`, at a similar size.
   - It never produced an unsafe action, because the guard holds (safe-fallback rate 1.0).
   - It is an older English-centred chat model (2023), and it struggles to fill a strict JSON schema from Korean or Vietnamese messages.
3. **`qwen3:0.6b` is the smallest usable option.** At 522 MB and about 0.8–1.0 GB of memory, it is the cheapest and fastest. It more than doubles the rules' accuracy, but is weak on Korean (0.25–0.33).
4. **`qwen3:1.7b` is the best trade-off.** It is the only model that handles all three languages (0.75–0.83). For that it costs 2.7× the disk of `qwen3:0.6b` (1.4 GB vs 522 MB) and about 1.9× the memory (1.9 GB vs 1.0 GB on the CPU), and it still runs comfortably on a CPU.

## Recommendation

- **Keep `qwen3:1.7b`** if the bot should understand Korean and Vietnamese messages. It needs no GPU: about 1.4 GB of disk and 2 GB of RAM, and well under a second per message on a CPU.
- **Use `qwen3:0.6b`** if space and computation matter most and users mostly write in English. Set `OLLAMA_MODEL=qwen3:0.6b`; nothing else changes.
- **Drop tinyllama.** It is not smaller in practice (637 MB vs 522 MB for `qwen3:0.6b`), and it is less accurate than having no model at all.

The bot already keeps model use low: the rules handle the most common messages with no model, and the model is only asked about messages the rules don't recognise. The per-message cost above is therefore an upper bound.

## Limitations

- **Small test set.** With 45 messages, the intervals are wide (about ±0.12): `qwen3:1.7b` vs `qwen3:0.6b` is a clear difference, but small differences are not.
- **One machine.** CPU timings come from an Apple Silicon laptop CPU. An older x86 laptop will be slower. Run `run.sh` there to measure it.
- **Not tested here:**
  - Default 4-bit quantization only; no 8-bit or full-precision variants.
  - Message understanding only: chat replies, sentence quizzes and photo reading (`qwen2.5vl:3b`, 3.2 GB) were not benchmarked.

## Reproduce

```bash
ollama pull tinyllama qwen3:0.6b qwen3:1.7b
for m in tinyllama qwen3:0.6b qwen3:1.7b; do
  printf "FROM %s\nPARAMETER num_gpu 0\n" "$m" > /tmp/cpu.Modelfile
  ollama create "cpu-$(echo $m | tr ':' '-')" -f /tmp/cpu.Modelfile
done
bash experiments/model_size/run.sh   # results in experiments/model_size/results/
```
