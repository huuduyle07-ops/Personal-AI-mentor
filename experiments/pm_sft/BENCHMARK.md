# PM intent extraction v0.1 benchmark

Run date November 24, 2026 (KST). The hypothesis was that completion-only QLoRA would improve strict JSON extraction from synthetic English, Vietnamese and Korean PM messages. The paired comparison uses the same held-out families, prompt, model revision, quantization and decoding settings.

## Environment and data

| Component | Version or setting |
| --- | --- |
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 at `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Host | Windows x86-64, NVIDIA GeForce RTX 3050 6GB Laptop GPU, driver 616.64 |
| Python and package installer | CPython 3.12.10, uv 0.11.29 |
| PyTorch and CUDA runtime | 2.15.0+cu132, CUDA 13.2 |
| Training packages | Transformers 5.22.0, PEFT 0.22.0, bitsandbytes 0.50.2, Accelerate 1.16.0 |
| Model I/O | huggingface-hub 2.4.0, tokenizers 0.23.3, safetensors 0.8.0 |
| Quantization | 4-bit NF4 with double quantization and FP16 compute |
| LoRA | 1,126,400 trainable parameters, rank 8, alpha 16, dropout 0.05, q_proj and v_proj |
| Optimization | AdamW8bit at 0.0002, FP16 autocast, FP32 adapter parameters, non-reentrant gradient checkpointing, one example per microbatch, eight microbatches per optimizer step |

There are 96 train, 18 validation and 30 holdout examples, balanced across languages. Train includes 78 originally authored cases and 18 aligned MASSIVE cases. Validation and holdout use authored PM scenarios exclusively. NVIDIA persona categories informed fictional situations. See [source attribution, mappings and review coverage](DATASET_AUDIT.md) and [the pinned environment](requirements.txt).

The final run used [the local sealed manifest](artifacts/dataset-sealed/manifest.json). The holdout SHA-256 is `6977c1b5a4d5242488ee2a74e1880aabb4bc3c9bc87bda99fd1701a124611168` and the prompt/evaluation protocol SHA-256 is `784ca30602d0ca04be698e1cf0cd956ad28a8e3504c54808755a5a52bef50ff4`. Two preflight prompt versions exceeded the 256-token budget. The final compact prompt and shortened training and validation utterances fit at 252 and 237 tokens respectively. The content hash for the holdout stayed identical during those revisions.

## Base validation and smoke test

The original 4-bit TinyLlama returned 0/18 parse-valid and 0/18 schema-valid validation responses. Exact intent sets, exact JSON and slot accuracy were all 0%. Median latency was 2.55 s and p95 was 3.79 s. [Base validation results](artifacts/baseline-sealed.json)

The smoke run completed 10 optimizer updates with finite, positive gradient norms on all steps and gradients on all 88 adapter tensors. The first update changed a trainable tensor by up to 0.000200. The run spent 33.4 s in the fit loop and 52.2 s end to end including load and reload. PyTorch peaked at 1,271.5 MiB allocated and 1,334 MiB reserved. All 88 saved tensors matched after reload, and enabling the adapter changed finite probe logits by a maximum of 8.38. [Smoke metrics](artifacts/smoke-sealed/metrics.json)

### Local Ollama validation comparison

The repository fallback `qwen3:0.6b` was absent from the server. The installed `qwen3.5:4b` comparator ran on Ollama 0.40.17 with model digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`. It returned parse-valid JSON on 18/18 validation examples and schema-valid JSON on 0/18. A recurring format error was a string in `i`, which requires an intent array. Exact intent, slot and whole-example correctness were 0% after schema validation. Median latency was 0.49 s and p95 was 1.16 s. The comparator used the same messages and output cap with its own chat template and backend. [Default-tag check](artifacts/ollama-default.json), [Qwen validation results](artifacts/ollama.json)

## SFT run

The final run started with fresh LoRA weights and completed the planned 48 updates, or 384 microbatches across four passes through the training set. Optimizer steps took 104.6 s in total with a median of 2.11 s. The fit loop took 191.6 s including validation, and the run took 212.9 s including model load, adapter save and a fresh reload.

First-step and last-step training losses were 1.5563 and 0.1092. Unscaled gradient norms stayed finite and ranged from 1.0683 to 4.7166. The peak PyTorch CUDA allocation and reservation were 1,271.5 MiB and 1,334 MiB. Both training runs completed without OOM or numerical instability. These figures measure the process allocator. Other applications and the Windows display stack use additional VRAM.

| Step | Validation loss | Schema valid | Exact intent set | Whole JSON | Slot accuracy | Policy flags |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.4623 | 6/18 | 0/18 | 0/18 | 1/27 | 3/18 |
| 32 | 0.2864 | 12/18 | 0/18 | 0/18 | 6/36 | 4/18 |
| 48 | 0.2121 | 17/18 | 2/18 | 2/18 | 1/22 | 1/18 |

All 88 final adapter tensors matched the saved values after a fresh reload. Enabling the adapter changed probe logits by up to 18.55. The adapter weights, tokenizer and configuration are under [artifacts/sft/adapter](artifacts/sft/adapter). The weight file is 4,517,152 bytes, with SHA-256 `e1872c13b694c25c72770883b81ee8872a1dce1aa9cc00b9f9fd6a3f7fe8679c`. The run's [metrics.json](artifacts/sft/metrics.json) includes step timing, validation scores and reload evidence.

## Sealed holdout

| Metric | 4-bit TinyLlama | TinyLlama + QLoRA |
| --- | ---: | ---: |
| Exact intent set | 0/30 | 4/30 (13.3%) |
| Slot accuracy | 0/36 | 11/45 (24.4%) |
| Exact missing-field set | 0/30 | 6/30 (20.0%) |
| Whole-example JSON correct | 0/30 | 4/30 (13.3%) |
| Parse-valid JSON | 0/30 | 30/30 (100%) |
| Schema-valid JSON | 0/30 | 26/30 (86.7%) |
| Unsupported or unsafe proposal flags | 0/30 | 2/30 (6.7%) |
| Median / p95 latency | 2.42 / 2.62 s | 1.04 / 1.57 s |

Slot scores count both expected and extra emitted keys. The base model mostly echoed the prompt or wrote prose. Its invalid output defaulted to clarification, so the 0/30 policy flags accompany 0/30 usable structured responses. The adapted model produced two policy flags for missed uncertainty. Both were display-only interpretations under the action policy. The experiment has no executor. [Paired scores and raw generations](artifacts/compare.json)

### Intent performance

The base model had zero recall and F1 for each intent. Precision was undefined for every label because the schema rejected all base responses. A dash also indicates undefined precision for any intent the adapter never emitted in a schema-valid response.

| Intent | Holdout support | QLoRA precision | QLoRA recall | QLoRA F1 |
| --- | ---: | ---: | ---: | ---: |
| REPORT_PROGRESS | 9 | 50.0% | 22.2% | 30.8% |
| REPORT_OBSTACLE | 3 | — | 0% | 0% |
| REQUEST_RECOVERY | 6 | — | 0% | 0% |
| QUERY_PROGRESS | 6 | — | 0% | 0% |
| REQUEST_VERIFICATION | 3 | — | 0% | 0% |
| APPROVE_ADJUSTMENT | 3 | 100% | 33.3% | 50.0% |
| REJECT_ADJUSTMENT | 3 | — | 0% | 0% |
| GENERAL_CONVERSATION | 3 | 25.0% | 33.3% | 28.6% |
| CLARIFY | 15 | 55.6% | 66.7% | 60.6% |

The single true-positive approval intent still omitted the required clarification about the absent pending proposal. The policy required clarification because there was no matching trusted proposal.

### Failures by language

Base exact intent, slot and whole-example scores were all zero in each language. The adapter's outcomes were:

| Language | Exact intent and whole JSON | Slot accuracy | Schema valid | Policy flags | Median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 2/10 | 2/16 | 8/10 | 0/10 | 1.08 / 1.74 s |
| Vietnamese | 1/10 | 3/13 | 9/10 | 0/10 | 0.99 / 1.35 s |
| Korean | 1/10 | 6/16 | 9/10 | 2/10 | 1.20 / 1.57 s |

- **English.** The model captured 28 cooking minutes but classified the report as general conversation and omitted the goal. It invented yesterday for an explicitly uncertain date. It labeled a ghost-proposal request as approval and put the ID under `goal`, which led to clarification at the policy boundary.
- **Vietnamese.** It put a proposed 17-minute target into `minutes`, used `today` for a request about tomorrow and missed the outage and recovery intents. It treated both a corrected duration and an everyday greeting as unsupported requests.
- **Korean.** It captured the corrected 36 minutes and yesterday, then omitted the progress query. It also treated an ambiguous goal and a 14-or-41-minute report as general conversation without requesting clarification. These were the two policy flags.

## Suitability and reproduction

The adapter is suitable for further offline testing. It learned the response format faster than the extraction task. Exact JSON was correct in four of 30 cases, five intents had zero recall, and both Korean uncertainty misses would require independent safeguards in any future integration. The evaluation contains 10 distinct scenario families and the corpus received author-only review. The adapter stays in the isolated experiment.

Use the PowerShell commands in [the experiment README](README.md) with [the local reviewed source](artifacts/reviewed.psv). A second preparation produced byte-for-byte identical train, validation and holdout JSONL files and manifest. All 13 experiment tests passed in the experiment environment, including native chat-template masking. The existing main test suite passed 127 tests.
