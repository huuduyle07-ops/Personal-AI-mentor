# PM teacher qualification benchmark

Run date November 24, 2026 (KST). The benchmark tests whether two installed Qwen checkpoints can produce accurate multilingual PM labels for a later offline distillation experiment. The benchmark has 60 provisional reference examples across 20 scenario families and three languages. Family variants share the underlying scenario. Independent human adjudication is pending. [Reference families](reference.json), [protocol and decision rules](README.md)

**Neither candidate qualifies as a primary teacher or secondary reviewer.** Both ran to completion. The 4B model had stronger aggregate extraction and lower resource cost. Both missed the reviewer gates on exact intents, slot F1, evidence and missing-field recall. Neither correctly marked any of the six explicitly ambiguous examples.

## Protocol and run provenance

The Windows host has 16,071.9 MiB of visible RAM and an NVIDIA GeForce RTX 3050 Laptop GPU with 6,144 MiB VRAM and driver 616.64. Both Q4_K_M tags ran on Ollama 0.40.17 through a single-request loopback server, using a 2,048-token context, a 256-token output budget, temperature 0, seed 42, JSON Schema, and disabled thinking. A first startup attempt reached the 20-second timeout during GPU discovery and loaded neither model. The measured retry used CUDA-only discovery and a 60-second startup window. It ran 16:52:49–17:33:18 KST including server startup and teardown. [Startup attempt](artifacts/phase2a-20261124/manifest.json), [measured run manifest](artifacts/phase2a-20261124-cuda/manifest.json)

The reference SHA-256 is `558b703cddd5858794ff72649f672912c17d71b40cf25b187079cd3b02b66775`. The prompt/schema SHA-256 is `426fb2a131e34267cbb29b5342391712aabba07dd1f821a2f089945121b85ed3`.

The installed digests are `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` for 4B and `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` for 9B.

## Quality and latency

Metrics use attempted rows as the denominator. Hard evidence validity checks literal quote coverage and slot grounding. Whole-example accuracy requires the exact reference interpretation and a clean validation record. Exact slot precision and recall use key/value pairs.

| Model / language | Attempts | Parse | Schema | PM contract | Exact intents | Macro F1 | Slot P / R / F1 | Exact missing | Valid evidence | Whole example | Policy flags | Median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 4B / overall | 60/60 | 60/60 | 60/60 | 48/60 | 32/60 | 77.6% | 64.6% / 46.7% / 54.2% | 35/60 | 32/60 | 9/60 | 19/60 | 2.68 / 3.59 s |
| 4B / English | 20/20 | 20/20 | 20/20 | 15/20 | 11/20 | 80.3% | 70.0% / 46.7% / 56.0% | 11/20 | 13/20 | 2/20 | 5/20 | 2.53 / 3.42 s |
| 4B / Vietnamese | 20/20 | 20/20 | 20/20 | 17/20 | 11/20 | 77.7% | 64.0% / 53.3% / 58.2% | 13/20 | 12/20 | 4/20 | 8/20 | 2.89 / 3.67 s |
| 4B / Korean | 20/20 | 20/20 | 20/20 | 16/20 | 10/20 | 75.1% | 60.0% / 40.0% / 48.0% | 11/20 | 7/20 | 3/20 | 6/20 | 2.62 / 3.50 s |
| 9B / overall | 60/60 | 60/60 | 60/60 | 29/60 | 24/60 | 52.0% | 66.7% / 33.3% / 44.4% | 22/60 | 33/60 | 7/60 | 26/60 | 37.80 / 48.62 s |
| 9B / English | 20/20 | 20/20 | 20/20 | 12/20 | 11/20 | 62.4% | 77.8% / 46.7% / 58.3% | 10/20 | 14/20 | 3/20 | 6/20 | 35.23 / 48.56 s |
| 9B / Vietnamese | 20/20 | 20/20 | 20/20 | 8/20 | 6/20 | 34.2% | 58.3% / 23.3% / 33.3% | 6/20 | 11/20 | 2/20 | 10/20 | 38.24 / 47.80 s |
| 9B / Korean | 20/20 | 20/20 | 20/20 | 9/20 | 7/20 | 52.7% | 60.0% / 30.0% / 40.0% | 6/20 | 8/20 | 2/20 | 10/20 | 37.98 / 49.67 s |

The models recalled 2/18 expected missing fields for 4B and 0/18 for 9B. Numeric validation passed for 52/60 and 46/60, respectively. Temporal validation passed for 55/60 and 49/60. These two checks test emitted slots, so they can pass when a slot is absent. Exact slot recall penalizes omissions. Literal evidence can pass with a quote whose interpretation still needs adjudication.

| Policy flag | 4B | 9B |
| --- | ---: | ---: |
| Missed required clarification | 16 | 18 |
| Ungrounded proposal ID | 4 | 17 |
| Decision without trusted pending context | 6 | 5 |
| Unsupported decision | 2 | 0 |
| Unsupported outbound request missed | 3 | 3 |
| Responses with any flag | 19/60 | 26/60 |

A response can carry several flags. The evaluator added the ungrounded-ID check after inspecting early outputs and applied it to both complete raw files. The references, decoding and semantic metrics stayed fixed. For example, both models recognized the English approval of P-317 and omitted clarification about the absent pending fixture. In the English case with two possible recovery dates, 4B marked the date missing but set ambiguity to false. 9B picked tomorrow and added an unspoken P-2026. Both outputs are preserved in the review CSV.

The interpretations differed on 46/60 examples, including 15 English, 15 Vietnamese and 16 Korean. Both models passed the whole-example check on three examples, and both failed it on 47. Exact boundaries on goal and reason strings can affect slot scores and are open to human adjudication. [Per-intent metrics](artifacts/phase2a-20261124-cuda/metrics.json), [side-by-side review CSV](artifacts/phase2a-20261124-cuda/review.csv)

## Resource feasibility

| Model | Cold load / total cold request | Decode median / p95 | Preload available RAM | Min available RAM | Peak system RAM used | Peak device VRAM | Ollama GPU share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 7.05 / 9.48 s | 45.56 / 46.07 tok/s | 6,949 MiB | 5,623 MiB | 10,449 MiB | 3,685 MiB | 100% |
| 9B | 17.03 / 44.41 s | 3.83 / 3.94 tok/s | 6,649 MiB | 3,581 MiB | 12,491 MiB | 4,549 MiB | 59.2% |

Both models had 0/60 transport failures, 0/60 timeouts and 0/60 output-cap stops. Neither emitted thinking text. 9B's median latency was 14.1 times 4B's. Ollama offloaded 33/33 layers for 4B and 20/34 for 9B. The reported allocations were 2,929 MiB GPU for 4B and 3,432 MiB GPU out of 5,802 MiB for 9B. The layer counts and allocation split come from Ollama. Device VRAM and RAM are sampled system-wide while the model remains resident across languages.

The monitor captured 251 samples for 4B and 1,827 for 9B. Peak RAM increases over preload were 1,326 and 3,068 MiB, respectively. Peak hard page reads were 1,027 and 1,121 MiB/s. Hard reads include file-backed loading. Minimum available RAM stayed above the 3,072 MiB paging-guard headroom threshold for both runs, and neither triggered a pressure stop or reported OOM. Cold load means a fresh model runner with the existing OS disk cache. Both unloads were verified, and the benchmark port closed. [4B run](artifacts/phase2a-20261124-cuda/qwen3.5-4b/run.json), [9B run](artifacts/phase2a-20261124-cuda/qwen3.5-9b/run.json), [Ollama runner log](artifacts/phase2a-20261124-cuda/ollama.log)

## Prior baseline

TinyLlama + QLoRA previously reached 30/30 parse-valid, 26/30 schema-valid, 4/30 exact intent sets (13.3%), 11/45 slot accuracy (24.4%), 4/30 whole-example JSON and 6/30 exact missing-field sets on its separate sealed holdout. Its median and p95 latencies were 1.04 s and 1.57 s. The Qwen runs recorded higher exact intent percentages on Phase 2A, which has different scenarios, generation settings and validation requirements. The two phases use different slot and whole-example scores. These observations provide no paired estimate of improvement. [TinyLlama benchmark](../pm_sft/BENCHMARK.md)

## Verification

All 18 qualification tests and all 127 existing application tests passed. The benchmark run and review export remain local under the ignored artifacts directory.
