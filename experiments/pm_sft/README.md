# TinyLlama QLoRA for PM intent extraction v0.1

This Windows CUDA experiment tests whether completion-only supervised fine-tuning improves multilingual intent and slot extraction from synthetic accountability messages. It runs independently of the Discord bot. The report gives the actual run evidence and limitations. See [benchmark report](BENCHMARK.md) and [dataset audit](DATASET_AUDIT.md).

## Reproduce

Run from the repository root in PowerShell. The isolated environment uses Python 3.12.10 and the exact package versions in [requirements.txt](requirements.txt). It needs the pinned TinyLlama model in the Hugging Face cache and an NVIDIA GPU with a driver compatible with the PyTorch CUDA 13.2 build.

```powershell
$pmDir = 'experiments/pm_sft'
$pmArtifacts = "$pmDir/artifacts"
$pmPython = "$pmDir/.venv/Scripts/python.exe"
$pmRun = "$pmArtifacts/reproduction"
$pmDataset = "$pmRun/dataset"
$pmBaseline = "$pmRun/baseline.json"
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:PYTHONUTF8 = '1'
uv venv "$pmDir/.venv" --python 3.12.10
uv pip install --python $pmPython --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match -r "$pmDir/requirements.txt"
```

If the pinned model is absent, populate the cache before setting offline mode. The base model is Apache 2.0. [Pinned TinyLlama revision](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6)

```powershell
& "$pmDir/.venv/Scripts/hf.exe" download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --revision fe8a4ea1ffedaf415f4da2f062534de366a451e6
$env:HF_HUB_OFFLINE = '1'
```

Use the local reviewed source `artifacts/reviewed.psv` created for this experiment. It has one row per scenario family with three independently worded language variants and language-indexed text slots where applicable. Retain this file for exact reproduction across machines. It is intentionally Git-ignored, so a fresh clone requires a copy of that source. Use a fresh output path for a reproduction so earlier runs and seals stay intact.

```powershell
New-Item -ItemType Directory -Force -Path $pmArtifacts | Out-Null
curl.exe --fail --location --output "$pmArtifacts/massive-1.0.tar.gz" https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz
& $pmPython -m experiments.pm_sft.dataset --output $pmDataset
& $pmPython -m experiments.pm_sft.evaluate baseline --dataset $pmDataset --output $pmBaseline
& $pmPython -m experiments.pm_sft.train smoke --dataset $pmDataset --baseline $pmBaseline --output "$pmRun/smoke"
& $pmPython -m experiments.pm_sft.train sft --dataset $pmDataset --baseline $pmBaseline --smoke-metrics "$pmRun/smoke/metrics.json" --output "$pmRun/sft"
& $pmPython -m experiments.pm_sft.reload_adapter "$pmRun/sft/adapter"
& $pmPython -m experiments.pm_sft.evaluate compare --dataset $pmDataset --adapter "$pmRun/sft/adapter" --output "$pmRun/compare.json"
```

The optional comparator uses the same validation utterances, system instructions and greedy decoding cap on a loopback-only Ollama server. The repo fallback tag is the default CLI tag. Supply `--ollama-model` for a different already-installed local tag. Run GPU-backed model invocations serially on the 6 GiB laptop.

```powershell
& $pmPython -m experiments.pm_sft.evaluate ollama --dataset $pmDataset --ollama-model qwen3.5:4b --output "$pmRun/ollama.json"
```

Run the ordinary CPU test suite with `.venv\Scripts\python.exe -m pytest -q`. The native-tokenizer test skips without Transformers or the pinned cache. To cover it in the experiment interpreter, run `uv pip install --python $pmPython pytest==9.2.0` and then `& $pmPython -m pytest -q experiments/pm_sft`. None of the unit tests train a model.

## Protocol

- Build distinct family-level training, validation and holdout partitions. Preparation records the SHA-256 of each partition and hashes the source prompt, configuration, loader and evaluator. Training loads the train and validation splits. The `compare` command loads the holdout after the matching SFT run and adapter verification succeed.
- Baseline and adapter inference use the same TinyLlama revision, 4-bit NF4 with double quantization, FP16 compute, greedy generation and an 80-token output cap. Ollama uses its own tokenizer and chat template.
- Native TinyLlama chat formatting is preserved. The fast tokenizer's offsets locate the assistant JSON, and a decoded-token round trip verifies the supervised span. The causal labels cover only the completion and EOS. Each sequence must fit the 256-token budget.
- Smoke testing runs 10 optimizer steps with gradient accumulation of 8. SFT starts from fresh LoRA weights and runs 48 steps, evaluating validation loss and task metrics at steps 16, 32 and 48. Gradient checkpointing uses the non-reentrant implementation. FP16 autocast and a loss scaler control mixed precision while the LoRA parameters stay FP32.
- Training writes per-step loss, unscaled gradient norm, elapsed time and peak CUDA allocator statistics. Reload verification matches all saved LoRA tensors and checks a finite change in probe logits when the adapter is enabled. The final artifact contains the LoRA adapter, tokenizer/config metadata and metrics.
- For OOM diagnosis, review the initial free VRAM and peaks in `metrics.json` and verify other GPU jobs with `nvidia-smi`. A shorter `--sequence-length` requires fitting every encoded example and matching smoke/SFT settings. `--optimizer adamw` is an explicit fallback if the installed Windows build cannot run AdamW8bit. Use separate output directories for retries.

## Extraction and policy

The exact JSON keys are `i` for intent labels, `s` for observed slots and `m` for explicitly missing or ambiguous information. `CLARIFY` accompanies nonempty `m`. Fixed relative markers preserve today, yesterday and tomorrow for resolution by a trusted caller. Reported total minutes and proposed target minutes use separate keys. A caller can supply trusted unresolved fields and a current pending proposal ID to the policy.

The validator checks keys, enumerations, types, ranges, dates, duplicate keys and consistency. The policy returns only `clarify`, `interpretation` or `human_review`. Adjustment decisions with a verified matching pending ID still require human review. Goal and schedule changes have no executor in this experiment.

Metrics separate strict parsing, schema validity, exact intent-set matching, slot accuracy and whole-example correctness. Slot accuracy uses the union of emitted and expected keys and gives invalid outputs zero credit on expected slots. Intent precision, recall and F1 include support and raw counts. The evaluator flags prohibited action proposals, unsupported adjustment decisions and misses on annotated uncertainty that would escape clarification. Raw generations and per-example scores stay in ignored local files.
