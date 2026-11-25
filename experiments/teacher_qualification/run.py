"""Run the two installed Ollama candidates serially and keep local raw artifacts."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from experiments.pm_sft.contract import dumps

from .annotation import SCHEMA, SYSTEM
from .dataset import ANCHOR_DAY, ANCHOR_ZONE, ROOT, load_reference
from .ollama import check_other_server, isolated_server, stop_tree
from .resources import Monitor, gpu_sample, memory_sample

WARMUP = {"text": "I practiced for 11 minutes today.", "pending": None,
          "anchor_day": ANCHOR_DAY, "anchor_zone": ANCHOR_ZONE}
RUN_ERRORS = (OSError, RuntimeError, ValueError, KeyError)
MEMORY_ERRORS = ("out of memory", "unable to allocate", "failed to allocate", "not enough memory",
                 "requires more system memory", "insufficient memory", "memory allocation")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_one(client, model, row, config):
    start = time.monotonic()
    result = {"id": row.get("id"), "status": "ok"}
    try:
        response = client.generate(model, row, config)
        result["response"] = response
        if response.get("done") is not True or type(response.get("response")) is not str:
            result["status"] = "invalid_envelope"
    except TimeoutError as error:
        result.update(status="timeout", error=str(error))
    except RUN_ERRORS as error:
        result.update(status="error", error=str(error))
    result["latency_s"] = time.monotonic() - start
    return result


def evaluate_model(client, process, model, rows, config, directory):
    directory.mkdir()
    result = {"model": model, "expected": len(rows), "attempted": 0, "status": "incomplete"}
    monitor = Monitor(config, lambda: stop_tree(process))
    attempted_load = False
    try:
        check_other_server(config)
        result["preload"] = {**memory_sample(), "gpus": gpu_sample()}
        if result["preload"]["free_ram_mib"] < config["minimum_preload_available_mib"]:
            result["reason"] = "insufficient_available_ram_before_load"
            return result
        if client.active():
            raise RuntimeError("Benchmark server already has a loaded model")
        with monitor, (directory / "raw.jsonl").open("x", encoding="utf-8") as raw:
            attempted_load = True
            result["warmup"] = generate_one(client, model, WARMUP, config)
            if result["warmup"]["status"] != "ok":
                result["reason"] = "warmup_" + result["warmup"]["status"]
                return result
            active = client.active()
            result["loaded"] = active
            if (len(active) != 1 or active[0]["name"] != model
                    or active[0].get("context_length") != config["options"]["num_ctx"]):
                raise RuntimeError("Loaded model or context length did not match the benchmark")
            for row in rows:
                if monitor.reason:
                    break
                check_other_server(config)
                item = generate_one(client, model, row, config)
                raw.write(dumps(item) + "\n")
                raw.flush()
                result["attempted"] += 1
                if item["status"] != "ok":
                    result["reason"] = "inference_" + item["status"]
                    break
                if result["attempted"] % 10 == 0:
                    print(f"{model} {result['attempted']}/{len(rows)}", flush=True)
            if result["attempted"] == len(rows) and "reason" not in result:
                result["status"] = "complete"
    except RUN_ERRORS as error:
        result["reason"] = str(error)
    finally:
        if monitor.reason:
            result.update(status="incomplete", reason=monitor.reason, pressure_stop=True)
        if attempted_load:
            if process.poll() is None:
                try:
                    client.unload(model)
                    result["unloaded"] = True
                except RUN_ERRORS as error:
                    result.update(status="incomplete", unloaded=False, unload_error=str(error))
            else:
                result["unloaded"] = True
                result["unloaded_by"] = "isolated_server_exit"
        result["postunload"] = {**memory_sample(), "gpus": gpu_sample()}
        write_json(directory / "resources.json", monitor.samples)
        write_json(directory / "run.json", result)
    return result


def run(config, root):
    if os.name != "nt":
        raise RuntimeError("Live qualification requires the Windows resource monitor")
    if config["parallel_requests"] != 1:
        raise ValueError("Only one parallel request is supported")
    rows = load_reference()
    root.mkdir(parents=True)
    manifest = {
        "started_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "config": config, "reference_sha256": sha256(dumps(rows).encode()).hexdigest(),
        "prompt_schema_sha256": sha256(dumps([SYSTEM, SCHEMA]).encode()).hexdigest(), "models": [],
    }
    try:
        check_other_server(config)
        with isolated_server(config, root / "ollama.log") as (client, process, version):
            manifest["ollama"] = version
            installed = {entry["name"]: entry for entry in client.call("tags")["models"]}
            manifest["installed_candidates"] = {name: installed.get(name) for name in config["models"]}
            for model in config["models"]:
                directory = root / model.replace(":", "-")
                if model not in installed:
                    manifest["models"].append({"model": model, "status": "incomplete", "reason": "not_installed"})
                    continue
                print(f"Starting {model}", flush=True)
                result = evaluate_model(client, process, model, rows, config, directory)
                manifest["models"].append(result)
                print(f"{model} {result['status']} {result.get('reason', '')}", flush=True)
                error = (directory / "raw.jsonl")
                failed_raw = error.read_text(encoding="utf-8").splitlines()[-1:] if (
                    error.exists() and result["status"] != "complete") else []
                diagnostic = " ".join([str(result.get("reason")), str(result.get("warmup", {}).get("error")), *failed_raw])
                if (result.get("pressure_stop") or process.poll() is not None or result.get("unloaded") is False
                        or any(word in diagnostic.lower() for word in MEMORY_ERRORS)):
                    manifest["stop_reason"] = "pressure, memory error or unverified isolation"
                    break
    except RUN_ERRORS as error:
        manifest["stop_reason"] = str(error)
    finally:
        manifest["finished_kst"] = datetime.now(timezone(timedelta(hours=9))).isoformat()
        write_json(root / "manifest.json", manifest)
    return len(manifest["models"]) != len(config["models"]) or any(
        result["status"] != "complete" for result in manifest["models"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(json.loads(args.config.read_text(encoding="utf-8")), args.output))


if __name__ == "__main__":
    main()
