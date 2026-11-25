from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "pilot.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "pilot"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(ROOT.rglob("*.py"))
            if not any(x in p.parts for x in (".venv", "artifacts", "tests"))}


class MemoryMonitor:
    """Sample whole-process resident memory, including libraries and allocator caches."""

    def __enter__(self):
        self.process = psutil.Process()
        self.start_rss = self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()
        self.started = time.perf_counter()

        def sample():
            while not self.stop.wait(0.02):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.result = {"start_rss_mb": self.start_rss / 2**20,
                       "sampled_peak_rss_mb": self.peak_rss / 2**20,
                       "elapsed_seconds": time.perf_counter() - self.started,
                       "sampling_interval_seconds": 0.02,
                       "scope": "whole Python process, not isolated model or Python allocations"}


def environment():
    import importlib.metadata
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(),
            "packages": {m: importlib.metadata.version(m) for m in
                         ["torch", "numpy", "scikit-learn", "pyarrow", "psutil", "matplotlib"]}}
