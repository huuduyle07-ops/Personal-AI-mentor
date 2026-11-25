"""Windows system memory, hard page reads, NVIDIA VRAM and pressure monitor."""

import csv
import ctypes
from ctypes import wintypes
import shutil
import subprocess
from threading import Event, Thread
import time

MIB = 1024 ** 2


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD), ("load", wintypes.DWORD),
        *[(name, ctypes.c_ulonglong) for name in (
            "total_ram", "free_ram", "total_commit", "free_commit",
            "total_virtual", "free_virtual", "extended_virtual")],
    ]


def memory_sample():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {name + "_mib": round(getattr(status, name) / MIB, 1)
            for name in ("total_ram", "free_ram", "total_commit", "free_commit")}


class CounterValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_double)]


class PageReads:
    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh")
        self.query = wintypes.HANDLE()
        self.counter = wintypes.HANDLE()
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
            self.query = None
            return
        if self.pdh.PdhAddEnglishCounterW(self.query, r"\Memory\Pages Input/sec", 0,
                                          ctypes.byref(self.counter)) != 0:
            self.close()

    def read(self):
        if not self.query or self.pdh.PdhCollectQueryData(self.query) != 0:
            return None
        value = CounterValue()
        status = self.pdh.PdhGetFormattedCounterValue(self.counter, 0x200, None, ctypes.byref(value))
        if status != 0 or value.status not in {0, 1}:
            return None
        return round(value.value * 4096 / MIB, 2)

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
        self.query = None


def gpu_sample():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, "--query-gpu=name,driver_version,memory.total,memory.used",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=2, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for name, driver, total, used in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        devices.append({"name": name, "driver": driver, "total_vram_mib": int(total) if total.isdigit() else None,
                        "used_vram_mib": int(used) if used.isdigit() else None})
    return devices


def pressure_reason(sample, config):
    free = sample["free_ram_mib"]
    if free < config["minimum_running_available_mib"]:
        return "low_available_ram"
    if sample["free_commit_mib"] < config["minimum_running_available_commit_mib"]:
        return "low_commit_headroom"
    if (free < config["paging_headroom_mib"] and sample["page_reads_mib_s"] is not None
            and sample["page_reads_mib_s"] >= config["paging_read_mib_s"]):
        return "sustained_hard_page_reads_under_ram_pressure"
    return None


class Monitor:
    def __init__(self, config, abort):
        self.config = config
        self.abort = abort
        self.samples = []
        self.reason = None
        self.done = Event()
        self.thread = Thread(target=self._watch, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join(timeout=18)
        if self.thread.is_alive():
            raise RuntimeError("Resource monitor did not stop")

    def _watch(self):
        counter = None
        previous = None
        streak = 0
        start = time.monotonic()
        try:
            counter = PageReads()
            while not self.done.is_set():
                sample = {"elapsed_s": round(time.monotonic() - start, 3), **memory_sample(),
                          "page_reads_mib_s": counter.read(), "gpus": gpu_sample()}
                self.samples.append(sample)
                reason = pressure_reason(sample, self.config)
                streak = (streak + 1 if reason == previous else 1) if reason else 0
                previous = reason
                if sample["free_ram_mib"] < 768 or streak >= self.config["pressure_samples"]:
                    self.reason = reason or "critical_ram"
                    break
                self.done.wait(self.config["sample_interval_s"])
        except Exception as error:
            self.reason = f"monitor_failed: {type(error).__name__}: {error}"
        finally:
            if counter:
                counter.close()
        if self.reason:
            self.abort()
