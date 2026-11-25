"""Bounded loopback HTTP and an isolated, single-request Ollama server."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from config import local_endpoint

from .annotation import SCHEMA, SYSTEM


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Ollama:
    def __init__(self, endpoint, timeout=5):
        self.endpoint = local_endpoint(endpoint)
        self.timeout = timeout
        self.http = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, path, payload=None, *, timeout=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(self.endpoint + "/api/" + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=timeout or self.timeout) as response:
                body = response.read(1024 * 1024 + 1)
        except HTTPError as error:
            with error:
                detail = error.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError(str(error.reason)) from error
            raise ConnectionError(str(error.reason)) from error
        if len(body) > 1024 * 1024:
            raise RuntimeError("Oversized Ollama envelope")
        value = json.loads(body)
        if type(value) is not dict or value.get("error"):
            raise RuntimeError(f"Ollama reported an error: {value}")
        return value

    def active(self):
        return self.call("ps")["models"]

    def generate(self, model, row, config):
        prompt = json.dumps({
            "anchor_day": row["anchor_day"], "anchor_zone": row["anchor_zone"],
            "pending_id": row["pending"], "message": row["text"],
        }, ensure_ascii=False)
        return self.call("generate", {
            "model": model, "system": SYSTEM, "prompt": prompt, "format": SCHEMA,
            "stream": False, "think": config["think"], "options": config["options"],
            "keep_alive": config["keep_alive"],
        }, timeout=config["timeout_s"])

    def unload(self, model):
        self.call("generate", {"model": model, "keep_alive": 0, "stream": False})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not any(model in {entry["name"], entry["model"]} for entry in self.active()):
                return
            time.sleep(0.2)
        raise RuntimeError(f"Ollama did not unload {model}")


def stop_tree(process):
    if process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                       timeout=10, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


@contextmanager
def isolated_server(config, log_path):
    endpoint = config["endpoint"]
    client = Ollama(endpoint)
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("The benchmark server requires an explicit IPv4 loopback port")
    address, port = endpoint.removeprefix("http://").split(":")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((address, int(port)))
    env = dict(os.environ, OLLAMA_HOST=f"{address}:{port}", OLLAMA_NUM_PARALLEL="1",
               OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_NO_CLOUD="1",
               OLLAMA_VULKAN=str(config["ollama_vulkan"]).lower())
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("ollama.exe is not on PATH")
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen([executable, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + config["server_startup_timeout_s"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Benchmark Ollama server exited with code {process.returncode}")
                try:
                    version = client.call("version", timeout=1)
                except (OSError, RuntimeError) as error:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Benchmark Ollama server did not start: {error}") from None
                    time.sleep(0.25)
                else:
                    break
            yield client, process, version
        finally:
            stop_tree(process)


def check_other_server(config):
    if config["other_endpoint"] == config["endpoint"]:
        raise ValueError("Benchmark and desktop Ollama endpoints must differ")
    try:
        active = Ollama(config["other_endpoint"], timeout=2).active()
    except ConnectionError as error:
        if "10061" in str(error) or "actively refused" in str(error).lower():
            return
        raise
    if active:
        raise RuntimeError("Desktop Ollama has a loaded model; wait until it is idle")
