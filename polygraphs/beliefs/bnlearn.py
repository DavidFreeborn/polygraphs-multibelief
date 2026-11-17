"""An optional persistent R bnlearn/gRain exact-inference backend."""

from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading

import numpy as np


class BnlearnBackend:
    """Build user-specified bn.fit models and condition them with gRain."""

    def __init__(self, rscript=None, timeout=60):
        executable = (
            rscript or os.environ.get("POLYGRAPHS_RSCRIPT") or shutil.which("Rscript")
        )
        if not executable:
            raise RuntimeError(
                "Rscript not found. Set POLYGRAPHS_RSCRIPT or select backend: numpy"
            )
        self.timeout = timeout
        self._responses = queue.Queue()
        self._errors = deque(maxlen=30)
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                str(executable),
                "--vanilla",
                str(Path(__file__).with_name("bnlearn_worker.R")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        threading.Thread(
            target=self._read,
            args=(self._process.stdout, self._responses.put),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read,
            args=(self._process.stderr, self._errors.append),
            daemon=True,
        ).start()
        try:
            response = self._receive()
            if not response.get("ready"):
                raise RuntimeError(f"R backend failed to initialise: {response}")
            self.versions = response["versions"]
        except Exception:
            self.close()
            raise

    @staticmethod
    def _read(stream, consume):
        try:
            for line in stream:
                consume(line.rstrip())
        finally:
            consume(None)

    def _receive(self):
        try:
            line = self._responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("R inference timed out") from exc
        if line is None:
            detail = "\n".join(str(line) for line in self._errors if line)
            raise RuntimeError(
                f"R inference worker exited. Install bnlearn, gRain and jsonlite.\n{detail}"
            )
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"bnlearn/gRain: {result['error']}")
        return result

    def infer(self, network, factors, queries):
        names = {name: f"V{index}" for index, name in enumerate(network.names)}
        nodes = []
        for node in network.nodes:
            values = np.moveaxis(np.asarray(node["cpt"]), -1, 0)
            nodes.append(
                {
                    **node,
                    "name": names[node["name"]],
                    "parents": [names[parent] for parent in node["parents"]],
                    "cpt": values.ravel(order="F").tolist(),
                }
            )
        evidence = []
        for scope, values in factors.items():
            values = np.asarray(values)
            weights = np.exp(values - np.max(values))
            evidence.append(
                {
                    "scope": [names[name] for name in scope],
                    "weights": weights.ravel(order="F").tolist(),
                }
            )
        request = {
            "nodes": nodes,
            "evidence": evidence,
            "queries": [[names[name] for name in query] for query in queries],
        }
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("R inference worker is closed")
            self._process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self._process.stdin.flush()
            response = self._receive()
        result = []
        for query, values in zip(queries, response["probabilities"]):
            array = np.array(values, dtype=float).reshape(
                network.shape(query), order="F"
            )
            if (
                not np.isfinite(array).all()
                or np.any(array < 0)
                or not np.isclose(array.sum(), 1, rtol=1e-10, atol=1e-12)
            ):
                raise RuntimeError("R backend returned an invalid posterior")
            result.append(array)
        if len(result) != len(queries):
            raise RuntimeError("R backend returned the wrong number of queries")
        return result

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
