"""Separate dependency workers, with saved setup errors before any model loads."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from brain_vlm.common import require, write
from brain_vlm.prima.setup import REQUIREMENTS as PRIMA_REQUIREMENTS

MEDGEMMA_REQUIREMENTS = (
    "transformers==4.57.6", "accelerate==1.15.0", "lm-format-enforcer==0.11.3",
    "jsonschema==4.26.0", "Pillow", "psutil",
)
PACKAGES = (
    "pip", "packaging", "torch", "numpy", "transformers", "tokenizers",
    "SimpleITK", "monai", "monai-generative", "einops", "nibabel", "scipy",
    "matplotlib", "psutil", "accelerate", "lm-format-enforcer", "jsonschema", "Pillow",
)
MARKER = "BRAIN_VLM_ENV_JSON="


def bootstrap_environment(folder, family, log_dir):
    """Create a Colab worker without ensurepip, then use the pinned installer."""
    root = Path(folder).absolute()
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    log = Path(log_dir) / ("bootstrap_" + family + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    _run_logged([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(root)],
                log, "Prepare " + family + " worker")
    try:
        _run_logged([str(python), "-m", "pip", "--version"], log, "Worker pip")
    except RuntimeError:
        _run_logged([sys.executable, "-m", "pip", "--python", str(python), "install", "--ignore-installed", "--no-deps", "pip"],
                    log, "Install worker pip")
    return str(python)


def _redact(text):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        token = os.environ.get(name)
        if token:
            text = text.replace(token, "[REDACTED]")
    return text


def _run_logged(command, log, label, stream=False, heartbeat_seconds=30):
    """Capture child stdout AND stderr; Colab otherwise may show only the wrapper."""
    if stream:
        return _run_streamed(command, log, label, heartbeat_seconds)
    environment = os.environ.copy()
    environment.update(USE_TF="0", USE_FLAX="0", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    print(label + " …", flush=True)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=environment)
        output, returncode = _redact(result.stdout or ""), result.returncode
    except OSError as error:
        output, returncode = _redact(str(error)), -1
    with Path(log).open("a", encoding="utf-8") as stream:
        stream.write(f"\n--- {label} (exit {returncode}) ---\n{output}\n")
    if returncode:
        tail = "\n".join(output.splitlines()[-80:])
        print(tail or "Child process returned no output.", flush=True)
        raise RuntimeError(f"{label} failed (exit {returncode}). Full log: {log}\n{tail}")
    return output


def _run_streamed(command, log, label, heartbeat_seconds):
    """Tee unbuffered worker output to Colab and disk before the worker exits."""
    import queue
    import threading
    import time
    require(heartbeat_seconds > 0, "Heartbeat interval must be positive")
    log = Path(log);log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(USE_TF="0", USE_FLAX="0", PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    lines, events = [], queue.Queue()
    started = time.monotonic()
    process = None
    print(label + " … Live log: " + str(log), flush=True)
    with log.open("a", encoding="utf-8") as target:
        def emit(message):
            message = _redact(message)
            target.write(message);target.flush()
            print(message, end="" if message.endswith("\n") else "\n", flush=True)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            emit(f"--- {label} started; worker PID={process.pid} ---\n")
            def collect():
                try:
                    for line in process.stdout:events.put(line)
                finally:events.put(None)
            reader = threading.Thread(target=collect, daemon=True);reader.start()
            while True:
                try:line = events.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    emit(f"[{label}] {time.monotonic()-started:.0f}s elapsed; worker has not produced a new line. This is not a progress estimate.\n")
                    continue
                if line is None:break
                line = _redact(line);lines.append(line);emit(line)
            returncode = process.wait()
            reader.join(timeout=1)
            emit(f"--- {label} exit {returncode}; elapsed {time.monotonic()-started:.1f}s ---\n")
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            emit(f"--- {label} interrupted/failed; direct worker stopped ---\n")
            raise
        finally:
            if process is not None and process.stdout is not None:process.stdout.close()
    output = "".join(lines)
    if returncode:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"{label} failed (exit {returncode}). Full log: {log}\n{tail}")
    return output




def _inspect(python, log):
    script = """import json, platform, sys
from importlib.metadata import version, PackageNotFoundError
packages = {}
for name in NAMES:
    try: packages[name] = version(name)
    except PackageNotFoundError: packages[name] = None
print(MARKER + json.dumps(dict(python=sys.version, executable=sys.executable,
    base_prefix=sys.base_prefix, platform=platform.platform(), packages=packages)))
"""
    script = "NAMES=" + repr(PACKAGES) + "\nMARKER=" + repr(MARKER) + "\n" + script
    output = _run_logged([str(python), "-c", script], log, "Inspect worker Python/packages")
    return json.loads(next(line[len(MARKER):] for line in output.splitlines() if line.startswith(MARKER)))


def _check(python, family, src, log):
    requirements = PRIMA_REQUIREMENTS if family == "prima" else MEDGEMMA_REQUIREMENTS
    imports = {
        "prima": """import brain_vlm.prima.data, brain_vlm.prima.loading, brain_vlm.prima.patches
from generative.networks.nets.vqvae import Encoder
from transformers import GPT2Model, AutoTokenizer
""",
        "medgemma": """from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from lmformatenforcer import JsonSchemaParser
import accelerate, jsonschema, PIL, psutil
""",
    }
    script = "import sys;sys.path.insert(0," + repr(src) + ")\n"
    script += "REQUIREMENTS=" + repr(requirements) + "\n"
    script += """from importlib.metadata import version
from packaging.requirements import Requirement
for specification in REQUIREMENTS:
    r = Requirement(specification)
    if not r.specifier.contains(version(r.name), prereleases=True):
        raise RuntimeError('Package does not satisfy ' + specification + ': ' + version(r.name))
import torch, numpy
""" + imports[family] + """print('Required inference imports OK; no weights loaded.')
print('Torch:', torch.__version__, '| compiled CUDA:', torch.version.cuda)
"""
    _run_logged([str(python), "-c", script], log, family + " inference import check")


def prepare_environment(folder, family, install=True, log_dir=None):
    """Keep Colab Torch/CUDA/NumPy; record setup attempts and verify model imports.

    log_dir should be RUN_ROOT/logs on Drive. INSTALL_DEPS=False still verifies
    packages/imports. A successful setup does not establish GPU or model correctness.
    """
    require(family in {"prima", "medgemma"}, "Unknown worker family")
    root = Path(folder)
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    require(python.exists() or install, "Runtime missing; enable INSTALL_DEPS or supply worker Python paths")
    logs = Path(log_dir) if log_dir is not None else root.parent / "setup_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = logs / f"environment_{family}_{stamp}.log"
    record_path = log.with_suffix(".json")
    record = {"family": family, "install_requested": install, "worker_python": str(python),
              "log": str(log), "status": "setting_up", "model_weights_loaded": False}
    print("Environment log:", log, flush=True)
    try:
        if not python.exists():
            bootstrap_environment(root, family, logs)
        record["before"] = _inspect(python, log)
        for name in ("torch", "numpy"):
            require(record["before"]["packages"][name] is not None,
                    "Base runtime lacks " + name + "; use a Colab runtime with Torch/NumPy installed")
        if install:
            src = str(Path(__file__).resolve().parents[1])
            if family == "prima":
                script = "import sys;sys.path.insert(0," + repr(src) + ");from brain_vlm.prima.setup import install_dependencies;install_dependencies(" + repr(str(root / "constraints.txt")) + ")"
                command = [str(python), "-c", script]
            else:
                # Keep the same base Torch/NumPy policy as PRIMA; don't replace Colab CUDA.
                constraints = root / "constraints.txt"
                constraints.write_text("\n".join(name + "==" + record["before"]["packages"][name]
                                                   for name in ("torch", "numpy")) + "\n", encoding="utf-8")
                command = [str(python), "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
                           "-c", str(constraints), *MEDGEMMA_REQUIREMENTS]
            _run_logged(command, log, "Install " + family + " dependencies")
        record["after"] = _inspect(python, log)
        _check(python, family, str(Path(__file__).resolve().parents[1]), log)
        record["status"] = "ready"
        print(family, "environment ready; no weights loaded. Metadata:", record_path, flush=True)
        return str(python)
    except Exception as error:
        record.update(status="failed", error=_redact(str(error)))
        print("Environment setup failed. Log:", log, "| metadata:", record_path, flush=True)
        raise
    finally:
        write(record_path, record)
