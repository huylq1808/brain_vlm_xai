"""Colab setup and small configuration files; independent of model loading.

Clone -> install only missing packages -> official setup/download -> read config.
This module uses the standard library until dependencies are explicitly installed.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

PRIMA_COMMIT = "604f1fc4dd7059c278d4d34bbbcaab3e404fbcd4"
PRIMA_URL = "https://github.com/MLNeurosurg/Prima.git"
INFERENCE_REQUIREMENTS = (
    "SimpleITK==2.5.3",
    "monai==1.5.1",
    "monai-generative==0.2.3",
    "transformers==4.49.0",
    "positional-encodings==6.0.3",
    "perceiver-pytorch==0.8.8",
    "einops>=0.8.1",
    "pathos==0.3.4",
    "natsort>=8.4",
    "pydicom==2.4.4",
    "nibabel>=5.3.2",
    "gdown>=5.2",
    "numpy>=1.24",
    "scipy>=1.10",
    "PyYAML>=6",
    "psutil>=5",
    "tqdm>=4.67",
    "matplotlib>=3.7",
)




def clone_prima(repo_root, commit=PRIMA_COMMIT):
    """Clone the official revision; refuse to reset an existing different checkout."""
    repo_root = Path(repo_root)
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    if not repo_root.exists():
        subprocess.run(["git", "clone", PRIMA_URL, str(repo_root)], check=True)
        subprocess.run(["git", "-C", str(repo_root), "checkout", commit], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(
            f"PRIMA revision {actual} differs from {commit}. Use a fresh PRIMA_ROOT."
        )
    changed = subprocess.check_output(
        ["git", "-C", str(repo_root), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if changed:
        raise RuntimeError(
            "Official PRIMA tracked files have local edits. Use a clean PRIMA_ROOT to preserve the baseline."
        )
    return actual


def install_dependencies(repo_root, constraints_path):
    """Keep Colab torch/CUDA and NumPy; install only missing/incompatible inference packages.

    Run before importing transformers/PRIMA in this kernel. If they were already
    imported under a different version, restart the runtime after installation.
    """
    from importlib.metadata import PackageNotFoundError, version
    from packaging.requirements import Requirement

    needed = []
    for specification in INFERENCE_REQUIREMENTS:
        requirement = Requirement(specification)
        try:
            if not requirement.specifier.contains(
                version(requirement.name), prereleases=True
            ):
                needed.append(specification)
        except PackageNotFoundError:
            needed.append(specification)
    constraints_path = Path(constraints_path)
    constraints_path.write_text(
        "\n".join(f"{name}=={version(name)}" for name in ("torch", "numpy")) + "\n",
        encoding="utf-8",
    )
    if needed:
        print("Installing inference packages:", needed, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "--upgrade-strategy",
                "only-if-needed",
                "-c",
                str(constraints_path),
                *needed,
            ],
            check=True,
        )
    else:
        print("Inference dependencies ready; pip skipped.")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'end-to-end_inference_pipeline'); "
            "from pipeline import Pipeline; from tools.models import ModelLoader; "
            "print('Official PRIMA inference imports OK')",
        ],
        cwd=repo_root,
        check=True,
    )
    return needed


def configure_prima(repo_root):
    """Make upstream Python modules importable, without constructing/loading a model."""
    repo_root = Path(repo_root).resolve()
    for path in (repo_root, repo_root / "end-to-end_inference_pipeline"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))




def read_configs(model_root):
    """Read official JSON configs and substitute checkpoint paths in memory only."""
    trained = Path(model_root) / "test/trained_models"
    prima = json.loads((trained / "prima_config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads(
        (trained / "tokenizer_model/config.json").read_text(encoding="utf-8")
    )
    prima_path = trained / "primafullmodel107.pt"
    tokenizer_path = trained / "tokenizer_model/vqvae_model_step16799.pth"
    for path in (prima_path, tokenizer_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing/empty checkpoint: {path}. Run the download cell or supply this file."
            )
    prima["full_model_ckpt"] = str(prima_path)
    tokenizer["vqvae_config"]["ckpt_path"] = str(tokenizer_path)
    return prima, tokenizer


def checkpoint_identity(prima_config, tokenizer_config, commit=PRIMA_COMMIT):
    """Practical file-stat cache key, matching existing runs; not a weight checksum."""

    def identity(path):
        path = Path(path)
        stat = path.stat()
        return {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    return {
        "prima": identity(prima_config["full_model_ckpt"]),
        "vqvae": identity(tokenizer_config["vqvae_config"]["ckpt_path"]),
        "prima_commit": commit,
    }
