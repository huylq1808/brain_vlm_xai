"""Artifact I/O and embedding contracts. No model weights are loaded here."""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch


def read_json(path):
    """Read a JSON artifact without relying on the notebook working directory."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    """Fail an artifact contract even when Python runs with assertions disabled."""
    if not condition:
        raise ValueError(message)


def validate_embedding(value, dimension, unit_norm=False):
    """Validate one finite, nonzero (1, dimension) vector; optionally require L2 norm 1."""
    a = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if a.shape != (1, dimension) or not np.isfinite(a).all() or np.linalg.norm(a) == 0:
        raise ValueError(
            f"Invalid embedding; expected finite, nonzero (1, {dimension}), got {a.shape}"
        )
    if unit_norm and not np.isclose(np.linalg.norm(a), 1.0, atol=1e-5):
        raise ValueError("CLIP embedding must have unit norm")
    return a


def save_tokens(folder, token_record):
    """Save CPU-only plain tensors so Stage C can use weights_only=True loading."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(token_record)
    record["series_tokens"] = [
        (t.as_tensor() if hasattr(t, "as_tensor") else t).detach().cpu()
        for t in token_record["series_tokens"]
    ]
    torch.save(record, folder / "tokenizer_embeddings.pt")


def save_embeddings(folder, pooled, clip):
    """Save pooled study and normalized CLIP image vectors as NumPy arrays."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "study_embedding.npy", pooled.detach().cpu().numpy())
    np.save(folder / "clip_embedding.npy", clip.detach().cpu().numpy())


def write_json(path, value):
    """Write UTF-8 JSON, converting tensors, arrays, NumPy scalars and paths."""

    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    """Hash file contents in chunks; used for data and small artifact integrity checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
