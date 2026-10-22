"""Load tensor-only compact checkpoints; no official clone or full checkpoint needed."""

import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from brain_vlm.prima.architecture import HierViT, TextEncoder
from brain_vlm.prima.patchifier import MedicalImagePatchifier

FORMAT = "prima_encoders_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(bundle_root):
    """Read a completed export; an incomplete export has no manifest.json."""
    root = Path(bundle_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != FORMAT or manifest["validation"]["status"] != "passed":
        raise ValueError("Expected a validated prima_encoders_v1 export")
    for relative in manifest["files"]:
        if not (root / relative).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe bundle path: {relative}")
    return manifest


def check_file(root, manifest, relative, checksum=True):
    path = Path(root) / relative
    info = manifest["files"][relative]
    if not path.is_file() or path.stat().st_size != info["bytes"]:
        raise ValueError(f"Missing/incomplete exported file: {path}")
    if checksum and file_sha256(path) != info["sha256"]:
        raise ValueError(f"Exported file checksum mismatch: {path}")
    return path


def verify_bundle(bundle_root):
    """Verify compact files once, without importing/deserializing the original model."""
    manifest = read_manifest(bundle_root)
    for relative in manifest["files"]:
        check_file(bundle_root, manifest, relative)
    return manifest


def checkpoint_identity(bundle_root):
    """Content-based cache identity distinct from the legacy full-model file-stat key."""
    manifest = read_manifest(bundle_root)
    return {
        "format": FORMAT,
        "manifest_sha256": file_sha256(Path(bundle_root) / "manifest.json"),
        "weights": {name: info["sha256"] for name, info in manifest["files"].items()},
    }


def _payload(root, manifest, filename, verify):
    path = check_file(root, manifest, filename, checksum=verify)
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_visual_encoder(bundle_root, device="cuda", verify=True):
    """Load sequence/study encoders, trained image projection and saved positions."""
    manifest = read_manifest(bundle_root)
    payload = _payload(bundle_root, manifest, "visual.pt", verify)
    encoder = HierViT(manifest["architecture"]["visual"])
    encoder.load_state_dict(payload["state_dict"], strict=True)
    for path, position in payload["visual_positions"].items():
        module_path, attribute = path.rsplit(".", 1)
        setattr(encoder.get_submodule(module_path), attribute, position)
    # This CPU table is intentionally not moved to CUDA with the visual encoder.
    patchifier = MedicalImagePatchifier(payload["patch_position"])
    encoder.eval().requires_grad_(False).to(device)
    return encoder, patchifier


class TokenizerEncoder(nn.Module):
    """Only the VQ-VAE encoder used by PRIMA's encode(); no quantizer or decoder."""

    def __init__(self, spec):
        super().__init__()
        from generative.networks.nets.vqvae import Encoder

        self.encoder = Encoder(**spec)

    def encode(self, patches):
        return self.encoder(patches)

    def forward(self, patches):
        return self.encode(patches)


def load_tokenizer_encoder(bundle_root, device="cuda", verify=True):
    import os
    progress = os.environ.get("BRAIN_VLM_PROGRESS") == "1"
    if progress:print(f"Loading/verifying frozen VQ encoder from {bundle_root} on {device} …", flush=True)
    manifest = read_manifest(bundle_root)
    payload = _payload(bundle_root, manifest, "vq_encoder.pt", verify)
    encoder = TokenizerEncoder(manifest["architecture"]["vq_encoder"])
    encoder.encoder.load_state_dict(payload["state_dict"], strict=True)
    encoder = encoder.eval().requires_grad_(False).to(device)
    if progress:
        from brain_vlm.prima.progress import attach_vq_progress
        attach_vq_progress(encoder.encoder)
        print("Frozen VQ ready. Next: CPU volume preparation/patch metadata, then VQ batches.", flush=True)
    return encoder


def load_text_encoder(bundle_root, device="cuda", dtype=None, verify=True):
    """Load the trained text branch and exported vocabulary completely offline."""
    from transformers import AutoTokenizer

    manifest = read_manifest(bundle_root)
    payload = _payload(bundle_root, manifest, "text.pt", verify)
    spec = manifest["architecture"]["text"]
    text = TextEncoder(spec["gpt2"], spec["feature_dim"])
    text.load_state_dict(payload["state_dict"], strict=True)
    for name in manifest["files"]:
        if name.startswith("text_tokenizer/"):
            check_file(bundle_root, manifest, name, checksum=verify)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(bundle_root) / "text_tokenizer"), local_files_only=True
    )
    dtype = dtype or (
        torch.float16 if str(device).startswith("cuda") else torch.float32
    )
    text.eval().requires_grad_(False).to(device=device, dtype=dtype)
    maximum = min(spec["max_length"], text.model.config.n_positions)
    import transformers

    info = {
        "tokenizer": spec["tokenizer_name"],
        "max_length": maximum,
        "dtype": str(dtype),
        "text_pooling": "last_valid_token_preserve_dtype_v1",
        "transformers_version": transformers.__version__,
        "checkpoint_format": FORMAT,
    }
    return text, tokenizer, maximum, info
