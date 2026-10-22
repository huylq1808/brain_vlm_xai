"""One-time conversion of trusted official checkpoints into inference-only tensors.

This module is separate from loading.py. Only export_from_official imports the
official clone and deserializes its full-object checkpoint. The published bundle
contains no pickled PRIMA modules, downstream heads, VQ decoder or codebook.
"""

from copy import deepcopy
from functools import partial
import gc
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
import torch.nn.functional as F
from brain_vlm.prima.architecture import HierViT, TextEncoder
from brain_vlm.prima.configuration import describe_visual, describe_text, describe_vq_encoder
from brain_vlm.prima.loading import FORMAT, TokenizerEncoder, file_sha256, load_visual_encoder, load_tokenizer_encoder, load_text_encoder


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _copy_state(source, target):
    """Copy only required keys into independent CPU storages, without precision loss."""
    original = source.state_dict()
    required = target.state_dict()
    missing = set(required) - set(original)
    if missing:
        raise ValueError(f"Missing trained tensors: {sorted(missing)}")
    tensors = {}
    for key, expected in required.items():
        value = original[key]
        if value.shape != expected.shape:
            raise ValueError(f"Tensor shape mismatch: {key}")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(
                f"Expected original FP32 weights; got {value.dtype}: {key}"
            )
        # clone() avoids saving a small view together with its large backing storage.
        tensors[key] = value.detach().cpu().clone()
    target.load_state_dict(tensors, strict=True)
    return tensors, sorted(set(original) - set(required))


def _positions(visual):
    positions = {}
    for name, module in visual.named_modules():
        for attribute in ("p_enc", "pos_embedding"):
            value = getattr(module, attribute, None)
            if isinstance(value, torch.Tensor):
                positions[f"{name}.{attribute}"] = value.detach().cpu().clone()
    return positions


def _tensor_stats(tensors):
    values = list(tensors.values())
    return {
        "tensor_elements": sum(t.numel() for t in values),
        "tensor_bytes": sum(t.numel() * t.element_size() for t in values),
    }


def _comparison(reference, actual, label, atol=1e-6, rtol=1e-5):
    reference, actual = reference.float().cpu(), actual.float().cpu()
    torch.testing.assert_close(
        actual, reference, atol=atol, rtol=rtol, msg=lambda msg: f"{label}: {msg}"
    )
    return {
        "test": label,
        "shape": list(actual.shape),
        "max_abs_error": float((actual - reference).abs().max()),
        "atol": atol,
        "rtol": rtol,
        "passed": True,
    }


def validate_export(bundle_root, full_model, vqvae, original_tokenizer):
    """Compare reloaded branches with the source on deterministic synthetic inputs.

    Tests all three patch orientations, patchifier outputs, multi-series pooled
    and projected image vectors, and real-tokenizer text vectors. This verifies
    conversion equivalence; it does not evaluate clinical accuracy on MRI data.
    """
    from brain_vlm.prima.models import restore_text_runtime_metadata
    from brain_vlm.prima.patches import chartovec

    visual, patchifier = load_visual_encoder(bundle_root, "cpu")
    vq = load_tokenizer_encoder(bundle_root, "cpu")
    text, tokenizer, max_length, _ = load_text_encoder(bundle_root, "cpu")
    source_visual = full_model.clipvisualmodel.eval()
    source_visual.patdis = False
    source_visual.getserieemb = False
    source_visual.retboth = False
    source_visual.make_no_flashattn()
    source_text = full_model.clipmodel.text_model.eval()
    restore_text_runtime_metadata(source_text)
    checks = []
    generator = torch.Generator().manual_seed(107)
    latents, coordinates = [], []
    with torch.inference_mode():
        for index, shape in enumerate(((8, 32, 32), (32, 8, 32), (32, 32, 8))):
            x = torch.rand(1, 1, *shape, generator=generator)
            before, after = vqvae.encode(x), vq.encode(x)
            checks.append(_comparison(before, after, f"vq_orientation_{index}"))
            latents.append(before.cpu())
            coordinates.append(torch.zeros(1, 3, dtype=torch.long))
        # Use multiple non-zero coordinates for each orientation as well.
        extended = [latent.repeat(3, 1, 1, 1, 1) for latent in latents]
        coordinates = [
            torch.tensor([[0, 0, 0], [4, 32, 32], [8, 64, 64]]),
            torch.tensor([[0, 0, 0], [32, 4, 32], [64, 8, 64]]),
            torch.tensor([[0, 0, 0], [32, 32, 4], [64, 64, 8]]),
        ]
        before = full_model.clipmodel.patchifier(extended, coordinates)
        after = patchifier(extended, coordinates)
        for i, (a, b) in enumerate(zip(before, after)):
            checks.append(_comparison(a, b, f"spatial_features_{i}", atol=0, rtol=0))
        if any(x.shape[-1] > visual.innerViT.dim for x in after):
            raise ValueError(
                "Patch feature dimension exceeds the trained sequence ViT dimension"
            )
        for count in (1, 3):
            names = [chartovec(n) for n in ("t1", "flair", "adc")[:count]]
            padded = torch.zeros(1, count, max(map(len, names)), dtype=torch.long)
            for i, name in enumerate(names):
                padded[0, i, : len(name)] = name
            inputs = {
                "visual": [x.unsqueeze(0) for x in after[:count]],
                "lens": torch.tensor([count]),
                "lenss": torch.full((1, count), 3),
                "serienames": padded,
                "studydescription": chartovec("MRI BRAIN")[None],
                "hash": ["export_test"],
            }
            h0 = source_visual(deepcopy(inputs), retpool=True)
            h1 = visual(deepcopy(inputs), retpool=True)
            checks.append(_comparison(h0, h1, f"study_embedding_{count}_series"))
            z0 = F.normalize(source_visual.outerViT.mlp_head(h0), dim=-1)
            z1 = F.normalize(visual.outerViT.mlp_head(h1), dim=-1)
            checks.append(_comparison(z0, z1, f"clip_image_{count}_series"))
        for sentence in (
            "No acute infarction.",
            "Right frontal T2 FLAIR hyperintense lesion.",
        ):
            original_ids = original_tokenizer(sentence + "<|endoftext|>")["input_ids"]
            ids = tokenizer(sentence + "<|endoftext|>")["input_ids"]
            if ids != original_ids:
                raise ValueError("Exported text tokenizer changed token IDs")
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = tokenizer.eos_token_id
            x = torch.tensor([ids])
            hidden = source_text.model(
                x, use_cache=False, return_dict=True
            ).last_hidden_state
            z0 = F.normalize(source_text.mlp_head(hidden[:, -1]).float(), dim=-1)
            z1 = F.normalize(text(x, torch.tensor([len(ids)])).float(), dim=-1)
            checks.append(_comparison(z0, z1, f"clip_text_{len(checks)}"))
    return {
        "status": "passed",
        "scope": "FP32 CPU conversion parity on synthetic inputs",
        "checks": checks,
    }


def export_models(
    full_model,
    vqvae,
    tokenizer_config,
    text_tokenizer,
    destination,
    source_identity=None,
):
    """Export already loaded trusted models. Publish only after reloading and parity checks.

    Refuse an existing destination. Use a new versioned folder for a new export;
    original checkpoints and previously completed exports are never overwritten.
    All original models must be on CPU to keep Colab VRAM free during conversion.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    clip, visual = full_model.clipmodel, full_model.clipvisualmodel
    if any(p.device.type != "cpu" for p in visual.parameters()) or any(
        p.device.type != "cpu" for p in clip.text_model.parameters()
    ):
        raise ValueError("Export on CPU; do not move the full checkpoint onto GPU")
    visual_spec = describe_visual(visual)
    text_spec = describe_text(clip.text_model)
    text_spec.update(
        max_length=clip.config["data"]["text_max_tokens"],
        tokenizer_name=clip.config["data"]["tokenizer"],
    )
    vq_spec = describe_vq_encoder(tokenizer_config)
    # Construct only lean branches. No training/clinical heads are constructed.
    lean_visual = HierViT(visual_spec)
    visual_state, removed_visual = _copy_state(visual, lean_visual)
    positions = _positions(visual)
    # Omit auxiliary positions belonging to an inactive name encoder.
    positions = {
        k: v
        for k, v in positions.items()
        if k.rsplit(".", 1)[0] in dict(lean_visual.named_modules())
    }
    vq_encoder = TokenizerEncoder(vq_spec)
    vq_state, _ = _copy_state(vqvae.encoder, vq_encoder.encoder)
    lean_text = TextEncoder(text_spec["gpt2"], text_spec["feature_dim"])
    text_state, removed_text = _copy_state(clip.text_model, lean_text)
    del lean_visual, vq_encoder, lean_text
    gc.collect()
    patch_position = clip.patchifier.p_enc.detach().cpu().reshape(1000000, -1).clone()
    if text_spec["feature_dim"] != visual_spec["outer"]["num_classes"]:
        raise ValueError("Image/text projections must share the same dimension")
    if (
        text_tokenizer.eos_token_id is None
        or len(text_tokenizer) > text_spec["gpt2"]["vocab_size"]
    ):
        raise ValueError("Text tokenizer vocabulary does not match this checkpoint")
    manifest = {
        "format": FORMAT,
        "source": source_identity or {},
        "architecture": {
            "visual": visual_spec,
            "vq_encoder": vq_spec,
            "text": text_spec,
        },
        "removed": {
            "full_model": [
                "diagnosis heads",
                "referral heads",
                "priority head",
                "training temperatures/losses",
            ],
            "vqvae": ["decoder", "quantizer/codebook"],
            "visual_state_keys": removed_visual,
            "text_legacy_buffers": removed_text,
        },
        "components": {
            "visual": dict(
                _tensor_stats(visual_state),
                parameters=sum(t.numel() for k, t in visual_state.items()),
            ),
            "vq_encoder": _tensor_stats(vq_state),
            "text": _tensor_stats(text_state),
            "positions": _tensor_stats(dict(positions, patch_position=patch_position)),
        },
        "files": {},
        # This provisional manifest exists only inside our temporary staging folder.
        "validation": {"status": "passed", "scope": "staging only"},
    }
    with tempfile.TemporaryDirectory(
        prefix=".prima_export_", dir=destination.parent
    ) as stage:
        stage = Path(stage)
        torch.save(
            {
                "state_dict": visual_state,
                "visual_positions": positions,
                "patch_position": patch_position,
            },
            stage / "visual.pt",
        )
        torch.save({"state_dict": vq_state}, stage / "vq_encoder.pt")
        torch.save({"state_dict": text_state}, stage / "text.pt")
        text_tokenizer.save_pretrained(stage / "text_tokenizer")
        del visual_state, vq_state, text_state, positions, patch_position
        gc.collect()
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest["files"][path.relative_to(stage).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
        _write_json(stage / "manifest.json", manifest)
        manifest["validation"] = validate_export(
            stage, full_model, vqvae, text_tokenizer
        )
        manifest["total_file_bytes"] = sum(
            item["bytes"] for item in manifest["files"].values()
        )
        _write_json(stage / "manifest.json", manifest)
        # Final directory only appears after successful validation. The temp parent
        # remains for TemporaryDirectory cleanup after moving its child contents.
        destination.mkdir()
        for path in stage.iterdir():
            if path.name != "manifest.json":
                path.rename(destination / path.name)
        (stage / "manifest.json").rename(destination / "manifest.json")
    return manifest


def export_from_official(repo_root, model_root, destination):
    """Convert the public full PRIMA and VQ-VAE checkpoints using their official loader.

    torch.load(weights_only=False) occurs only here, for the trusted source.
    mmap keeps clinical head tensor pages lazy; export reads only required weights.
    """
    from brain_vlm.prima import runtime

    runtime.clone_prima(repo_root)  # Verify the pinned, unmodified source revision.
    runtime.configure_prima(repo_root)
    prima_config, tokenizer_config = runtime.read_configs(model_root)
    from tools.models import ModelLoader
    from transformers import AutoTokenizer

    original_load = torch.load
    with patch("torch.load", new=partial(original_load, mmap=True)), patch(
        "torch.cuda.is_available", return_value=False
    ):
        full = ModelLoader.load_full_prima_model(prima_config)
    data = full.clipmodel.config["data"]
    tokenizer_name = {"biomed": "stanford-crfm/BioMedLM", "gpt2": "gpt2"}.get(
        data["tokenizer"]
    )
    if tokenizer_name is None:
        raise ValueError(f"Unknown tokenizer: {data['tokenizer']}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vqvae = ModelLoader.load_vqvae_model(tokenizer_config).eval().requires_grad_(False)
    try:
        return export_models(
            full,
            vqvae,
            tokenizer_config,
            tokenizer,
            destination,
            runtime.checkpoint_identity(prima_config, tokenizer_config),
        )
    finally:
        del full, vqvae
        gc.collect()
