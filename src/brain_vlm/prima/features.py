"""M1: frozen visual features, token QC and resumable per-study runs.

Tokens are VQ-VAE latents with coordinates, not contextual HierViT features.
The preprocessing version and file hashes are part of the cache contract."""

from pathlib import Path
import json
from unittest.mock import patch
import torch
import numpy as np
from brain_vlm.prima.artifacts import read_json, require, sha256, write_json, save_tokens, save_embeddings, validate_embedding
from brain_vlm.prima.data import PREPROCESSING_VERSION, prepare_volumes, relative_adc_mask
from brain_vlm.prima.models import prepare_inputs, get_study_embeddings

STAGE_A_FILES = (
    "geometry.json",
    "token_qc.json",
    "tokenizer_embeddings.pt",
    "study_embedding.npy",
    "clip_embedding.npy",
    "provenance.json",
)


def check_saved_study(
    case_id,
    case,
    sample_root,
    output_root,
    checkpoint_identity,
    study_description="MRI BRAIN",
    chunk_size=16,
    embedding_dimensions=(10240, 128),
):
    """Validate a saved case on CPU before loading GPU models or reusing its tokens.

    Raises on missing/stale artifacts, input changes or checkpoint/config mismatch.
    Dimensions are supplied by the loaded trained projection; the standalone
    default matches the public PRIMA checkpoint (10240 and 128).
    """
    folder = Path(output_root) / case_id
    for filename in STAGE_A_FILES:
        require(
            (folder / filename).is_file(), f"Missing A artifact: {folder / filename}"
        )
    provenance = read_json(folder / "provenance.json")
    expected = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "checkpoint": checkpoint_identity,
        "study_description": study_description,
        "chunk_size": chunk_size,
        "inputs": {
            name: sha256(Path(sample_root) / relative)
            for name, relative in case["images"].items()
        },
    }
    require(
        provenance == expected,
        f"A provenance mismatch: {case_id}. Check paths/config before recomputing.",
    )
    validate_embedding(
        np.load(folder / "study_embedding.npy", allow_pickle=False), embedding_dimensions[0]
    )
    validate_embedding(
        np.load(folder / "clip_embedding.npy", allow_pickle=False), embedding_dimensions[1], True
    )
    geometry = read_json(folder / "geometry.json")
    qc = read_json(folder / "token_qc.json")
    names = list(case["images"])
    require(
        names
        == [item["sequence"] for item in geometry]
        == [item["sequence"] for item in qc],
        f"A sequence order mismatch: {case_id}",
    )
    return provenance


def load_saved_study(
    case_id, case, sample_root, output_root, checkpoint_identity, pipeline, device,
    embedding_dimensions=(10240, 128),
):
    """Reuse trusted local token tensors and check geometry before a C experiment."""
    provenance = check_saved_study(
        case_id,
        case,
        sample_root,
        output_root,
        checkpoint_identity,
        pipeline.config.study_description,
        pipeline.config.max_tokens_per_chunk,
        embedding_dimensions,
    )
    folder = Path(output_root) / case_id
    record = torch.load(
        folder / "tokenizer_embeddings.pt", map_location="cpu", weights_only=True
    )
    geometry = read_json(folder / "geometry.json")
    qc = read_json(folder / "token_qc.json")
    require(
        record["series_names"] == [item["sequence"] for item in geometry],
        "Token/geometry order mismatch",
    )
    require(
        all(
            item["filter_percentage"] == 5 and not item["fallback_all_tokens"]
            for item in qc
        ),
        "The focal attribution pilot requires filter 5 without fallback; review QC first.",
    )
    inputs = prepare_inputs(pipeline, record, device)
    for index, item in enumerate(geometry):
        require(
            int(inputs["lenss"][0, index]) == len(item["retained_token_indices"]),
            "Retained token count differs from prepared input",
        )
    return record, geometry, inputs, provenance




def tokenize_with_qc(pipeline, tokenizer, volumes, names, geometry):
    """Encode each sequence with official VQ-VAE and preserve pre-resize patch coordinates."""
    VolUtils = pipeline.vol_utils

    require(
        len(volumes) == len(names) == len(geometry) > 0,
        "Volume/name/geometry length mismatch",
    )
    tokens, metadata, qc = [], [], []
    for volume, name, geo in zip(volumes, names, geometry):
        pipeline.tokenizer_model = tokenizer
        original_mask = VolUtils.percentile_mask
        mask_function = relative_adc_mask if name.lower() == "adc" else original_mask
        with patch.object(VolUtils, "percentile_mask", new=mask_function):
            result, encoded, metas = pipeline.run_tokenizer_model(
                [volume], series_names=[name]
            )
        if encoded != [name]:
            raise RuntimeError(f"Tokenization failed for {name}")
        meta = metas[0]
        # Upstream mutates PatchShape when resizing 4->8; coordinates refer to original patches.
        meta["PatchShape"] = geo["original_patch_shape_zyx"]
        tensor = result[0]
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite tokens: {name}")
        filtercoords = pipeline.filtercoords

        chosen = None
        for percentage in range(5, -1, -1):
            kept, coords, indices = filtercoords(meta, percentage, tensor)
            if len(coords) > 25:
                chosen = percentage
                break
        if chosen is None:
            raise ValueError(f"Insufficient foreground tokens: {name}")
        geo["retained_coords_padded_zyx"] = coords
        geo["retained_token_indices"] = indices
        qc.append(
            {
                "sequence": name,
                "raw_tokens": len(tensor),
                "retained_tokens": len(kept),
                "filter_percentage": chosen,
                "mask_policy": (
                    "positive_p995_10pct" if name.lower() == "adc" else "upstream"
                ),
                "fallback_all_tokens": chosen == 0,
                "coordinate_count_matches": len(meta["emb_index"]) == len(tensor),
            }
        )
        if not qc[-1]["coordinate_count_matches"]:
            raise ValueError("Token/coordinate mismatch")
        tokens.append(tensor)
        metadata.append(meta)
    return {
        "series_tokens": tokens,
        "series_names": names,
        "token_metadata": metadata,
    }, qc


def encode_study(
    case_id,
    case,
    sample_root,
    output_root,
    pipeline,
    tokenizer,
    encoder,
    checkpoint_identity,
    overwrite=False,
):
    """M0 -> M1 for one study; reuse matching cache or save tokens, geometry, h and z.

    The provenance marker is written last. A failed recomputation must not leave
    an old marker identifying partially replaced outputs as a complete cache.
    """
    folder = Path(output_root) / case_id
    expected = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "checkpoint": checkpoint_identity,
        "study_description": pipeline.config.study_description,
        "chunk_size": pipeline.config.max_tokens_per_chunk,
        "inputs": {
            name: sha256(Path(sample_root) / rel)
            for name, rel in case["images"].items()
        },
    }
    head = encoder.outerViT.mlp_head[0]
    if not overwrite and (folder / "provenance.json").is_file():
        previous = json.loads((folder / "provenance.json").read_text())
        if previous == expected and all(
            (folder / f).is_file()
            for f in [
                "geometry.json",
                "token_qc.json",
                "tokenizer_embeddings.pt",
                "study_embedding.npy",
                "clip_embedding.npy",
            ]
        ):
            validate_embedding(
                np.load(folder / "study_embedding.npy"), head.in_features
            )
            validate_embedding(
                np.load(folder / "clip_embedding.npy"), head.out_features, True
            )
            return {"case_id": case_id, "status": "cached"}
    names = list(case["images"])
    require(names, f"No image sequences: {case_id}")
    (folder / "provenance.json").unlink(missing_ok=True)
    volumes, geometry = prepare_volumes(
        [Path(sample_root) / p for p in case["images"].values()], names
    )
    record, qc = tokenize_with_qc(pipeline, tokenizer, volumes, names, geometry)
    save_tokens(folder, record)
    write_json(folder / "geometry.json", geometry)
    write_json(folder / "token_qc.json", qc)
    inputs = prepare_inputs(pipeline, record, pipeline.config.device)
    h, z = get_study_embeddings(encoder, inputs)
    save_embeddings(folder, h, z)
    write_json(folder / "provenance.json", expected)
    return {"case_id": case_id, "status": "success", "token_qc": qc}
