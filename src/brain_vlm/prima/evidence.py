"""M5 candidate selection, M6 native evidence packaging and M7 review scaffold.

Active v4 selection lives in spatial.py; this module packages native evidence.
The historical midpoint helper remains only for compatibility, not active M5.
Neither path establishes a validated lesion ROI or an automatic medical verdict.
"""

from pathlib import Path
import re
import numpy as np
import nibabel as nib

from brain_vlm.prima.artifacts import read_json, require, sha256, write_json
from brain_vlm.prima.data import processed_box_to_native


def check_previous_review(run_dir, claim):
    """Stop before replacing evidence that belongs to a differently worded review."""
    path = Path(run_dir) / "verification_review.json"
    if path.is_file():
        require(
            read_json(path)["claim_text"] == claim["reviewed_text"],
            "Prior image review uses different text. Choose a new claim ID or C_ROOT.",
        )


def select_representative_context(ranking, geometry, target_series, margin=8):
    """M5 partial: midpoint token of the strongest positive target-sequence group.

    Margin is in native voxels. The score belongs to the complete ablated group;
    it must never be interpreted as importance of this selected token or crop.
    No annotation enters selection. Return None if no positive target group.
    """
    require(
        isinstance(margin, int) and margin >= 0,
        "Margin must be nonnegative native voxels",
    )
    positive = [
        row
        for row in ranking
        if row["series_index"] == target_series and row["similarity_drop"] > 0
    ]
    if not positive:
        return None
    best = max(positive, key=lambda row: row["similarity_drop"])
    geo = geometry[target_series]
    indices = best["token_indices"]
    require(indices, "Empty positive group")
    token_index = indices[len(indices) // 2]
    require(
        0 <= token_index < len(geo["retained_coords_padded_zyx"]),
        "Retained token index out of bounds",
    )
    start = np.asarray(geo["retained_coords_padded_zyx"][token_index], dtype=int)
    end = start + np.asarray(geo["original_patch_shape_zyx"])
    patch_low, patch_high = processed_box_to_native(geo, start, end)
    low = np.maximum(0, np.asarray(patch_low) - margin).astype(int)
    high = np.minimum(geo["shape_xyz"], np.asarray(patch_high) + margin).astype(int)
    require(np.all(high > low), "Empty native context box")
    return {
        "native_box_xyz": [low.tolist(), high.tolist()],
        "representative_patch_box_xyz": [patch_low, patch_high],
        "retained_token_index": token_index,
        "similarity_drop": best["similarity_drop"],
        "score_scope": "whole occluded group; not representative token/crop",
    }


def _render_context(images, crops, names, center, claim_id, destination):
    """Native axis panels with affine-derived orientation labels."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(names), 4, figsize=(15, 4 * len(names)), squeeze=False
    )
    opposite = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
    for row, (name, image, crop) in enumerate(zip(names, images, crops)):
        full = np.asarray(image.dataobj)
        data = np.asarray(crop.dataobj)
        codes = nib.aff2axcodes(image.affine)
        panels = [
            (full[:, :, center[2]], 0, 1, "Full native axis-2 context"),
            (data[:, :, data.shape[2] // 2], 0, 1, "Crop axis-2"),
            (data[:, data.shape[1] // 2, :], 0, 2, "Crop axis-1"),
            (data[data.shape[0] // 2, :, :], 1, 2, "Crop axis-0"),
        ]
        for column, (panel, horizontal, vertical, title) in enumerate(panels):
            axis = axes[row, column]
            vmin, vmax = np.percentile(panel[np.isfinite(panel)], [1, 99])
            axis.imshow(
                panel.T,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax if vmax > vmin else vmin + 1,
            )
            axis.set_title(f"{name} | {title}")
            axis.set_xlabel(f"{opposite[codes[horizontal]]} → {codes[horizontal]}")
            axis.set_ylabel(f"{opposite[codes[vertical]]} → {codes[vertical]}")
    figure.suptitle(
        f"{claim_id} | representative patch context; clinical review pending"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def package_native_evidence(
    claim, case, sample_root, series_names, context, run_dir, provenance=None
):
    """M6: save same-grid paired native crops, affines, context PNG and a manifest.

    A grid mismatch stops before writing crops; there is no implicit registration.
    An empty selection produces an unresolved manifest with no authoritative files.
    Historical filenames/boxes remain compatible; score_scope clarifies the old score.
    """
    run_dir = Path(run_dir)
    check_previous_review(run_dir, claim)
    evidence = {
        "claim_id": claim["claim_id"],
        "claim_text": claim["reviewed_text"],
        "target_sequence": claim["sequence"],
        "image_verification": "not_run",
        "status": "unresolved_no_positive_target_group",
        "method": "representative token of target-sequence group; not lesion localization",
        "files": [],
        "claim_scope": "proposed context only; finding identity and complete claim coverage remain unverified",
    }
    if provenance is not None:
        evidence["provenance"] = provenance
        for name in series_names:
            require(
                sha256(Path(sample_root) / case["images"][name])
                == provenance["inputs"][name],
                "Native image changed since Stage A",
            )
    if context is not None:
        target = nib.load(Path(sample_root) / case["images"][claim["sequence"]])
        images = [
            nib.load(Path(sample_root) / case["images"][name]) for name in series_names
        ]
        low, high = np.asarray(context["native_box_xyz"], dtype=int)
        require(
            np.all(low >= 0) and np.all(high <= target.shape) and np.all(high > low),
            "Native box out of bounds",
        )
        for image in images:
            require(
                image.shape == target.shape
                and np.allclose(image.affine, target.affine, atol=1e-4),
                "Paired sequence grid mismatch; explicit registration is required",
            )
        filenames = [
            f"{re.sub(r'[^A-Za-z0-9_-]', '_', name)}_native_context.nii.gz"
            for name in series_names
        ]
        require(len(set(filenames)) == len(filenames), "Sequence filenames collide")
        run_dir.mkdir(parents=True, exist_ok=True)
        crops = [
            image.slicer[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
            for image in images
        ]
        for filename, crop in zip(filenames, crops):
            nib.save(crop, run_dir / filename)
        center = ((low + high - 1) // 2).astype(int)
        _render_context(
            images,
            crops,
            series_names,
            center,
            claim["claim_id"],
            run_dir / "evidence_review.png",
        )
        evidence.update(context)
        evidence.update(
            status="candidate_requires_image_review",
            files=filenames,
            native_affine_ras=target.affine,
            world_center_ras_mm=nib.affines.apply_affine(target.affine, center),
            annotation_used_for_selection=False,
        )
    write_json(run_dir / "evidence.json", evidence)
    return evidence
