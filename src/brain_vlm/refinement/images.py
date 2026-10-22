"""Render explicitly selected native MRI slices to model-input PNGs; no annotation selection."""
import hashlib
from pathlib import Path
from brain_vlm.refinement.contracts import require

def render_native_slices(series_paths, destination, slice_indices, max_slices=5, intensity_bounds=None, include_labels=True):
    """Render selected axial views of the supplied native volume or native crop."""
    import nibabel as nib
    import numpy as np
    from PIL import Image, ImageDraw

    require(bool(series_paths), "No sequences supplied")
    require(type(max_slices) is int and max_slices > 0, "Invalid slice budget")
    images = {seq: nib.as_closest_canonical(nib.load(str(path))) for seq, path in series_paths.items()}
    reference = next(iter(images.values()))
    require(len(reference.shape) == 3, "Expected 3D native MRI")
    require(nib.aff2axcodes(reference.affine) == ("R", "A", "S"), "Expected canonical RAS axes")
    for image in images.values():
        require(image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-4),
                "Sequence grids differ; explicit registration is required")
    indices = list(slice_indices)
    require(indices and len(indices) <= max_slices and len(indices) == len(set(indices)), "Invalid slice list")
    require(all(type(z) is int and 0 <= z < reference.shape[2] for z in indices), "Slice index out of range")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    evidence = []
    for seq, image in images.items():
        data = np.asarray(image.dataobj, dtype=np.float32)
        require(np.isfinite(data).all(), "Non-finite native intensity")
        foreground = data[data > 0]
        require(foreground.size > 0, "No positive image intensity")
        low, high = np.percentile(foreground, [1, 99]) if intensity_bounds is None else intensity_bounds
        require(np.isfinite([low, high]).all(), "Non-finite display window")
        require(high > low, "Degenerate image normalization")
        source_hash = hashlib.sha256(Path(series_paths[seq]).read_bytes()).hexdigest()
        for z in indices:
            plane = np.flipud(data[:, :, z].T)
            display = np.clip((plane - low) / (high - low), 0, 1)
            canvas = Image.new("RGB", (display.shape[1] + 40, display.shape[0] + 60), "black")
            canvas.paste(Image.fromarray((display * 255).astype(np.uint8)).convert("RGB"), (20, 35))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 4), f"{seq.upper()} | axial RAS z={z}", fill="white")
            draw.text((canvas.width // 2, 21), "A", fill="white")
            draw.text((canvas.width // 2, canvas.height - 18), "P", fill="white")
            draw.text((3, canvas.height // 2), "L", fill="white")
            draw.text((canvas.width - 15, canvas.height // 2), "R", fill="white")
            content_box = [20,35,20+display.shape[1],35+display.shape[0]]
            if not include_labels:
                canvas = canvas.crop(tuple(content_box))
                content_box = [0,0,display.shape[1],display.shape[0]]
            filename = f"{seq}_axial_{z:03d}.png"
            path = destination / filename
            canvas.save(path)
            evidence.append({"evidence_id": f"{seq}_axial_{z:03d}", "sequence": seq,
                             "kind": "native_slice", "scope": "sampled_context",
                             "path": f"images/{filename}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "orientation_verified": True, "plane": "axial", "canonical_ras_z": z,
                             "world_z_mm": float(nib.affines.apply_affine(image.affine, [0, 0, z])[2]),
                             "display_convention": "neurological: patient L left, R right, A top, P bottom",
                             "labels_in_pixels":include_labels,"content_box_pixels":content_box,
                             "normalization": {"method": "per-volume positive-voxel p1/p99" if intensity_bounds is None else "shared source-volume positive-voxel p1/p99", "low": float(low), "high": float(high)},
                             "source_sha256": source_hash, "selection": "explicit_native_slices",
                             "mask_overlay": False})
    return evidence
