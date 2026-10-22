"""M0: native NIfTI QC, XY resampling, LPS orientation and coordinate mapping.

Native indices are XYZ; tokenizer coordinates are padded prepared ZYX.
The inverse affine maps between index frames; it does not register anatomy."""

from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from brain_vlm.prima.artifacts import sha256, require

PREPROCESSING_VERSION = "nifti_xy256_lps_nativez_adc_p995_10pct_v1"
ADC_FOREGROUND_FRACTION = 0.10




def native_qc(path):
    """Inspect a finite 3D native image and preserve its affine, units and input hash."""
    if not str(path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Current input adapter requires 3D NIfTI (.nii/.nii.gz), not DICOM: " + str(path))
    image = nib.load(path)
    a = np.asarray(image.dataobj, dtype=np.float32)
    if (
        a.ndim != 3
        or not np.isfinite(a).all()
        or not np.isfinite(image.affine).all()
        or np.linalg.det(image.affine[:3, :3]) == 0
    ):
        raise ValueError(f"Invalid 3D image: {path}")
    return {
        "shape_xyz": list(a.shape),
        "spacing": list(image.header.get_zooms()),
        "units": list(image.header.get_xyzt_units()),
        "affine_ras": image.affine.tolist(),
        "axis_codes": list(nib.aff2axcodes(image.affine)),
        "sha256": sha256(path),
        "percentiles": np.percentile(a, [0, 50, 90, 99.5, 100]).tolist(),
        "nonzero_fraction": float(np.mean(a != 0)),
        "warnings": (
            ["unknown_spatial_units"]
            if image.header.get_xyzt_units()[0] != "mm"
            else []
        ),
    }


def sitk_affine_ras(image):
    """Convert SimpleITK voxel XYZ to RAS world coordinates (SimpleITK uses LPS)."""
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(
        image.GetSpacing()
    )
    affine[:3, 3] = image.GetOrigin()
    return np.diag([-1, -1, 1, 1]) @ affine


def prepare_volumes(paths, names):
    """Prepare each sequence at 256 x 256 with native Z sampling and record geometry."""
    import SimpleITK as sitk
    from brain_vlm.prima.geometry import adjusted_patch_shape

    volumes, geometry = [], []
    if len(paths) != len(names):
        raise ValueError("Paths/names length mismatch")
    for path, name in zip(paths, names):
        native = native_qc(path)
        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
        size, spacing = image.GetSize(), image.GetSpacing()
        reference = sitk.Image((256, 256, size[2]), sitk.sitkFloat32)
        reference.SetOrigin(image.GetOrigin())
        reference.SetDirection(image.GetDirection())
        reference.SetSpacing(
            (size[0] * spacing[0] / 256, size[1] * spacing[1] / 256, spacing[2])
        )
        prepared = sitk.DICOMOrient(
            sitk.Resample(
                image,
                reference,
                sitk.Transform(),
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32,
            ),
            "LPS",
        )
        affine = sitk_affine_ras(prepared)
        shape_zyx = list(sitk.GetArrayFromImage(prepared).shape)
        patch_shape, thin_axis = adjusted_patch_shape(shape_zyx)
        if thin_axis is None:
            raise ValueError("Upstream patch-axis heuristic ambiguous for a 256-cube")
        padded = [((s + p - 1) // p) * p for s, p in zip(shape_zyx, patch_shape)]
        before = [(p - s) // 2 for p, s in zip(padded, shape_zyx)]
        transform = np.linalg.inv(np.asarray(native["affine_ras"])) @ affine
        native.update(
            sequence=name,
            prepared_shape_xyz=list(prepared.GetSize()),
            prepared_affine_ras=affine.tolist(),
            prepared_to_native_xyz=transform.tolist(),
            native_to_prepared_xyz=np.linalg.inv(transform).tolist(),
            original_patch_shape_zyx=list(patch_shape),
            pad_before_zyx=before,
            padded_shape_zyx=padded,
            preprocessing_version=PREPROCESSING_VERSION,
        )
        # Algebraic mapping check; not a validation of anatomical registration.
        points = np.array([[0, 0, 0, 1], [10, 20, 5, 1]], dtype=float).T
        native["roundtrip_index_error"] = float(
            np.max(np.abs(np.linalg.inv(transform) @ transform @ points - points))
        )
        geometry.append(native)
        volumes.append(prepared)
    return volumes, geometry


def relative_adc_mask(image, mask_threshold=50):
    # Foreground heuristic, not expert lesion mask or a physical ADC unit conversion.
    """Select ADC foreground using 10% of positive p99.5; not a lesion mask."""
    array = (
        image.detach().cpu().numpy()
        if isinstance(image, torch.Tensor)
        else np.asarray(image)
    )
    positive = array[np.isfinite(array) & (array > 0)]
    if not len(positive):
        raise ValueError("ADC has no positive signal")
    threshold = np.percentile(positive, 99.5) * ADC_FOREGROUND_FRACTION
    mask = array > threshold
    return (
        torch.from_numpy(mask).to(image.device)
        if isinstance(image, torch.Tensor)
        else mask
    )


def processed_box_to_native(geometry, start_zyx, end_zyx):
    # Token coords are in padded processed ZYX, before VQ-VAE resize.
    """Map a padded prepared ZYX box to clipped native XYZ bounds, end exclusive."""
    import itertools

    before = np.asarray(geometry["pad_before_zyx"])
    start = np.asarray(start_zyx) - before
    end = np.asarray(end_zyx) - before
    corners = np.array(
        [list(c)[::-1] + [1] for c in itertools.product(*zip(start, end))]
    ).T
    native = (np.asarray(geometry["prepared_to_native_xyz"]) @ corners)[:3]
    lower = np.floor(native.min(axis=1)).astype(int)
    upper = np.ceil(native.max(axis=1)).astype(int)
    shape = np.asarray(geometry["shape_xyz"])
    return np.maximum(0, lower).tolist(), np.minimum(shape, upper).tolist()
