"""Single-series patch dataset and upstream foreground occupancy metadata.

The default DataLoader collate format is retained because filtercoords consumes
that format. No lesion mask or report participates in this step.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.ndimage as ndi
import SimpleITK as sitk
from brain_vlm.prima.patches import tokenize_volume, resize_tokens_batch


class MrVoxelDataset(Dataset):

    def __init__(self, series_volumes, transform=None):
        self.series_volumes = series_volumes
        self.transform = transform

    def __len__(self):
        return len(self.series_volumes)

    def __getitem__(self, idx):
        volume = self.series_volumes[idx]
        if sitk is not None and hasattr(volume, "GetSize"):
            volume = np.asarray(sitk.GetArrayFromImage(volume), dtype=np.float64)
        tokens, coords, otsu, pad_shape, patch_shape, z_idx = tokenize_volume(
            volume, mask_perc=50
        )
        otsu_thresholds = generate_otsu_thresholds(coords, otsu, pad_shape, patch_shape)
        ser_emb_meta = {
            "PaddedVolShape": pad_shape,
            "PatchShape": patch_shape,
            "OtsuThresholds": otsu_thresholds,
            "emb_index": {idx: coord for idx, coord in enumerate(coords)},
        }
        if not tokens:
            print("No tokens found for a certain sequence in the study.")
            return (torch.tensor([]), ser_emb_meta)
        patch_shape[z_idx] = 8
        try:
            tokens = resize_tokens_batch(tokens, patch_shape)
        except Exception as e:
            print(f"Error resizing tokens for volume {idx}: {e}")
            return (torch.tensor([]), ser_emb_meta)
        return (torch.stack(tokens), ser_emb_meta)


def generate_otsu_thresholds(
    coordinates,
    otsu,
    vol_shape,
    patch_shape,
    find_holes=True,
    find_holes_threshold=20,
    step=1,
):
    thresholds = list(range(0, 102, step))
    otsu_dict = {}
    for threshold in thresholds:
        threshold_coords = [
            (idx, coordinates[idx])
            for idx, val in enumerate(otsu)
            if val >= threshold and val < threshold + step
        ]
        otsu_dict[threshold] = {}
        otsu_dict[threshold]["OutfillCoords"] = threshold_coords
        highlight_coords = [coord[1] for coord in threshold_coords]
        if find_holes:
            if threshold <= find_holes_threshold:
                filled_coords = find_fully_filled_patches(
                    create_filled_mask(
                        vol_shape, highlight_coords, patch_size=patch_shape
                    ),
                    patch_size=patch_shape,
                )
                otsu_dict[threshold]["InfillCoords"] = [
                    (z, y, x)
                    for z, y, x in filled_coords
                    if (z, y, x) not in highlight_coords
                ]
    return otsu_dict


def create_filled_mask(original_shape, highlight_coords, patch_size=(4, 32, 32)):
    mask = np.zeros(original_shape, dtype=np.bool_)
    for z, y, x in highlight_coords:
        mask[z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]] = True
    filled_mask = ndi.binary_fill_holes(mask)
    return filled_mask


def find_fully_filled_patches(filled_mask, patch_size=(4, 32, 32)):
    original_shape = filled_mask.shape
    new_coords = []
    for z in range(0, original_shape[0], patch_size[0]):
        for y in range(0, original_shape[1], patch_size[1]):
            for x in range(0, original_shape[2], patch_size[2]):
                patch = filled_mask[
                    z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]
                ]
                if np.average(patch) == 1:
                    new_coords.append((z, y, x))
    return new_coords
