"""Official PRIMA patch extraction/filtering, with unused DICOM utilities removed.

Adapted under the upstream MIT license. Bucket ordering and duplicate hole-fill
coordinates are deliberately preserved to match existing embeddings.
"""

import time
import logging
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Resize
from brain_vlm.prima.geometry import adjusted_patch_shape

CHAR_TO_INDEX = {
    c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz_+-0123456789.*(),")
}


def percentile_mask(
    image: Union[np.ndarray, torch.Tensor], mask_threshold: int = 50
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply PRIMA's intensity threshold heuristic, not an Otsu estimator.

    Use threshold 50 for ordinary intensity ranges, or 0.5 when max < 5.

    Args:
        image: Input image as numpy array or torch tensor
        mask_threshold: Threshold value for masking

    Returns:
        Binary mask of same type as input
    """
    try:
        if isinstance(image, torch.Tensor):
            if image.max() < 5:
                mask = image > mask_threshold / 100
            else:
                mask = image > mask_threshold
        elif image.max() < 5:
            mask = image > mask_threshold / 100
        else:
            mask = image > mask_threshold
        return mask
    except Exception as e:
        raise RuntimeError(f"Failed to create percentile mask: {str(e)}")


def pad_volume_for_patches(
    volume: Union[np.ndarray, torch.Tensor], patch_size: List[int]
) -> torch.Tensor:
    """
    Pad the volume so that it can be evenly divided into patches.

    Args:
        volume: Input volume as numpy array or torch tensor
        patch_size: Size of patches to create

    Returns:
        Padded volume as torch tensor
    """
    try:
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume.astype(np.float32))
        pad_sizes = [(ps - s % ps) % ps for s, ps in zip(volume.shape, patch_size)]
        pad = []
        for p in pad_sizes[::-1]:
            pad.extend([p // 2, p - p // 2])
        padded_volume = F.pad(volume, pad)
        return padded_volume
    except Exception as e:
        raise RuntimeError(f"Failed to pad volume: {str(e)}")


def scale(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Scale input to [0, 1] range.

    Args:
        x: Input array or tensor

    Returns:
        Scaled array or tensor
    """
    try:
        max_x = x.max().item() if isinstance(x, torch.Tensor) else x.max()
        if max_x > 0:
            return x / max_x
        return x
    except Exception as e:
        raise RuntimeError(f"Failed to scale input: {str(e)}")


def tokenize_volume(
    volume: Union[np.ndarray, torch.Tensor], mask_perc: int = 50
) -> Tuple[
    List[torch.Tensor],
    List[Tuple[int, int, int]],
    List[float],
    Tuple[int, int, int],
    List[int],
    Optional[int],
]:
    """
    Chop a volume into patches and collect relevant information.

    Args:
        volume: Input volume as numpy array or torch tensor
        mask_perc: Percentage threshold for masking

    Returns:
        Tuple containing:
        - List of patches
        - List of patch coordinates
        - List of patch values
        - Volume shape
        - Patch shape
        - Z dimension index
    """
    try:
        start = time.time()
        img = volume
        patch_size, z_idx = adjusted_patch_shape(img.shape)
        logging.info(f"Patch shape is {patch_size}")
        padded_volume = pad_volume_for_patches(img, patch_size)
        z_patches = padded_volume.shape[0] // patch_size[0]
        y_patches = padded_volume.shape[1] // patch_size[1]
        x_patches = padded_volume.shape[2] // patch_size[2]
        mask_ = percentile_mask(padded_volume, mask_perc)
        scaled_padded_vol = scale(padded_volume)
        patches = []
        coordinates = []
        values_ = []
        for z in range(z_patches):
            for y in range(y_patches):
                for x in range(x_patches):
                    z_start = z * patch_size[0]
                    y_start = y * patch_size[1]
                    x_start = x * patch_size[2]
                    patch = scaled_padded_vol[
                        z_start : z_start + patch_size[0],
                        y_start : y_start + patch_size[1],
                        x_start : x_start + patch_size[2],
                    ]
                    otsu_test = mask_[
                        z_start : z_start + patch_size[0],
                        y_start : y_start + patch_size[1],
                        x_start : x_start + patch_size[2],
                    ]
                    patches.append(patch)
                    coordinates.append((z_start, y_start, x_start))
                    values_.append(np.mean(otsu_test.numpy()) * 100)
        elapsed_time = time.time() - start
        logging.info(
            f"Finished chopping volume into patches in {elapsed_time:.2f} seconds"
        )
        return (patches, coordinates, values_, padded_volume.shape, patch_size, z_idx)
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize volume: {str(e)}")


def resize_tokens_batch(
    tensor_list: List[torch.Tensor], patch_shape: List[int]
) -> List[torch.Tensor]:
    """
    Resize a batch of tokens to a target shape.

    Args:
        tensor_list: List of input tensors
        patch_shape: Target shape for resizing

    Returns:
        List of resized tensors
    """
    try:
        resize = Resize(spatial_size=patch_shape)
        batch_tensor = np.stack(tensor_list)
        resized_batch = resize(batch_tensor)
        return list(resized_batch)
    except Exception as e:
        raise RuntimeError(f"Failed to resize tokens batch: {str(e)}")


def filtercoords(meta, percentagetouse, embs, fillhole=True, debuginfo="None"):
    percentagemeta = meta["OtsuThresholds"]
    uses = []
    if fillhole:
        metadict = {}
        for idx in meta["emb_index"]:
            x, y, z = meta["emb_index"][idx]
            metadict[x * 1000000 + y * 1000 + z] = int(idx)
    for i in range(percentagetouse, 101):
        uses += percentagemeta[i]["OutfillCoords"]
        if fillhole and i <= 20:
            infillcoords = percentagemeta[i]["InfillCoords"]
            uses += [
                (metadict[b[0] * 1000000 + b[1] * 1000 + b[2]], b) for b in infillcoords
            ]
    useids = torch.LongTensor([u[0] for u in uses])
    embspos = torch.LongTensor([u[1] for u in uses])
    return (embs[useids], embspos, useids)


def chartovec(s: str) -> torch.Tensor:
    """Convert a string to a tensor of character indices.

    Args:
        s: Input string to convert

    Returns:
        Tensor of character indices, with unknown characters mapped to index 45
        and an end token (46) appended
    """
    ret = []
    for c in s.lower():
        try:
            ret.append(CHAR_TO_INDEX[c] + 1)
        except KeyError:
            ret.append(45)
    ret.append(46)
    return torch.LongTensor(ret)
