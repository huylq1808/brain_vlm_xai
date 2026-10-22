"""M10: fixed-target group latent occlusion.

A group score is not a crop score. Groups follow token index order and may span
several disconnected regions; they are not lesion masks or verified grounding."""

import copy
import numpy as np
from brain_vlm.prima.models import get_study_embeddings
from brain_vlm.prima.artifacts import require


def occlusion_groups(encoder, inputs, text_vector, groups_per_series=4, spatial_dimensions=33):
    """Zero latent values in bounded token groups; preserve positions and input tensors."""

    require(
        isinstance(groups_per_series, int) and groups_per_series > 0,
        "Number of occlusion groups must be a positive integer",
    )
    require(isinstance(spatial_dimensions, int) and spatial_dimensions >= 3,
            "Provide the checkpoint position dimension plus three orientation values")
    target = np.array(text_vector, dtype=float, copy=True)
    require(
        target.ndim == 1 and np.isfinite(target).all() and np.linalg.norm(target) > 0,
        "Invalid target text vector",
    )
    target /= np.linalg.norm(target)

    def score(value):
        fresh = copy.deepcopy(value)  # HierViT mutates lens.
        _, clip = get_study_embeddings(encoder, fresh)
        require(clip.shape[1] == len(target), "Image/text dimension mismatch")
        return float(clip.numpy()[0] @ target)

    baseline = score(inputs)
    rows = []
    for sequence_index, visual in enumerate(inputs["visual"]):
        count = int(inputs["lenss"][0, sequence_index].item())
        require(0 < count <= visual.shape[1], "Invalid retained token count")
        for indices in np.array_split(np.arange(count), min(groups_per_series, count)):
            if not len(indices):
                continue
            edited = copy.deepcopy(inputs)
            # Keep position/orientation; ablate latent values only. ViT rejects all-zero tokens.
            latent_dim = visual.shape[-1] - spatial_dimensions
            if latent_dim != 256:
                raise ValueError(
                    "Unexpected token layout; inspect checkpoint patchifier before ablation"
                )
            edited["visual"][sequence_index][:, indices, :latent_dim] = 0
            rows.append(
                {
                    "series_index": sequence_index,
                    "token_indices": indices.tolist(),
                    "baseline": baseline,
                    "similarity_drop": baseline - score(edited),
                    "method": "group latent occlusion; attribution candidate, not verified grounding",
                }
            )
    return sorted(rows, key=lambda r: r["similarity_drop"], reverse=True)
