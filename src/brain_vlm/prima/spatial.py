"""Bounded M5/M10: score spatial cells, never an index-group midpoint.

Cells are proposals, not lesion masks. Same header grids permit shared index
boxes; they do not certify registration, finding identity or clinical truth.
"""
from copy import deepcopy
from itertools import product
import numpy as np
from brain_vlm.prima.artifacts import require
from brain_vlm.prima.data import processed_box_to_native

VERSION = "spatial_octree_occlusion_v1"


def token_boxes(geometry):
    starts = geometry["retained_coords_padded_zyx"]
    shape = np.asarray(geometry["original_patch_shape_zyx"])
    return np.asarray([processed_box_to_native(geometry, start, np.asarray(start)+shape)
                       for start in starts], dtype=int)


def same_grid(left, right):
    return (left["shape_xyz"] == right["shape_xyz"]
            and left.get("units") == right.get("units")
            and np.allclose(left["affine_ras"], right["affine_ras"], atol=1e-4, rtol=0))


def children(box):
    low, high = np.asarray(box, dtype=int)
    middle = (low+high)//2
    axes = [[(int(a),int(m)),(int(m),int(b))] if a < m < b else [(int(a),int(b))]
            for a,m,b in zip(low,middle,high)]
    return [[[v[0] for v in cell],[v[1] for v in cell]] for cell in product(*axes)]


def search(geometry, score, reference_sequence=None):
    """8 coarse cells, then children of up to2 positive parents (<=24 scores).

    score receives {series_index: token_indices}. All same-grid sequences are
    occluded in the same region. Centers assign each retained token once per
    level; patch support can extend beyond cell edges and is recorded separately.
    No report signal, annotation, handpicked slice or anatomy-coordinate rule.
    """
    require(geometry, "Spatial search needs geometry")
    names = [g["sequence"] for g in geometry]
    reference_sequence = reference_sequence or next((s for s in ["flair","t2","t1"] if s in names),names[0])
    ref = names.index(reference_sequence)
    linked = [i for i,g in enumerate(geometry) if same_grid(geometry[ref],g)]
    boxes = {i:token_boxes(geometry[i]) for i in linked}
    require(all(len(b) for b in boxes.values()), "Spatial search has no retained tokens")
    centers = {i:b.mean(axis=1) for i,b in boxes.items()}
    extent = [boxes[ref][:,0].min(axis=0).tolist(),boxes[ref][:,1].max(axis=0).tolist()]
    rows = []
    def evaluate(cell,parent,depth):
        lo,hi = np.asarray(cell)
        indices = {i:np.flatnonzero(np.all((c>=lo)&(c<hi),axis=1)).tolist() for i,c in centers.items()}
        if not indices[ref]:return None
        drop = float(score(indices))
        require(np.isfinite(drop), "Non-finite region score")
        support = np.concatenate([boxes[i][ids] for i,ids in indices.items() if ids])
        row = {"region_id":f"region_{len(rows)+1:02d}","parent_id":parent,"depth":depth,
               "native_cell_xyz":cell,"native_box_xyz":[support[:,0].min(0).tolist(),support[:,1].max(0).tolist()],
               "series_index":ref,"token_indices":indices[ref],
               "masked_tokens_by_series":{names[i]:ids for i,ids in indices.items()},
               "similarity_drop":drop,"method":VERSION,
               "score_scope":"joint same-grid token support in this spatial cell; not voxel truth or a lesion probability"}
        rows.append(row)
        return row
    coarse = [r for cell in children(extent) if (r:=evaluate(cell,None,0)) is not None]
    parents = sorted([r for r in coarse if r["similarity_drop"]>0],key=lambda r:(-r["similarity_drop"],r["region_id"]))[:2]
    fine = [r for parent in parents for cell in children(parent["native_cell_xyz"])
            if (r:=evaluate(cell,parent["region_id"],1)) is not None]
    positives = [r for r in fine if r["similarity_drop"]>0] or parents
    ranked = sorted(positives,key=lambda r:(-r["similarity_drop"],r["region_id"]))
    return {"version":VERSION,"reference_sequence":reference_sequence,"linked_sequences":[names[i] for i in linked],
            "grid_relation":"same header grid only; anatomical registration/finding identity not independently verified",
            "search_extent_native_xyz":extent,"candidates":rows,"selected":ranked[0] if ranked else None,
            "alternatives":ranked[1:3],"selection_uses_annotations":False,"score_calls":len(rows),
            "status":"proposed_region" if ranked else "no_positive_region"}


def occlusion_scorer(encoder, inputs, text_vector, spatial_dimensions):
    """Reuse frozen PRIMA's exact cosine target; positions/weights are unchanged."""
    from brain_vlm.prima.models import get_study_embeddings
    target = np.asarray(text_vector,dtype=float).copy()
    require(target.ndim==1 and np.isfinite(target).all() and np.linalg.norm(target)>0,"Invalid spatial text target")
    target /= np.linalg.norm(target)
    def similarity(value):
        _,clip = get_study_embeddings(encoder,deepcopy(value))
        return float(clip.numpy()[0]@target)
    baseline = similarity(inputs)
    def score(indices):
        edited = deepcopy(inputs)
        for series,ids in indices.items():
            latent_dim = edited["visual"][series].shape[-1]-spatial_dimensions
            require(latent_dim==256,"Unexpected PRIMA latent layout")
            edited["visual"][series][:,ids,:latent_dim] = 0
        return baseline-similarity(edited)
    return score,baseline


def context(selection, geometry, margin=8):
    """Native box is the scored token support plus an explicit display margin."""
    require(type(margin) is int and margin>=0,"Invalid native margin")
    if selection["selected"] is None:return None
    chosen = selection["selected"]
    low,high = np.asarray(chosen["native_box_xyz"],dtype=int)
    return {"native_box_xyz":[np.maximum(0,low-margin).tolist(),np.minimum(geometry["shape_xyz"],high+margin).tolist()],
            "scored_token_support_native_xyz":chosen["native_box_xyz"],"spatial_cell_native_xyz":chosen["native_cell_xyz"],
            "candidate_id":chosen["region_id"],
            **({"foreground_score":chosen["foreground_score"]} if "foreground_score" in chosen else {"similarity_drop":chosen["similarity_drop"]}),
            "score_scope":chosen["score_scope"],"display_margin_native_voxels":margin,
            "finding_identity":"proposed; not independently matched"}
