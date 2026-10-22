"""PRIMA patch shape selection, independent of neural-network dependencies."""


def adjusted_patch_shape(image_shape, patch_shape=None, z_val=4):
    """Use thin patches along the first axis whose prepared length is not 256.

    This preserves official VolUtils.adjusted_patch_shape behavior. Geometry
    preparation can use it without importing MONAI's network/transforms package.
    """
    if patch_shape is None:
        patch_shape = [32, 32, 32]
    z_idx = None
    for index, size in enumerate(image_shape):
        if size != 256:
            z_idx = index
            patch_shape[index] = z_val
            break
    return patch_shape, z_idx
