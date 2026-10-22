"""Image tokenization and input preparation without full PRIMA model imports."""

from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from brain_vlm.prima import patches
from brain_vlm.prima.dataset import MrVoxelDataset


class EncoderPipeline:
    """Small adapter compatible with brain_vlm.prima.features and attribution.

    Models are injected explicitly. This object never downloads or loads a model.
    Image reading/resampling stays in brain_vlm.prima.data, as in earlier runs.
    """

    def __init__(
        self, patchifier, device="cuda", chunk_size=16, study_description="MRI BRAIN"
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.patchifier = patchifier
        self.tokenizer_model = None
        self.vol_utils = patches
        self.filtercoords = patches.filtercoords
        self.config = SimpleNamespace(
            device=device,
            max_tokens_per_chunk=chunk_size,
            study_description=study_description,
        )

    def run_tokenizer_model(self, mri_study, series_names):
        """Encode every raw patch, bounded by chunk_size; return CPU latent tensors."""
        if self.tokenizer_model is None:
            raise RuntimeError("Inject a loaded VQ encoder before tokenization")
        if len(mri_study) != len(series_names):
            raise ValueError("Volumes and sequence names must have the same length")
        loader = DataLoader(
            MrVoxelDataset(mri_study), batch_size=1, shuffle=False, num_workers=0
        )
        embeddings, metadata = [], []
        try:
            with torch.inference_mode():
                for batch, meta in loader:
                    raw = batch[0]
                    if not 0 < len(raw) <= 5000:
                        raise ValueError(f"Expected 1..5000 patches, got {len(raw)}")
                    chunks = []
                    for start in range(0, len(raw), self.config.max_tokens_per_chunk):
                        x = raw[start : start + self.config.max_tokens_per_chunk]
                        chunks.append(
                            self.tokenizer_model.encode(
                                x.unsqueeze(1).to(self.config.device)
                            )
                            .detach()
                            .cpu()
                        )
                    embeddings.append(torch.cat(chunks))
                    metadata.append(meta)
        finally:
            self.tokenizer_model = None
        return embeddings, list(series_names), metadata

    def prepare_prima_input(
        self, series_embeddings, series_names, all_ser_emb_meta, otsu_percentage=5
    ):
        """Filter tokens, append saved spatial features and attach acquisition names."""
        if not (
            len(series_embeddings) == len(series_names) == len(all_ser_emb_meta) > 0
        ):
            raise ValueError("Provide tokens, names and metadata for each sequence")
        kept, coords = [], []
        for latents, name, meta in zip(
            series_embeddings, series_names, all_ser_emb_meta
        ):
            for percentage in range(otsu_percentage, -1, -1):
                selected, locations, _ = patches.filtercoords(meta, percentage, latents)
                if len(locations) > 25:
                    break
            print(
                f"Using otsu percentage: {percentage} for series: {name}; "
                f"{len(latents)} -> {len(selected)} tokens"
            )
            kept.append(selected)
            coords.append(locations)
        lenss = torch.tensor([[len(x) for x in kept]], dtype=torch.long)
        max_len = int(lenss.max())
        visuals = []
        for x in self.patchifier(kept, coords):
            padding = torch.zeros(max_len - len(x), x.shape[1], dtype=x.dtype)
            visuals.append(torch.cat([x, padding]).unsqueeze(0))
        names = [patches.chartovec(name) for name in series_names]
        name_tensor = torch.zeros(1, len(names), max(map(len, names)), dtype=torch.long)
        for index, name in enumerate(names):
            name_tensor[0, index, : len(name)] = name
        return {
            "visual": visuals,
            "lens": torch.tensor([len(kept)]),
            "lenss": lenss,
            "serienames": name_tensor,
            "studydescription": patches.chartovec(
                self.config.study_description
            ).unsqueeze(0),
            "hash": ["study_0"],
        }
