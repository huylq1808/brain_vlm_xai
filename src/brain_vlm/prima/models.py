"""Prepared PRIMA tensors and trained image/text projections; compact loaders live in loading.py."""
import numpy as np
import torch
import torch.nn.functional as F
from brain_vlm.prima.artifacts import validate_embedding

def prepare_inputs(pipeline, token_record, device):
    # Build fresh tensors each call: upstream HierViT can modify inputs['lens'].
    """Build fresh model inputs; upstream HierViT may mutate its length tensors."""
    inputs = pipeline.prepare_prima_input(
        series_embeddings=token_record["series_tokens"],
        series_names=token_record["series_names"],
        all_ser_emb_meta=token_record["token_metadata"],
    )
    inputs["visual"] = [t.to(device) for t in inputs["visual"]]
    for key in ["lens", "lenss", "serienames", "studydescription"]:
        inputs[key] = inputs[key].to(device)
    return inputs

def project_clip_embedding(encoder, pooled):
    """Apply the trained visual projection and L2 normalization without report input."""
    head = encoder.outerViT.mlp_head
    parameter = next(head.parameters())
    h = torch.as_tensor(pooled, dtype=parameter.dtype, device=parameter.device)
    with torch.inference_mode():
        z = F.normalize(head(h).float(), dim=-1)
    validate_embedding(z, head[0].out_features, unit_norm=True)
    return z.cpu()

def get_study_embeddings(encoder, inputs):
    """Return CPU float32 pooled h and normalized CLIP z from one study forward."""
    with torch.inference_mode():
        h = encoder(inputs, retpool=True)
    validate_embedding(h, encoder.outerViT.mlp_head[0].in_features)
    z = project_clip_embedding(encoder, h)
    return h.float().cpu(), z

def restore_text_runtime_metadata(text_encoder):
    """Restore GPT-2 runtime fields omitted by older full-object checkpoints.

    This is for the Transformers 4.x inference environment used by the notebooks;
    it neither replaces the trained backbone/projection nor changes their weights.
    """
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model

    model = text_encoder.model
    if not isinstance(model, GPT2Model):
        return {}
    model.config._attn_implementation = "eager"
    model._attn_implementation = "eager"
    model.config.use_cache = False  # Full-text embedding, not autoregressive decoding.
    for layer_index, block in enumerate(model.h):
        for attention in (block.attn, getattr(block, "crossattention", None)):
            if not isinstance(attention, GPT2Attention):
                continue
            # These constructor fields are used by newer GPT-2 forward methods.
            if not hasattr(attention, "config"):
                attention.config = model.config
            attention.config._attn_implementation = "eager"
            if not hasattr(attention, "head_dim"):
                attention.head_dim = attention.embed_dim // attention.num_heads
            if not hasattr(attention, "is_causal"):
                attention.is_causal = True
            if not hasattr(attention, "layer_idx"):
                attention.layer_idx = layer_index
    return {"text_runtime_compatibility": "legacy_gpt2_eager_v1"}

def encode_texts(text_encoder, tokenizer, texts, max_length):
    # Also repairs an encoder already loaded before a helper refresh in Colab.
    """Encode one text at a time with EOS, last-token pooling and the trained text head."""
    if not texts or max_length < 1:
        raise ValueError(
            "Provide at least one text and a positive maximum token length"
        )
    restore_text_runtime_metadata(text_encoder)
    parameter = next(text_encoder.parameters())
    vectors, truncated = [], []
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer needs the checkpoint EOS token")
    vocab_size = text_encoder.model.get_input_embeddings().num_embeddings
    for text in texts:
        if not text.strip():
            raise ValueError("Empty text")
        ids = tokenizer(text.strip() + "<|endoftext|>")["input_ids"]
        was_truncated = len(ids) > max_length
        if was_truncated:
            ids = ids[:max_length]
            ids[-1] = eos
        if not ids or max(ids) >= vocab_size:
            raise ValueError("Tokenizer IDs incompatible with checkpoint vocabulary")
        tokens = torch.tensor([ids], device=parameter.device)
        with torch.inference_mode():
            # Same last-token pooling and trained head as upstream GPTWrapper.
            # Its zeros() pooling buffer defaults to float32, incompatible with
            # the FP16 head used here on Colab. Indexing preserves hidden dtype.
            hidden = text_encoder.model(
                tokens, use_cache=False, return_dict=True
            ).last_hidden_state
            pooled = hidden[:, len(ids) - 1, :]
            head_parameter = next(text_encoder.mlp_head.parameters())
            vector = F.normalize(
                text_encoder.mlp_head(pooled.to(head_parameter.dtype)).float(), dim=-1
            )
        if not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("Invalid text embedding")
        vectors.append(vector.cpu().numpy()[0])
        truncated.append(was_truncated)
    return np.stack(vectors), truncated
