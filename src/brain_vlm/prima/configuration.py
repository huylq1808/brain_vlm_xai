"""Describe the loaded architecture rather than guessing from sample YAML files.

Called by the exporter. Unsupported architectures fail before an export is
published; runtime reconstruction uses the saved descriptions only.
"""

from torch import nn


def transformer_spec(module):
    layers = []
    for attention, feedforward in module.layers:
        a, f = attention.fn, feedforward.fn
        if a.causal:
            raise ValueError("Causal visual/name attention is not supported")
        for norm in (attention.norm, feedforward.norm):
            if norm.eps != 1e-5:
                raise ValueError("Non-default visual LayerNorm epsilon")
        layers.append(
            {
                "dim": a.to_qkv.in_features,
                "heads": a.heads,
                "dim_head": a.dim_head,
                "mlp_dim": f.net[0].out_features,
                "dropout": a.dropoutp,
            }
        )
    if not layers or any(item != layers[0] for item in layers):
        raise ValueError("Expected a non-empty homogeneous PRIMA transformer")
    return dict(layers[0], depth=len(layers))


def vit_spec(module):
    if module.layernorm.eps != 1e-5 or not isinstance(module.to_latent, nn.Identity):
        raise ValueError("Unsupported ViT pooling/norm")
    if len(module.mlp_head) != 1 or not isinstance(module.mlp_head[0], nn.Linear):
        raise ValueError("Expected the official single Linear visual projection")
    return dict(
        transformer_spec(module.transformer),
        num_classes=module.mlp_head[0].out_features,
        clsnum=module.cls_token.shape[1],
        pool=module.pool,
        emb_dropout=module.dropout.p,
    )


def name_spec(module):
    if isinstance(module, nn.Sequential):
        if len(module) != 2 or not isinstance(module[1], nn.Linear):
            raise ValueError("Unsupported pretrained name adapter")
        return {
            "kind": "sequential",
            "base": name_spec(module[0]),
            "in_dim": module[1].in_features,
            "out_dim": module[1].out_features,
        }
    if type(module).__name__ == "SerieTransformerEncoder":
        return {
            "kind": "transformer",
            "transformer": transformer_spec(module.transformer),
            "vocab_size": module.embed.num_embeddings,
            "embedding_dim": module.embed.embedding_dim,
            "out_dim": module.linear.out_features,
            "prelinear": bool(getattr(module, "prelinear", False)),
            "position_shape": list(module.p_enc.shape),
        }
    if type(module).__name__ == "SerieEncoder":
        if module.embed.weight.shape != (47, 200) or module.lstm.num_layers != 1:
            raise ValueError("Unsupported name LSTM")
        return {"kind": "lstm", "out_dim": module.linear.out_features}
    raise ValueError(f"Unsupported name encoder: {type(module).__name__}")


def describe_visual(visual):
    spec = {
        "inner": vit_spec(visual.innerViT),
        "outer": vit_spec(visual.outerViT),
        "useseriename": visual.useseriename,
        "usestudydescription": visual.usestudydescription,
    }
    if spec["useseriename"]:
        spec["serieencoder"] = name_spec(visual.serieencoder)
    if spec["usestudydescription"]:
        spec["studyencoder"] = name_spec(visual.studyencoder)
    return spec


def describe_text(text):
    from transformers import GPT2Model

    if not isinstance(text.model, GPT2Model):
        raise ValueError(
            "This exporter supports the public PRIMA GPT-2 family text branch"
        )
    head = text.mlp_head
    if len(head) != 3 or not isinstance(head[1], nn.Linear):
        raise ValueError("Unsupported text projection")
    if any(head[i].eps != 1e-5 for i in (0, 2)):
        raise ValueError("Non-default text projection LayerNorm epsilon")
    return {"gpt2": text.model.config.to_dict(), "feature_dim": head[1].out_features}


def describe_vq_encoder(config):
    p = config["vqvae_config"]
    return {
        "spatial_dims": p["spatial_dims"],
        "in_channels": p["in_channels"],
        "out_channels": p["embedding_dim"],
        "num_channels": p["num_channels"],
        "num_res_layers": p["num_res_layers"],
        "num_res_channels": p["num_res_channels"],
        "downsample_parameters": p["downsample_parameters"],
        "dropout": 0.0,
        "act": "RELU",
    }
