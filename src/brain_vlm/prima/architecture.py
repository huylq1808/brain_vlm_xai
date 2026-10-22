"""PRIMA inference architecture, adapted from the official MIT-licensed model_parts.py.

State names and FP32 packed attention arithmetic are preserved. Only encoders
and the trained image/text projections remain; no clinical or loss heads.
Upstream revision: 604f1fc4dd7059c278d4d34bbbcaab3e404fbcd4.
"""

import torch
from torch import nn
from typing import Dict, Tuple, Optional, Union
from einops import rearrange, repeat


class PreNorm(nn.Module):
    """Pre-normalization wrapper for transformer blocks."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(
        self,
        x: torch.Tensor,
        culen: Optional[torch.Tensor] = None,
        mxlen: Optional[torch.Tensor] = None,
    ):
        if culen is None:
            return self.fn(self.norm(x))
        return self.fn(self.norm(x), culen, mxlen)


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Packed multi-head attention using PRIMA's FP32 reference arithmetic."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** (-0.5)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropoutp = dropout
        self.causal = causal
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, culen: torch.Tensor, mxlen: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.to_qkv(x).view(len(x), 3, self.heads, self.dim_head)
        out = no_flash_attn_varlen_substitute(qkv, culen.to(torch.int32))
        return self.to_out(out.flatten(start_dim=1))


def no_flash_attn_varlen_substitute(
    qkv: torch.Tensor, culen: torch.Tensor
) -> torch.Tensor:
    """Fallback attention implementation when flash attention is not available."""
    qkv = qkv.transpose(0, 1)
    q, k, v = map(lambda t: rearrange(t, "n h d -> h n d"), qkv)
    n = qkv.size()[1]
    h = qkv.size()[2]
    d = qkv.size()[-1]
    out = torch.zeros(h, n, d).to(qkv.device)
    for i in range(len(culen) - 1):
        dots = torch.matmul(
            q[:, culen[i] : culen[i + 1]],
            k[:, culen[i] : culen[i + 1]].transpose(-1, -2),
        ) * d ** (-0.5)
        attn = torch.nn.functional.softmax(dots, dim=-1)
        out[:, culen[i] : culen[i + 1]] = torch.matmul(
            attn, v[:, culen[i] : culen[i + 1]]
        )
    out = rearrange(out, "h n d -> n (h d)")
    return out


class Transformer(nn.Module):
    """Transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                causal=causal,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(
        self, x: torch.Tensor, culen: torch.Tensor, mxlen: torch.Tensor
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, culen, mxlen) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """Vision Transformer for processing image sequences."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        clsnum: int = 10,
    ):
        """
        dim: dimension of embeddings in this transformer
        num_classes: final MLP head output dimension
        depth: depth of transforemr
        heads: attention heads of transformer
        mlp_dim: intermediate dimension of transformer MLP
        pool: either 'cls' (meaning classification token) or 'mean'
        dim_head: dimension of each attention head
        dropout: dropout rate
        emb_dropout: embedding dropout rate
        clsnum: how many classification tokens
        """
        super().__init__()
        self.dim = dim
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pos_embedding = None
        self.clsnum = clsnum
        self.cls_token = nn.Parameter(torch.randn(1, self.clsnum, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.layernorm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.Linear(self.clsnum * dim, num_classes))

    def make_no_flashattn(self) -> None:
        for layer in self.transformer.layers:
            layer[0].fn.noflashattn = True

    def forward(
        self,
        xdict: Dict[str, torch.Tensor],
        retpool: bool = False,
        retboth: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(self, "dim"):
            self.dim = self.layernorm.normalized_shape[0]
        x = xdict["visual"]
        lens = xdict["lens"]
        b, n, embsize = x.shape
        cls_tokens = repeat(self.cls_token, "1 c d -> b c d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embedding is not None:
            x += self.pos_embedding[:, : n + 1]
        x = self.dropout(x)
        mxlen = n + self.clsnum
        cl = torch.cumsum(lens + self.clsnum, dim=0)
        culen = torch.zeros(len(cl) + 1).long().to(cl.device)
        culen[1:] = cl
        nx = torch.zeros(culen[-1].item(), embsize).to(x.device)
        for i in range(len(cl)):
            assert lens[i] + self.clsnum == culen[i + 1] - culen[i]
            nx[culen[i] : culen[i + 1]] = x[i][0 : lens[i] + self.clsnum]
        x = nx
        zerocheck = torch.logical_and(
            x.max(dim=1).values == 0, x.min(dim=1).values == 0
        )
        assert zerocheck.int().sum() == 0
        xx = self.transformer(x, culen, mxlen)
        xxoutdim = xx.size()[-1]
        x = torch.zeros(b, self.clsnum, xxoutdim).to(x.device)
        for i in range(b):
            x[i] = xx[culen[i] : culen[i] + self.clsnum]
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0 : self.clsnum]
        x = self.to_latent(x)
        x = self.layernorm(x).view(b, -1)
        if retboth:
            return (x, self.mlp_head(x))
        if retpool:
            return x
        return self.mlp_head(x)


class SerieEncoder(nn.Module):
    """LSTM-based encoder for series names."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(47, 200)
        self.lstm = nn.LSTM(200, 200, batch_first=True)
        self.linear = nn.Linear(200, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x, _ = self.lstm(x)
        ret = self.linear(x[:, -1, :])
        return ret

    def make_no_flashattn(self) -> None:
        pass


class SerieTransformerEncoder(nn.Module):
    """Transformer-based encoder for series names."""

    def __init__(self, spec):
        super().__init__()
        self.embsize = spec["transformer"]["dim"]
        self.embed = nn.Embedding(spec["vocab_size"], spec["embedding_dim"])
        self.transformer = Transformer(**spec["transformer"])
        self.prelinear = spec["prelinear"]
        self.linear = (
            nn.Identity()
            if self.prelinear
            else nn.Linear(self.embsize, spec["out_dim"])
        )
        self.p_enc = torch.empty(spec["position_shape"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        occur46 = (x == 46).nonzero()
        assert occur46.size()[0] == len(x)
        assert occur46.size()[1] == 2
        assert torch.all(occur46[:, 0] == torch.arange(end=len(x)).to(occur46.device))
        lens = occur46[:, 1] + 1
        x = self.embed(x)
        posenc = self.p_enc[0 : x.size()[1]].to(x.device).repeat(len(x), 1, 1)
        x = torch.cat([x, posenc], dim=2)
        cl = torch.cumsum(lens, dim=0)
        culen = torch.zeros(len(cl) + 1).long().to(cl.device)
        culen[1:] = cl
        mxlen = lens.max()
        nx = torch.zeros(culen[-1].item(), self.embsize).to(x.device)
        for i in range(len(cl)):
            assert lens[i] == culen[i + 1] - culen[i]
            nx[culen[i] : culen[i + 1]] = x[i][0 : lens[i]]
        nx = self.transformer(nx, culen, mxlen)
        xout = torch.stack([nx[culen[i + 1] - 1] for i in range(len(lens))])
        if hasattr(self, "prelinear") and self.prelinear:
            return xout
        return self.linear(xout)

    def make_no_flashattn(self) -> None:
        for layer in self.transformer.layers:
            layer[0].fn.noflashattn = True


class HierViT(nn.Module):
    """Frozen hierarchical image encoder. Clinical and training heads are absent."""

    def __init__(self, spec):
        super().__init__()
        self.innerViT = ViT(**spec["inner"])
        self.outerViT = ViT(**spec["outer"])
        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)
        self.useseriename = spec["useseriename"]
        self.usestudydescription = spec["usestudydescription"]
        if self.useseriename:
            self.serieencoder = build_name_encoder(spec["serieencoder"])
        if self.usestudydescription:
            self.studyencoder = build_name_encoder(spec["studyencoder"])

    def forward(
        self, xdict: Dict[str, torch.Tensor], retpool: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        mydevice = self.dummy_param.device
        lenss = xdict["lenss"]
        lens = xdict["lens"].clone()
        x = xdict["visual"]
        lenss = lenss.transpose(0, 1)
        extra = 0
        if self.useseriename:
            extra += 1
        mapper = {}
        counter = 0
        slens = []
        if not hasattr(self.innerViT, "dim"):
            self.innerViT.dim = 289
        serienameencoded = torch.zeros(len(lens), len(lenss), self.innerViT.dim).to(
            self.dummy_param.device
        )
        if self.useseriename:
            for j in range(len(lens)):
                serienamevecs = xdict["serienames"][j][0 : lens[j]]
                serienameencoded[j][0 : lens[j]] = self.serieencoder(serienamevecs)
        totalimgs = lens.sum()
        imgmax = lenss.max()
        newx = torch.zeros(totalimgs, imgmax + extra, self.innerViT.dim).to(mydevice)
        for i in range(len(x)):
            im = x[i]
            for j in range(len(im)):
                if lenss[i][j] > 0:
                    mapper[counter] = (i, j)
                    slen = lenss[i][j]
                    if self.useseriename:
                        newx[counter][0] = serienameencoded[j][i]
                    theserie = im[j][0:slen]
                    theserielastdim = theserie.shape[-1]
                    newx[counter][extra : slen + extra, 0:theserielastdim] = theserie
                    counter += 1
                    slens.append(slen + extra)
        slens = torch.LongTensor(slens).to(mydevice)
        outs = self.innerViT({"visual": newx, "lens": slens})
        extra = 0
        if self.usestudydescription:
            extra = 1
        nextx = torch.zeros(len(lens), lens.max() + extra, outs.size()[-1]).to(mydevice)
        if self.usestudydescription:
            nextx[:, 0] = self.studyencoder(
                xdict["studydescription"].long().to(mydevice)
            )
        for pos, out in enumerate(outs):
            i, j = mapper[pos]
            nextx[j][i + extra] = out
        lens += extra
        return self.outerViT(
            {"visual": nextx, "lens": lens.to(mydevice)}, retpool=retpool
        )

    def make_no_flashattn(self) -> None:
        self.innerViT.make_no_flashattn()
        self.outerViT.make_no_flashattn()
        for module in self.modules():
            if isinstance(module, Attention):
                module.noflashattn = True


def build_name_encoder(spec):
    """Rebuild the actual name encoder, including a pretrained sequential adapter."""
    if spec["kind"] == "transformer":
        return SerieTransformerEncoder(spec)
    if spec["kind"] == "sequential":
        return nn.Sequential(
            build_name_encoder(spec["base"]), nn.Linear(spec["in_dim"], spec["out_dim"])
        )
    if spec["kind"] == "lstm":
        return SerieEncoder(spec["out_dim"])
    raise ValueError(f"Unsupported name encoder: {spec['kind']}")


class TextEncoder(nn.Module):
    """Trained GPT-2 family backbone and PRIMA text projection, without downloads."""

    def __init__(self, config, feature_dim):
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        cfg = GPT2Config.from_dict(config)
        cfg._attn_implementation = "eager"
        cfg.use_cache = False
        self.model = GPT2Model(cfg)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.n_embd),
            nn.Linear(cfg.n_embd, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, tokens, lengths):
        hidden = self.model(tokens, use_cache=False, return_dict=True).last_hidden_state
        pooled = hidden[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.mlp_head(pooled)
