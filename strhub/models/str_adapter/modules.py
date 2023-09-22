# coding=utf-8
import os
import copy
import math
from functools import partial
from collections import OrderedDict
from typing import Optional, Sequence, Callable, Tuple
from timm.models.helpers import named_apply

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.functional import _in_projection_packed
from torch.nn.modules import transformer
from strhub.clip.model import ResidualAttentionBlock, LayerNorm, QuickGELU
from strhub.models.utils import init_weights


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class Hook():
    # A simple hook class that returns the input and output of a layer during forward/backward pass
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class IndentityAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IndentityAdapter, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)


class LinearAdapter(nn.Module):
    """
    CLIP adapter from
    https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
    """
    def __init__(self, c_in, reduction=2, ratio=0.2):
        super(LinearAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ratio = ratio

        # init
        for m in self.modules():
            init_weights(m)

    def forward(self, x):
        x = self.ratio * self.fc(x) + (1.0 - self.ratio) * x

        return x


class AttentionAdapter(nn.Module):
    def __init__(self, c_in, n_head, ratio=0.5):
        super(AttentionAdapter, self).__init__()
        self.attn = ResidualAttentionBlock(d_model=c_in, n_head=n_head)
        self.ratio = ratio
        # init
        named_apply(partial(init_weights, exclude=['none']), self)

    def forward(self, x):
        # x [N, L, D]
        x = x.permute(1, 0, 2)      # NLD -> LND
        x = self.ratio * self.attn(x) + (1.0 - self.ratio) * x
        x = x.permute(1, 0, 2)      # LND -> NLD

        return x


class LadderSideAdapter(nn.Module):
    def __init__(self, clip_model, block_ids=[0, 4, 8, 11], T=0.1) -> None:
        """
        A Ladder Side Adapter
        Args:
            transformer: a transformer object defined by CLIP
            block_ids: ids of blocks in the transformer
            T: temperature
        """
        super(LadderSideAdapter, self).__init__()
        self.T = T
        self.block_ids = block_ids

        self.alpha = nn.Parameter(torch.zeros(len(block_ids)), requires_grad=True)
        resblocks = []
        self.hooks = []
        for i in block_ids:
            self.hooks.append(Hook(clip_model.visual.transformer.resblocks[i], backward=False))
            resblocks.append(copy.deepcopy(clip_model.visual.transformer.resblocks[i]))
        self.resblocks = nn.ModuleList(resblocks)
        self.ln_post = copy.deepcopy(clip_model.visual.ln_post)
        self.proj = copy.deepcopy(clip_model.visual.proj)

    def forward(self, memory=None):
        # x: LND in hooks
        
        # ladder side forward
        W = torch.sigmoid(self.alpha / self.T)
        for i, j in enumerate(self.block_ids):
            if i == 0:
                x = self.resblocks[i](self.hooks[i].input[0])
            else:
                x = W[i] * self.hooks[i].input[0] + (1.0 - W[i]) * x
                x = self.resblocks[i](x)
        x = x.permute(1, 0, 2)                  # LND -> NLD

        # output transformation
        x = self.ln_post(x)
        if self.proj is not None:
            x = torch.matmul(x, self.proj)

        if memory is not None:
            x = W[0] * memory + (1.0 - W[0]) * x

        return x


class LinearLadderSideAdapter(nn.Module):
    def __init__(self, clip_model, block_ids=[1, 3, 5, 7, 9, 11], T=0.1) -> None:
        """
        A Ladder Side Adapter
        Args:
            transformer: a transformer object defined by CLIP
            block_ids: ids of blocks in the transformer
            T: temperature
        """
        super(LinearLadderSideAdapter, self).__init__()
        self.T = T
        self.block_ids = block_ids

        self.alpha = nn.Parameter(torch.zeros(len(block_ids)), requires_grad=True)
        width = clip_model.visual.class_embedding.shape[0]
        resblocks = []
        self.hooks = []
        for i in block_ids:
            self.hooks.append(Hook(clip_model.visual.transformer.resblocks[i], backward=False))
            resblocks.append(LinearAdapter(width, reduction=4, ratio=0.5))
        self.resblocks = nn.ModuleList(resblocks)
        self.ln_post = copy.deepcopy(clip_model.visual.ln_post)
        self.proj = copy.deepcopy(clip_model.visual.proj)
        # init
        named_apply(partial(init_weights, exclude=['ln_post', 'proj', 'alpha']), self)

    def forward(self, memory=None):
        # x: LND in hooks
        
        # ladder side forward
        W = torch.sigmoid(self.alpha / self.T)
        for i, j in enumerate(self.block_ids):
            if i == 0:
                x = self.resblocks[i](self.hooks[i].input[0])
            else:
                x = W[i] * self.hooks[i].input[0] + (1.0 - W[i]) * x
                x = self.resblocks[i](x)
        x = x.permute(1, 0, 2)                  # LND -> NLD

        # output transformation
        x = self.ln_post(x)
        if self.proj is not None:
            x = torch.matmul(x, self.proj)

        if memory is not None:
            x = W[0] * memory + (1.0 - W[0]) * x

        return x


class LadderSideAdapterPruning(nn.Module):
    def __init__(self, clip_model, block_ids=[1, 3, 5, 7, 9, 11], T=0.1, reduction=4.0) -> None:
        """
        A Ladder Side Adapter using the pruned model as the side network
        Args:
            clip_model: clip model
            block_ids: ids of blocks in the transformer
            T: temperature
            reduction: prune the row or columns of weight matrix to (1 / reduction) of the original
        """
        super(LadderSideAdapterPruning, self).__init__()
        self.T = T
        self.block_ids = block_ids
        self.alpha = nn.Parameter(torch.zeros(len(block_ids)), requires_grad=True)
        
        d_model = clip_model.visual.transformer.resblocks[0].attn.embed_dim
        n_head = clip_model.visual.transformer.resblocks[0].attn.num_heads
        resblocks = []
        downsamples = []
        self.hooks = []
        for i in block_ids:
            self.hooks.append(Hook(clip_model.visual.transformer.resblocks[i], backward=False))
    
            # prune and init
            # block = ResidualAttentionBlock(int(d_model / reduction), n_head)
            block = ResidualAttentionBlockCustom(d_model, n_head, reduction=reduction)
            state_dict = clip_model.visual.transformer.resblocks[i].state_dict()
            # new_state_dict = self.prune(state_dict, reduction=reduction)
            new_state_dict = self.prune_v2(state_dict, reduction=reduction)
            block.load_state_dict(new_state_dict)
            resblocks.append(block)
            # downsamples.append(nn.Linear(d_model, int(d_model / reduction), bias=False))
            downsamples.append(nn.Identity())

        self.resblocks = nn.ModuleList(resblocks)
        self.downsamples = nn.ModuleList(downsamples)
        # self.upsample = nn.Linear(int(d_model / reduction), d_model, bias=False)
        self.upsample = nn.Identity()
        self.ln_post = copy.deepcopy(clip_model.visual.ln_post)
        self.proj = copy.deepcopy(clip_model.visual.proj)
        # init
        named_apply(partial(init_weights, exclude=['resblocks', 'ln_post', 'proj', 'alpha']), self)

    def forward(self, memory=None):
        # x: LND in hooks
        
        # ladder side forward
        W = torch.sigmoid(self.alpha / self.T)
        for i, j in enumerate(self.block_ids):
            res_x = self.downsamples[i](self.hooks[i].input[0])
            if i == 0:
                x = self.resblocks[i](res_x)
            else:
                x = W[i] * res_x + (1.0 - W[i]) * x
                x = self.resblocks[i](x)
        x = self.upsample(x)
        x = x.permute(1, 0, 2)                  # LND -> NLD

        # output transformation
        x = self.ln_post(x)
        if self.proj is not None:
            x = torch.matmul(x, self.proj)

        if memory is not None:
            x = W[0] * memory + (1.0 - W[0]) * x

        return x

    def prune(self, state_dcit, reduction):
        new_sd = {}
        for k, v in state_dcit.items():
            if "in_proj" in k:
                v_q, v_k, v_v = torch.chunk(v, 3, dim=0)
                new_v_q = ln_pruning(v_q, reduction=reduction, prune_col=True)
                new_v_k = ln_pruning(v_k, reduction=reduction, prune_col=True)
                new_v_v = ln_pruning(v_v, reduction=reduction, prune_col=True)
                new_v = torch.cat([new_v_q, new_v_k, new_v_v], dim=0).contiguous()
            else:
                new_v = ln_pruning(v, reduction=reduction, prune_col=True)
            new_sd[k] = new_v

        return new_sd

    def prune_v2(self, state_dcit, reduction):
        new_sd = {}
        for k, v in state_dcit.items():
            if "in_proj" in k:
                v_q, v_k, v_v = torch.chunk(v, 3, dim=0)
                new_v_q = ln_pruning(v_q, reduction=reduction, prune_col=False)
                new_v_k = ln_pruning(v_k, reduction=reduction, prune_col=False)
                new_v_v = ln_pruning(v_v, reduction=reduction, prune_col=False)
                new_v = torch.cat([new_v_q, new_v_k, new_v_v], dim=0).contiguous()
            
            elif "out_proj.weight" in k or "mlp.c_proj.weight" in k:
                # only prune in_feature dimension
                new_v = ln_pruning(v.transpose(0, 1), reduction=reduction, prune_col=False)
                new_v = new_v.transpose(0, 1)
            
            elif "mlp.c_fc" in k:
                new_v = ln_pruning(v, reduction=reduction, prune_col=False)

            else:
                new_v = v

            new_sd[k] = new_v

        return new_sd


def ln_pruning(W, reduction=4.0, p=1, prune_col=False):
    if W.ndim == 1:
        n = W.numel()
        ln_norm = torch.abs(W)
        n_to_prune = int(n / reduction)
        _, indices = torch.topk(ln_norm, k=n_to_prune, sorted=False)
        s_indices, _ = torch.sort(indices, descending=False)
        pruned_W = W[s_indices]

    elif W.ndim == 2:
        row, col = W.shape
        # first prune the row
        ln_norm = torch.norm(W, p=p, dim=1)
        n_to_prune = int(row / reduction)
        _, indices = torch.topk(ln_norm, k=n_to_prune, sorted=False)
        s_indices, _ = torch.sort(indices, descending=False)
        pruned_W = W[s_indices, ...]

        if prune_col:
            # then prune the column
            W_ = pruned_W.transpose(0, 1)
            ln_norm = torch.norm(W_, p=p, dim=1)
            n_to_prune = int(col / reduction)
            _, indices = torch.topk(ln_norm, k=n_to_prune, sorted=False)
            s_indices, _ = torch.sort(indices, descending=False)
            pruned_W = W_[s_indices, ...]
            pruned_W = pruned_W.transpose(0, 1)

    else:
        raise NotImplementedError

    return pruned_W


class ResidualAttentionBlockCustom(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, reduction = 4):
        super().__init__()

        self.attn = MultiheadAttentionCustom(d_model, n_head, reduction=reduction)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4 // reduction)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4 // reduction, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiheadAttentionCustom(nn.Module):
    def __init__(self, d_model: int, n_head: int, bias = True, attn_mask: torch.Tensor = None, reduction = 4):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model // reduction, d_model)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model // reduction))
        self.out_proj = nn.Linear(d_model // reduction, d_model, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        # by default, length frsit in input, e.g., [L, N D]
        num_heads = self.num_heads

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # we assume q,k,v are the same
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        head_dim = q.shape[-1] // num_heads

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        
        # update source sequence length after adjustments
        src_len = k.size(1)

        dropout_p = 0.0
        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, head_dim * num_heads)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

# https://github.com/pytorch/pytorch/blob/71f889c7d265b9636b93ede9d651c0a9c4bee191/torch/nn/functional.py#L4809
def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)

    return output, attn


if __name__ == "__main__":
    # W = torch.rand(4, 4)
    # W = torch.rand(4)
    # b = ln_pruning(W, reduction=2.0, p=1)
    # print(W)
    # print(b)

    m = MultiheadAttentionCustom(768, 12, reduction=4)
    m = ResidualAttentionBlockCustom(768, 12, reduction=4)
    num = 0
    for n, p in m.named_parameters():
        num += p.numel()
    print(num)

    x = torch.rand(16, 4, 768)
    # y = m(x, x, x)
    y = m(x)
    print(y[0].shape)
