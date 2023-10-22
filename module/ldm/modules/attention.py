import math
import torch
from torch import nn

from .diffusionmodules.util import checkpoint

from aitemplate.compiler import ops
from aitemplate.frontend import nn

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype, operations=None):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, specialization="mul", dtype=dtype)
        self.gate = nn.Linear(dim_in, dim_out, specialization="fast_gelu", dtype=dtype)

    def forward(self, x):
        return self.proj(x, self.gate(x))


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim, specialization="fast_gelu", dtype=dtype),
            )
            if not glu
            else GEGLU(dim, inner_dim, dtype=dtype)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x, residual=None):
        shape = ops.size()(x)
        x = self.net(x)
        x = ops.reshape()(x, shape)
        if residual is not None:
            return x + residual
        else:
            return x

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype="float16",
        device=None,
        operations=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype), nn.Dropout(dropout)
        )
        self.attn_op = ops.mem_eff_attention(causal=False)

    def forward(self, x, context=None, value=None, mask=None):
        nheads = self.heads
        d = self.dim_head

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        bs = q.shape()[0]

        q = ops.reshape()(q, [bs, -1, self.heads, self.dim_head])
        k = ops.reshape()(k, [bs, -1, self.heads, self.dim_head])
        v = ops.reshape()(v, [bs, -1, self.heads, self.dim_head])
        q = ops.permute()(q, [0, 2, 1, 3])
        k = ops.permute()(k, [0, 2, 1, 3])
        v = ops.permute()(v, [0, 2, 1, 3])

        out = self.attn_op(
            (ops.reshape()(q, [bs, nheads, -1, d])),
            (ops.reshape()(k, [bs, nheads, -1, d])),
            (ops.reshape()(v, [bs, nheads, -1, d])),
        )
        out = ops.reshape()(out, [bs, -1, nheads * d])
        proj = self.to_out(out)
        proj = ops.reshape()(proj, [bs, -1, nheads * d])
        return proj

def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, dtype=dtype, device=device, operations=operations)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype, device=device, operations=operations)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x, context=None, transformer_options={}):
        return checkpoint(self._forward, (x, context, transformer_options), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = None
        block_index = 0
        if "current_index" in transformer_options:
            extra_options["transformer_index"] = transformer_options["current_index"]
        if "block_index" in transformer_options:
            block_index = transformer_options["block_index"]
            extra_options["block_index"] = block_index
        if "original_shape" in transformer_options:
            extra_options["original_shape"] = transformer_options["original_shape"]
        if "block" in transformer_options:
            block = transformer_options["block"]
            extra_options["block"] = block
        if "cond_or_uncond" in transformer_options:
            extra_options["cond_or_uncond"] = transformer_options["cond_or_uncond"]
        if "patches" in transformer_options:
            transformer_patches = transformer_options["patches"]
        else:
            transformer_patches = {}

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        if "patches_replace" in transformer_options:
            transformer_patches_replace = transformer_options["patches_replace"]
        else:
            transformer_patches_replace = {}

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        n = self.norm2(x)

        context_attn2 = context
        value_attn2 = None
        if "attn2_patch" in transformer_patches:
            patch = transformer_patches["attn2_patch"]
            value_attn2 = context_attn2
            for p in patch:
                n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

        attn2_replace_patch = transformer_patches_replace.get("attn2", {})
        block_attn2 = transformer_block
        if block_attn2 not in attn2_replace_patch:
            block_attn2 = block

        if block_attn2 in attn2_replace_patch:
            if value_attn2 is None:
                value_attn2 = context_attn2
            n = self.attn2.to_q(n)
            context_attn2 = self.attn2.to_k(context_attn2)
            value_attn2 = self.attn2.to_v(value_attn2)
            n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
            n = self.attn2.to_out(n)
        else:
            n = self.attn2(n, context=context_attn2, value=value_attn2)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dtype=None, device=None, operations=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, dtype=dtype, device=device, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim,in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, h, w, c = x.shape()
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)

        x = ops.reshape()(x, [b, -1, c])

        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)

        x = ops.reshape()(x, [b, h, w, c])

        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

