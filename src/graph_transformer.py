# -*- coding: utf-8 -*-
"""
Graph Transformer (node-only) for topic refinement graphs.

Designed to be called from train_gt.py as:
    Z_out = gt(Z_pair, edges, slice_ids=slice_ids)

Where:
- Z_pair: (N, d) topic vectors (e.g., N=2K for two adjacent slices)
- edges:  list of (src:int, dst:int, weight:float) or (src, dst)
         (your train_gt.py builds edges as (a, j, w))
- slice_ids: (N,) LongTensor with absolute slice indices (for RTPE bias)

Implements:
- Sparse adjacency-masked multi-head attention over provided edges
- Optional edge weight modulation (multiplying logits by weight)
- Relative temporal position bias (RTPE) added to logits based on slice_ids difference
- Residual + (BatchNorm or LayerNorm) + FFN + Residual + Norm
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


EdgeTuple = Union[Tuple[int, int], Tuple[int, int, float]]


def _ensure_edge_tensors(
    edges: Sequence[EdgeTuple],
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.Tensor]]:
    """
    Convert python edges list to tensors:
      src: (E,)
      dst: (E,)
      w  : (E,) float or None
    """
    if len(edges) == 0:
        raise ValueError("edges is empty; Graph Transformer needs at least one edge.")

    # edges could be (src, dst) or (src, dst, w)
    if len(edges[0]) == 2:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
        w = None
    elif len(edges[0]) == 3:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
        w = torch.tensor([float(e[2]) for e in edges], dtype=torch.float32, device=device)
    else:
        raise ValueError("Each edge must be a tuple of (src,dst) or (src,dst,weight).")

    return src, dst, w


def _scatter_reduce_max(src_vals: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Compute per-index max for 1D or 2D tensors using torch.scatter_reduce if available.
    src_vals: (E,) or (E,H)
    index:    (E,)
    returns:  (dim_size,) or (dim_size,H)
    """
    if hasattr(torch.Tensor, "scatter_reduce_"):
        # torch >= 1.12ish
        if src_vals.dim() == 1:
            out = torch.full((dim_size,), -torch.inf, device=src_vals.device, dtype=src_vals.dtype)
            out.scatter_reduce_(0, index, src_vals, reduce="amax", include_self=True)
            return out
        elif src_vals.dim() == 2:
            H = src_vals.size(1)
            out = torch.full((dim_size, H), -torch.inf, device=src_vals.device, dtype=src_vals.dtype)
            out.scatter_reduce_(0, index.view(-1, 1).expand(-1, H), src_vals, reduce="amax", include_self=True)
            return out
        else:
            raise ValueError("src_vals must be 1D or 2D.")
    else:
        # Fallback: loop (slower, but correct)
        if src_vals.dim() == 1:
            out = torch.full((dim_size,), -torch.inf, device=src_vals.device, dtype=src_vals.dtype)
            for i in range(src_vals.size(0)):
                out[index[i]] = torch.maximum(out[index[i]], src_vals[i])
            return out
        elif src_vals.dim() == 2:
            H = src_vals.size(1)
            out = torch.full((dim_size, H), -torch.inf, device=src_vals.device, dtype=src_vals.dtype)
            for i in range(src_vals.size(0)):
                out[index[i]] = torch.maximum(out[index[i]], src_vals[i])
            return out
        else:
            raise ValueError("src_vals must be 1D or 2D.")


def _scatter_reduce_sum(src_vals: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Compute per-index sum for 1D or 2D tensors.
    """
    if hasattr(torch.Tensor, "scatter_add_"):
        if src_vals.dim() == 1:
            out = torch.zeros((dim_size,), device=src_vals.device, dtype=src_vals.dtype)
            out.scatter_add_(0, index, src_vals)
            return out
        elif src_vals.dim() == 2:
            H = src_vals.size(1)
            out = torch.zeros((dim_size, H), device=src_vals.device, dtype=src_vals.dtype)
            out.scatter_add_(0, index.view(-1, 1).expand(-1, H), src_vals)
            return out
        else:
            raise ValueError("src_vals must be 1D or 2D.")
    else:
        # Extremely old fallback
        if src_vals.dim() == 1:
            out = torch.zeros((dim_size,), device=src_vals.device, dtype=src_vals.dtype)
            for i in range(src_vals.size(0)):
                out[index[i]] += src_vals[i]
            return out
        elif src_vals.dim() == 2:
            H = src_vals.size(1)
            out = torch.zeros((dim_size, H), device=src_vals.device, dtype=src_vals.dtype)
            for i in range(src_vals.size(0)):
                out[index[i]] += src_vals[i]
            return out
        else:
            raise ValueError("src_vals must be 1D or 2D.")


def segment_softmax(logits_eh: torch.Tensor, dst: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax over incoming edges per destination node, independently per head.

    logits_eh: (E, H)
    dst:       (E,)
    returns:   (E, H) attention weights
    """
    # subtract max per dst for stability
    max_per_dst = _scatter_reduce_max(logits_eh, dst, num_nodes)          # (N,H)
    stabilized = logits_eh - max_per_dst[dst]                             # (E,H)
    exp = stabilized.exp()
    denom = _scatter_reduce_sum(exp, dst, num_nodes).clamp_min(1e-12)     # (N,H)
    return exp / denom[dst]

class GlobalMemoryMHA(nn.Module):
    """
    Global memory attention stream.

    Each node attends to TWO memory tokens:
      (1) slice-specific memory token: mem_slice[slice_id]
      (2) shared global memory token: mem_global (one token)

    This makes attention non-trivial (softmax over 2 tokens).
    Returns a node-wise global message (N,d).
    """

    def __init__(self, d: int, num_heads: int, num_slices: int, dropout: float = 0.0):
        super().__init__()
        if d % num_heads != 0:
            raise ValueError(f"d={d} must be divisible by num_heads={num_heads}")
        self.d = d
        self.num_heads = num_heads
        self.dk = d // num_heads
        self.scaling = self.dk ** -0.5
        self.dropout = dropout

        # memory params
        self.mem_slice = nn.Embedding(num_slices, d)
        self.mem_global = nn.Parameter(torch.zeros(1, d))

        # projections
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

        nn.init.normal_(self.mem_slice.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mem_global, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor, slice_ids: torch.LongTensor) -> torch.Tensor:
        """
        h: (N,d)
        slice_ids: (N,)  each node's slice index (must be < num_slices)
        returns: (N,d)
        """
        N = h.size(0)
        H = self.num_heads
        dk = self.dk

        # Build 2 memory tokens per node: [mem_slice, mem_global]
        ms = self.mem_slice(slice_ids)                         # (N,d)
        mg = self.mem_global.expand(N, -1)                     # (N,d)
        mem = torch.stack([ms, mg], dim=1)                     # (N,2,d)

        # Project queries from nodes
        q = self.q_proj(h).view(N, H, dk) * self.scaling       # (N,H,dk)

        # Project keys/values from memory tokens
        k = self.k_proj(mem).view(N, 2, H, dk)                 # (N,2,H,dk)
        v = self.v_proj(mem).view(N, 2, H, dk)                 # (N,2,H,dk)

        # Attention logits over the 2 memory tokens
        # logits: (N,H,2)
        logits = (q.unsqueeze(1) * k).sum(dim=-1).transpose(1, 2)  # (N,H,2)

        # softmax over the 2 memory tokens
        attn = torch.softmax(logits, dim=-1)                   # (N,H,2)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # weighted sum of memory values -> (N,H,dk)
        out = (attn.unsqueeze(-1) * v.transpose(1, 2)).sum(dim=2)  # (N,H,dk)

        out = out.reshape(N, self.d)                           # (N,d)
        out = self.out_proj(out)                               # (N,d)
        return out



class SparseMHA_Competitive(nn.Module):
    """
    Sparse competitive multi-head attention.
    Removes averaging behavior by enforcing winner-take-most dynamics.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        dropout: float = 0.0,
        rtpe_clip: int = 8,
        temperature: float = 0.5,   # NEW: sharp attention
    ):
        super().__init__()
        assert d % num_heads == 0
        self.d = d
        self.num_heads = num_heads
        self.dk = d // num_heads
        self.scale = self.dk ** -0.5
        self.dropout = dropout
        self.rtpe_clip = rtpe_clip
        self.temperature = temperature

        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

        self.rtpe = nn.Embedding(2 * rtpe_clip + 1, num_heads)

    #def forward(self, h, src, dst, edge_w=None, slice_ids=None):
    def forward(self, h, src, dst, edge_w=None, slice_ids=None, node_gate=None):
        N = h.size(0)
        H = self.num_heads
        dk = self.dk

        q = self.q(h).view(N, H, dk)
        k = self.k(h).view(N, H, dk)
        v = self.v(h).view(N, H, dk)

        logits = (q[dst] * k[src]).sum(-1) * self.scale   # (E,H)

        # RTPE bias (soft, not enforcing)
        if slice_ids is not None:
            rel = (slice_ids[dst] - slice_ids[src]).clamp(-self.rtpe_clip, self.rtpe_clip)
            logits = logits + self.rtpe(rel + self.rtpe_clip)

        if edge_w is not None:
            logits = logits * edge_w.unsqueeze(-1)
        if node_gate is not None:
            eps = 1e-8
            gs = node_gate[src].clamp_min(eps)   # (E,)
            gd = node_gate[dst].clamp_min(eps)   # (E,)
            logits = logits + (gs.log().unsqueeze(-1) + gd.log().unsqueeze(-1))

        # 🔥 COMPETITION: sharp softmax
        logits = logits / self.temperature

        # Winner-take-most: keep top-1 per dst per head
        max_logits = _scatter_reduce_max(logits, dst, N)   # (N,H)
        mask = logits >= (max_logits[dst] - 1e-6)
        logits = logits.masked_fill(~mask, -1e9)

        attn = segment_softmax(logits, dst, N)
        attn = F.dropout(attn, self.dropout, self.training)

        msg = v[src] * attn.unsqueeze(-1)

        # If node is inactive, it should not send/receive messages
        if node_gate is not None:
            msg = msg * (node_gate[src] * node_gate[dst]).unsqueeze(-1).unsqueeze(-1)
        out = torch.zeros((N, H, dk), device=h.device)
        out.scatter_add_(0, dst.view(-1,1,1).expand(-1,H,dk), msg)

        return self.out(out.reshape(N, self.d))


class GTLayer(nn.Module):
    """
    Anti-smoothing Graph Transformer Layer.
    """

    def __init__(
        self,
        d,
        num_heads,
        dropout,
        rtpe_clip=8,
        use_global_memory=False,
        num_slices=1000,
    ):
        super().__init__()

        self.mha = SparseMHA_Competitive(
            d=d,
            num_heads=num_heads,
            dropout=dropout,
            rtpe_clip=rtpe_clip,
            temperature=0.4,
        )

        # Evolution gate: how much to update this topic
        self.update_gate = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.Sigmoid()
        )

        # NO BatchNorm → LayerNorm only
        self.norm = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.Linear(d, 2*d),
            nn.GELU(),
            nn.Linear(2*d, d),
        )

    #def forward(self, h, src, dst, edge_w=None, slice_ids=None):
    def forward(self, h, src, dst, edge_w=None, slice_ids=None, node_gate=None):
        msg = self.mha(h, src, dst, edge_w=edge_w, slice_ids=slice_ids, node_gate=node_gate)


        g = self.update_gate(h)        # (N,1)

        # inactive topics should not evolve (or evolve minimally)
        if node_gate is not None:
            g = g * node_gate.view(-1, 1)

        h = h + g * msg   
        h = self.norm(h)

        h = h + self.ffn(h)
        h = self.norm(h)

        return h

class TopicGraphTransformer(nn.Module):
    def __init__(
        self,
        d: int,
        num_heads: int = 8,
        num_layers: int = 2,
        pos_enc_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        rtpe_clip: int = 8,
        refine_alpha: float = 0.001,

        # NEW: global memory controls
        use_global_memory: bool = True,
        num_slices: int = 1000,
        global_dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.refine_alpha = float(refine_alpha)

        self.pos_enc_dim = pos_enc_dim
        self.pos_proj = nn.Linear(pos_enc_dim, d) if pos_enc_dim is not None else None
        self.layers = nn.ModuleList([
            GTLayer(
                d=d,
                num_heads=num_heads,
                dropout=dropout,
                rtpe_clip=rtpe_clip,
                use_global_memory=use_global_memory,
                num_slices=num_slices,
            )
            for _ in range(num_layers)
        ])
    def forward(
        self,
        x: torch.Tensor,
        edges: Sequence[EdgeTuple],
        slice_ids: Optional[torch.LongTensor] = None,
        pos_enc: Optional[torch.Tensor] = None,
        node_gate: Optional[torch.Tensor] = None,   # (N,)
    ) -> torch.Tensor:
        device = x.device
        src, dst, w = _ensure_edge_tensors(edges, device=device)

        h0 = x
        h = x
        if self.pos_proj is not None and pos_enc is not None:
            h = h + self.pos_proj(pos_enc)

        for layer in self.layers:
            #h = layer(h, src, dst, edge_w=w, slice_ids=slice_ids)
            h = layer(h, src, dst, edge_w=w, slice_ids=slice_ids, node_gate=node_gate)

        # refinement gate (same as yours)
        alpha = self.refine_alpha
        h = (1.0 - alpha) * h0 + alpha * h
        return h
