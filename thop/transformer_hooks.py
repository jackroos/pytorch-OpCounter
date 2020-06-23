import torch
import torch.nn as nn


def count_attn(m: nn.MultiheadAttention, x: (torch.Tensor, torch.Tensor, torch.Tensor), y: torch.Tensor):

    q = x[0]
    k = x[1]
    v = x[2]

    Lq, Nq, Dq = q.shape
    Lk, Nk, Dk = k.shape
    Lv, Nv, Dv = v.shape

    assert Lk == Lv and Nq == Nk and Nq == Nv

    in_proj_ops = Nq * Dq * (Lq * Dq + Lk * Dk + Lv * Dv)
    attn_ops = 2 * Nq * Lq * Lk * Dq
    out_proj_ops = Nq * Lq * Dq * Dq

    total_ops = in_proj_ops + attn_ops + out_proj_ops

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_ln(m: nn.LayerNorm, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    nelements = x.numel()
    if m.elementwise_affine:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += torch.DoubleTensor([int(total_ops)])

