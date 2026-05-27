import math

import torch
import torch.nn as nn

from .modules import clones
from .lora import LoRALinear


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, use_lora = False, lora_rank=8, lora_alpha=16, lora_targets=("q", "v")):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        target_map = {
            "q": 0,
            "k": 1,
            "v": 2,
            "o": 3,
        }

        if use_lora:
            for target in lora_targets:
                if target not in target_map:
                    raise ValueError(f"Invalid LoRA target: {target}")
                
                idx = target_map[target]
                self.linears[idx] = LoRALinear(
                    self.linears[idx],
                    lora_rank,
                    lora_alpha
                )


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)
