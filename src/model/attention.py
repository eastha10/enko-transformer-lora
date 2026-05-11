import math

import torch
import torch.nn as nn

from .modules import clones

def attention(query, key, value, mask = None, dropout = None):
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None: # 마스크가 주어졌다면
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = scores.softmax(dim=-1)
  if dropout is not None: # 드롭아웃이 주어졌다면
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0 # model dim이 head수와 딱떨어지는지 조건검사.
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    # 1~3번 Linear -> Q,K,V / 4번 Linear -> attention결과를 d_model로 변환하는 output projection용
    self.attn = None
    self.dropout = nn.Dropout(p = dropout)

  def forward(self, query, key, value, mask = None):
    if mask is not None:
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    query, key, value = [
        lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
        for lin, x in zip(self.linears, (query, key, value))
    ]

    x, self.attn = attention(
        query, key, value, mask=mask, dropout = self.dropout
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

