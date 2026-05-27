import copy

import torch
import torch.nn as nn

from .modules import (
    clones,
    LayerNorm,
    SublayerConnection,
    PositionwiseFeedForward,
    Embeddings,
    PositionalEncoding,
    Generator,
)
from .attention import MultiHeadedAttention


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
  def __init__(self, layer, N):
    super(Decoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = LayerNorm(layer.size)

  def forward(self, x, memory, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)

class DecoderLayer(nn.Module):
  def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size, dropout),3)

  def forward(self, x, memory, src_mask, tgt_mask):
    m = memory
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
    x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

    return self.sublayer[2](x, self.feed_forward)
  
def subsequent_mask(size): # 디코더가 예측에서 미래 토큰을 못보게 하기위해 사용
  attn_shape = (1, size, size)
  subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
      #diagonal=0 : 주대각선 포함 위쪽 제거 |  1: 주대각선 제외 위쪽 남김 |  -1: 주대각선 아래 한줄까지포함, 위쪽남김
      torch.uint8 # 8비트 부호없는 정수
  )
  return subsequent_mask == 0

def make_model(src_vocab, tgt_vocab, N_en=6, N_de=3, d_model=256, d_ff=1024, h=4, dropout=0.1): # 레이어 갯수 수정 시 N_en, N_de 건드릴 것.
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout)
  model = EncoderDecoder(
      Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_en),
      Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_de),
      nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
      nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
      Generator(d_model, tgt_vocab),
  )

  for p in model.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)
  return model