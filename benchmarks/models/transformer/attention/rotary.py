# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from math import sqrt
from torch import softmax, Tensor, view_as_real, view_as_complex
from torch.nn import Module
from torch.nn.functional import dropout
from typing import Optional, Tuple
from ...lora import Linear


class RotaryAttention(Module):
    """
    LoRA wrapped Rotary Graph Multi-Head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout: float = .1, separate_proj: bool = False,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param separate_proj: use separated projections calculations or optimized
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout
        """
        assert not embed_dim % num_heads, 'embed_dim must be divisible by num_heads'
        assert not embed_dim // num_heads % 2, 'embed_dim // num_heads must be even'
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.lora_r = lora_r
        self.separate_proj = separate_proj or bool(lora_r)
        self._scale = 1 / sqrt(self.head_dim)

        if separate_proj or lora_r:
            self.q_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.k_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.v_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
        self.o_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], pad_mask: Optional[Tensor] = None, *,
                cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = x.shape

        # do projection
        if self.separate_proj:
            q = self.q_proj(x)  # BxTxH*E
            k = self.k_proj(x)  # BxSxH*E (KV seq len can differ from tgt_len with enabled cache trick)
            v = self.v_proj(x)  # BxSxH*E
        else:  # optimized
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        if cache is not None:
            # inference caching. batch should be left padded. shape should be BxSxH*E
            ck, cv = cache
            ck[:bsz, -tgt_len:] = k
            cv[:bsz, -tgt_len:] = v
            k, v = ck[:bsz], cv[:bsz]

        # BxTxH*E > BxTxHx1xE > BxHxTx1xE
        q = q.reshape(bsz, -1, self.num_heads, 1, self.head_dim).transpose(1, 2)

        # apply rotation matrix
        # BxSxH*E > BxSx1xHxE/2x2 > BxHx1xSxE/2x2 > BxHx1xSxE/2c
        k = view_as_complex(k.reshape(bsz, -1, 1, self.num_heads, self.head_dim // 2, 2).transpose(1, 3))
        # BxHx1xSxE/2c * Bx1xTxSxE/2c > BxHxTxSxE/2c > BxHxTxSxE/2x2 > BxHxTxSxE > BxHxTxExS
        k = view_as_real(k * attn_mask).flatten(start_dim=-2).transpose(-1, -2)

        # BxSxH*E > BxSxHxE > BxHxSxE
        v = v.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # BxHxTx1xE @ BxHxTxExS > BxHxTx1xS > BxHxTxS
        a = (q @ k).squeeze(3) * self._scale
        if pad_mask is not None:
            # BxHxTxS + Bx1xTxS > BxHxTxS
            a = a + pad_mask
        a = softmax(a, dim=-1)
        if self.training and self.dropout:
            a = dropout(a, self.dropout)

        # BxHxTxS @ BxHxSxE > BxHxTxE > BxTxHxE > BxTxH*E
        o = (a @ v).transpose(1, 2).flatten(start_dim=2)
        o = self.o_proj(o)

        if need_weights:
            a = a.sum(dim=1) / self.num_heads
            return o, a
        else:
            return o, None

    def merge_lora(self):
        """
        Transform LoRA MHA to normal
        """
        if not self.lora_r:
            return
        self.q_proj.merge_lora()
        self.k_proj.merge_lora()
        self.v_proj.merge_lora()
        self.o_proj.merge_lora()


__all__ = ['RotaryAttention']
