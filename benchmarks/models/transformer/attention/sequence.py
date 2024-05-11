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
from torch import Tensor, view_as_real, view_as_complex, arange, polar, ones_like
from torch.nn import Module
from torch.nn import functional  # ad-hoc for pytorch<2.0 (scaled_dot_product_attention)
from torch.nn.functional import dropout
from typing import Optional, Tuple
from ...lora import Linear


class SequenceAttention(Module):
    """
    LoRA wrapped Rotary Multi-Head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout: float = .1, separate_proj: bool = False,
                 theta: float = 10000., max_length: int = 1024,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param separate_proj: use separated projections calculations or optimized
        :param theta: rotation base (see article)
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
        self.head_dim = head_dim = embed_dim // num_heads
        self.theta = theta
        self.max_length = max_length
        self.lora_r = lora_r
        self.separate_proj = separate_proj or bool(lora_r)

        if separate_proj or lora_r:
            self.q_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.k_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.v_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
        self.o_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        angle = arange(max_length).outer(theta ** (arange(0, -head_dim, -2) / head_dim))
        rotary = polar(ones_like(angle), angle).unsqueeze_(1)  # SxE/2c
        self.register_buffer('rotary_matrix', rotary, persistent=False)

    def forward(self, x: Tensor, attn_mask=None, pad_mask=None, *,
                cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        assert not need_weights, 'need_weights is not supported and required only for API compatibility'
        if cache is None:
            left = 0
            right = x.size(1)
        else:
            # queries should be properly rotated
            right = cache[0].size(1)
            left = right - x.size(1)

        # do projection
        if self.separate_proj:
            q = self.q_proj(x)  # BxTxH*E
            k = self.k_proj(x)  # BxSxH*E (KV seq len can differ from tgt_len with enabled cache trick)
            v = self.v_proj(x)  # BxSxH*E
        else:  # optimized
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # BxTxH*E > BxTxHxE
        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))  # BxSxHxE
        v = v.unflatten(-1, (self.num_heads, -1))  # BxSxHxE

        # apply rotation matrix
        # BxTxHxE > BxTxHxE/2x2 > BxTxHxE/2c * Tx1xE/2c > BxTxHxE/2x2 > BxTxHxE
        q = view_as_real(view_as_complex(q.unflatten(-1, (-1, 2))) * self.rotary_matrix[left:right]).flatten(-2)
        k = view_as_real(view_as_complex(k.unflatten(-1, (-1, 2))) * self.rotary_matrix[left:right]).flatten(-2)

        if cache is not None:
            # inference caching. shape should be BxSxH*E
            # use right padding if effective sentence length shorter than cached
            bsz = x.size(0)
            ck, cv = cache
            # BxSxHxE > BxSxH*E
            ck[:bsz, left:] = k.flatten(-2)
            cv[:bsz, left:] = v.flatten(-2)
            k = ck[:bsz].unflatten(-1, (self.num_heads, -1))
            v = cv[:bsz].unflatten(-1, (self.num_heads, -1))

        q = q.transpose(1, 2)  # BxTxHxE > BxHxTxE
        k = k.transpose(1, 2)  # BxHxSxE
        v = v.transpose(1, 2)  # BxHxSxE

        o = functional.scaled_dot_product_attention(q, k, v, None, self.dropout, True)
        # BxHxSxE > BxSxHxE > BxSxH*E
        o = self.o_proj(o.transpose(1, 2).flatten(-2))
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


__all__ = ['SequenceAttention']
