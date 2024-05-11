from itertools import repeat
from math import inf, sqrt
from torch import arange, empty_like, zeros_like, polar, ones_like, empty, no_grad, addmm, Tensor, softmax, cat, view_as_real, view_as_complex
from torch.nn import GELU, Module, ModuleList, LayerNorm, Embedding as tEmbedding, Parameter, init, Dropout, Linear as tLinear
from torch.nn.functional import embedding, dropout
from torchtyping import TensorType
from typing import Optional, Tuple, List, Type
from warnings import warn

class RotaryMultiheadAttention(Module):
    """
    LoRA wrapped Rotary Graph Multi-Head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=.1, separate_proj: bool = False,
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


class Linear(tLinear):
    """
    LoRA wrapped Linear layer.
    """
    def __init__(self, in_features: int, out_features: int, *args, lora_r: int = 0, lora_alpha: float = 1.,
                 lora_dropout: float = 0., **kwargs):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout

        See torch.nn.Linear for other params
        """
        super().__init__(in_features, out_features, *args, **kwargs)
        self.lora_r = lora_r
        if lora_r:  # enable lora
            self.weight.requires_grad = False  # freeze main weights
            self.lora_a = Parameter(init.kaiming_uniform_(empty(lora_r, in_features), a=sqrt(5)))
            self.lora_b = Parameter(init.zeros_(empty(out_features, lora_r)))
            self.lora_dropout = lora_dropout
            self.lora_alpha = lora_alpha
            self._lora_scaling = lora_alpha / lora_r

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        if self.lora_r:
            if self.training and self.lora_dropout:
                x = dropout(x, self.lora_dropout)
            a = x @ self.lora_a.transpose(0, 1)
            return addmm(out.flatten(end_dim=-2), a.flatten(end_dim=-2), self.lora_b.transpose(0, 1),
                         alpha=self._lora_scaling).view(out.shape)
        return out

    def merge_lora(self):
        """
        Transform LoRA linear to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_b @ self.lora_a) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_dropout, self.lora_alpha, self._lora_scaling

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.lora_r:
            return  r + f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}'
        return r


def _update_lora(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        q_w, k_w, v_w = state_dict.pop(prefix + 'in_proj_weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'in_proj_bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b
    elif prefix + 'qkv_proj.weight' in state_dict:  # transform packed projection
        q_w, k_w, v_w = state_dict.pop(prefix + 'qkv_proj.weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'qkv_proj.bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b


def _update_packed(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        state_dict[prefix + 'qkv_proj.weight'] = state_dict.pop(prefix + 'in_proj_weight')
        state_dict[prefix + 'qkv_proj.bias'] = state_dict.pop(prefix + 'in_proj_bias')
    elif prefix + 'q_proj.weight' in state_dict:  # transform unpacked projection
        q_w = state_dict.pop(prefix + 'q_proj.weight')
        k_w = state_dict.pop(prefix + 'k_proj.weight')
        v_w = state_dict.pop(prefix + 'v_proj.weight')
        q_b = state_dict.pop(prefix + 'q_proj.bias')
        k_b = state_dict.pop(prefix + 'k_proj.bias')
        v_b = state_dict.pop(prefix + 'v_proj.bias')
        state_dict[prefix + 'qkv_proj.weight'] = cat([q_w, k_w, v_w])
        state_dict[prefix + 'qkv_proj.bias'] = cat([q_b, k_b, v_b])


class MultiheadAttention(Module):
    """
    LoRA wrapped Multi-Head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=.1, separate_proj: bool = False,
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
            self._register_load_state_dict_pre_hook(_update_lora)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
            self._register_load_state_dict_pre_hook(_update_packed)
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

        # BxTxH*E > BxTxHxE > BxHxTxE
        q = q.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # BxSxH*E > BxSxHxE > BxHxExS
        k = k.reshape(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        # BxSxH*E > BxSxHxE > BxHxSxE
        v = v.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # BxHxTxE @ BxHxExS > BxHxTxS
        a = (q @ k) * self._scale
        if attn_mask is not None:
            a = a + attn_mask
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


class EncoderLayer(Module):
    r"""EncoderLayer based on torch.nn.TransformerEncoderLayer, but batch always first and returns also attention.

    :param d_model: the number of expected features in the input (required).
    :param nhead: the number of heads in the multiheadattention models (required).
    :param dim_feedforward: the dimension of the feedforward network model (required).
    :param dropout: the dropout value (default=0.1).
    :param activation: the activation function of the intermediate layer. Default: GELU.
    :param layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    :param norm_first: if `True`, layer norm is done prior to self attention, multihead
        attention and feedforward operations, respectively. Otherwise, it's done after.
    :param lora_r: LoRA factorization dimension
    :param lora_alpha: LoRA scaling factor
    :param lora_dropout: LoRA input dropout
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5,
                 norm_first: bool = False, attention: Type[Module] = MultiheadAttention,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        super().__init__()
        self.self_attn = attention(d_model, nhead, dropout, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)  # noqa

        self.linear1 = Linear(d_model, dim_feedforward, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear2 = Linear(dim_feedforward, d_model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()
        self.norm_first = norm_first

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], pad_mask: Optional[Tensor] = None, *,
                cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_embedding: bool = True, need_weights: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        nx = self.norm1(x) if self.norm_first else x  # pre-norm or post-norm
        e, a = self.self_attn(nx, attn_mask, pad_mask, cache=cache, need_weights=need_weights)

        if need_embedding:
            x = x + self.dropout1(e)
            if self.norm_first:
                return x + self._ff(self.norm2(x)), a
            # else: post-norm
            x = self.norm1(x)
            return self.norm2(x + self._ff(x)), a
        return None, a

    def merge_lora(self):
        """
        Transform LoRA Encoder to normal
        """
        self.self_attn.merge_lora()
        self.linear1.merge_lora()
        self.linear2.merge_lora()

    def _ff(self, x):
        return self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))

class Embedding(tEmbedding):
    """
    LoRA wrapped Embedding layer.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, *args,
                 lora_r: int = 0, lora_alpha: float = 1., neg_inf_idx: Optional[int] = None, **kwargs):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param neg_inf_idx: -inf frozen embedding vector

        See torch.nn.Embedding for other params
        """
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.neg_inf_idx = neg_inf_idx
        self.lora_r = lora_r
        if neg_inf_idx is not None:
            with no_grad():
                self.weight[neg_inf_idx].fill_(-inf)
        if lora_r:  # enable lora
            self.weight.requires_grad = False  # freeze main weights
            self.lora_a = Parameter(init.zeros_(empty(num_embeddings, lora_r)))
            self.lora_b = Parameter(init.normal_(empty(embedding_dim, lora_r)))
            self.lora_alpha = lora_alpha
            self._lora_scaling = lora_alpha / lora_r

    def forward(self, x: Tensor) -> Tensor:
        emb = super().forward(x)
        if self.lora_r:
            a = embedding(x, self.lora_a, self.padding_idx, self.max_norm,
                          self.norm_type, self.scale_grad_by_freq, self.sparse)
            return addmm(emb.flatten(end_dim=-2), a.flatten(end_dim=-2), self.lora_b.transpose(0, 1),
                         alpha=self._lora_scaling).view(emb.shape)
        return emb

    def merge_lora(self):
        """
        Transform LoRA embedding to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_a @ self.lora_b.transpose(0, 1)) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_alpha, self._lora_scaling

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.neg_inf_idx is not None:
            r += f', neg_inf_idx={self.neg_inf_idx}'
        if self.lora_r:
            r += f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}'
        return r


class ChytorchRotary(Module):
    """
    Inspired by https://arxiv.org/pdf/2104.09864.pdf
    """
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, nhead: int = 16,
                 num_layers: int = 8, dim_feedforward: int = 3072, shared_weights: bool = False, dropout: float = 0.1,
                 activation=GELU, layer_norm_eps: float = 1e-5, norm_first: bool = True, post_norm: bool = True,
                 perturbation: float = 0., theta: float = 10000.,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        """
        Molecule RotaryTransformerEncoder layer.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param shared_weights: ALBERT-like encoder weights sharing.
        :param norm_first: do pre-normalization in encoder layers.
        :param post_norm: do normalization of output. Works only when norm_first=True.
        :param perturbation: add perturbation to embedding (https://aclanthology.org/2021.naacl-main.460.pdf).
            Disabled by default
        :param lora_r: LoRA factorization dimension size in encoder embeddings. Disabled by default.
        :param lora_alpha: LoRA scaling factor.
        :param lora_dropout: LoRA input dropout.
        """
        assert perturbation >= 0, 'zero or positive perturbation expected'
        assert not d_model % nhead, 'd_model must be divisible by nhead'
        assert not d_model // nhead % 2, 'd_model // nhead must be even'
        super().__init__()

        # same as graphormer
        self.atoms_encoder = Embedding(121, d_model, 0, lora_r=lora_r, lora_alpha=lora_alpha)
        self.neighbors_encoder = Embedding(max_neighbors + 3, d_model, 0, lora_r=lora_r, lora_alpha=lora_alpha)

        # do roll for disabling bias of disconnected atoms attention
        head_dim = d_model // nhead
        angle = arange(max_distance + 3).outer(theta ** (arange(0, -head_dim, -2) / head_dim)).roll(1, 0)
        rotary = polar(ones_like(angle), angle)  # DxE/2c
        self.register_buffer('rotary_matrix', rotary, persistent=False)

        self.max_distance = max_distance
        self.max_neighbors = max_neighbors
        self.perturbation = perturbation
        self.num_layers = num_layers
        self.nhead = nhead
        self.post_norm = post_norm
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        self.shared_weights = shared_weights
        if shared_weights:
            self.layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first,
                                      attention=RotaryMultiheadAttention,
                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.layers = [self.layer] * num_layers
        else:
            # layers sharing scheme can be manually changed. e.g. pairs of shared encoders
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first, attention=RotaryMultiheadAttention,
                                                  lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                                     for _ in range(num_layers))

    def forward(self, batch, /, *,
                cache: Optional[List[Tuple[TensorType['batch', 'atoms', 'embedding'],
                                           TensorType['batch', 'atoms', 'embedding']]]] = None) -> \
            TensorType['batch', 'atoms', 'embedding']:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for reaction cls or training tricks like MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for training tricks like MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.
        """
        cache = repeat(None) if cache is None else iter(cache)
        atoms, neighbors, distances = batch

        # cls token in neighbors coded by 0
        x = self.atoms_encoder(atoms) + self.neighbors_encoder(neighbors)

        if self.perturbation and self.training:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)

        # BxTxS > Bx1xTxS
        p_mask = zeros_like(distances, dtype=x.dtype).masked_fill_(distances == 0, -inf).unsqueeze_(1)

        # BxTxS > Bx1xTxS > Bx1xTxSxE/2c
        d_mask = embedding(distances.unsqueeze(1), self.rotary_matrix)
        for lr, c in zip(self.layers, cache):
            x, _ = lr(x, d_mask, p_mask, cache=c)  # noqa

        if self.post_norm:
            return self.norm(x)
        return x

    def merge_lora(self):
        """
        Transform LoRA layers to normal
        """
        self.atoms_encoder.merge_lora()
        self.neighbors_encoder.merge_lora()
        for layer in self.layers:
            layer.merge_lora()