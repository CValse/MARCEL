from itertools import repeat
from math import log, inf,sqrt
from torch import no_grad, empty_like, arange, exp, sin, cos, Tensor, empty, addmm, baddbmm, bmm, softmax, cat, long
from torch.nn import GELU, Module, ModuleList, LayerNorm, Dropout, Embedding as tEmbedding, Parameter, init, Linear as tLinear
from torchtyping import TensorType
from typing import Optional, Tuple, List, Callable, NamedTupleMeta, NamedTuple
from warnings import warn
#from .lora import Embedding, EncoderLayer
#from ..utils.data import MoleculeDataBatch

from numpy import empty, ndarray, sqrt, square, ones, digitize, arange, int32
from torch import IntTensor, Size, zeros as t_zeros, ones as t_ones, int32 as t_int32, LongTensor, FloatTensor, vstack, empty as tempty, arange as t_arange, tensor, zeros_like

from torch.nn.functional import embedding, dropout
#from torch_scatter import scatter
import torch

class ChytorchDiscrete(Module):
    def __init__(
            self, 
            max_neighbors: int = 14,
            max_distance: int = 83,
            d_model: int = 1024,
            nhead: int = 16,
            num_layers: int = 8,
            dim_feedforward: int = 3072,
            shared_weights: bool = True,
            shared_attention_bias: bool = True,
            dropout: float = 0.1,
            activation=GELU,
            layer_norm_eps: float = 1e-5,
            norm_first: bool = False,
            post_norm: bool = False,
            zero_bias: bool = False,
            perturbation: float = 0.,
            max_tokens: int = 121,
            lora_r: int = 0,
            lora_alpha: float = 1.,
            lora_dropout: float = 0.):
        #super(ChytorchDiscrete, self).__init__()
        super().__init__()
        
        self.embedding = EmbeddingBag(max_neighbors, d_model, perturbation, max_tokens, lora_r, lora_alpha, consider_chirality = self.consider_chirality)

        self.shared_attention_bias = shared_attention_bias
        if shared_attention_bias:
            self.distance_encoder = Embedding(max_distance + 3, nhead, int(zero_bias) or None, neg_inf_idx=0)
            # None filled encoders mean reusing previously calculated bias. possible manually create different arch.
            # this done for speedup in comparison to layer duplication.
            self.distance_encoders = [None] * num_layers
            self.distance_encoders[0] = self.distance_encoder  # noqa
        else:
            self.distance_encoders = ModuleList(Embedding(max_distance + 3, nhead,
                                                          int(zero_bias) or None, neg_inf_idx=0)
                                                for _ in range(num_layers))

        self.max_distance = max_distance
        self.max_tokens = max_tokens
        self.max_neighbors = max_neighbors
        self.perturbation = perturbation
        self.num_layers = num_layers
        self.post_norm = post_norm
        
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        self.shared_weights = shared_weights
        if shared_weights:
            self.layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first,
                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.layers = [self.layer] * num_layers
        else:
            # layers sharing scheme can be manually changed. e.g. pairs of shared encoders
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first, lora_r=lora_r, lora_alpha=lora_alpha,
                                                  lora_dropout=lora_dropout) for _ in range(num_layers))
        self._register_load_state_dict_pre_hook(_update)
    
    def forward(self, 
        z: Tensor, 
        pos: Tensor,
        batch: Tensor = None):
        assert z.dim() == 1 and z.dtype == long

        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for reaction cls or training tricks like MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for training tricks like MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.
        """
        #cache = repeat(None) if cache is None else iter(cache)
        batch = zeros_like(z) if batch is None else batch
        
        #############################
        # Identify non-hydrogen atoms
        non_hydrogen_indices = torch.where(z != 0)[0]
        # Prune tensors
        pruned_batch = batch[non_hydrogen_indices]
        pruned_z = z[non_hydrogen_indices]
        pruned_pos = pos[non_hydrogen_indices]
        
        N = max(pruned_batch.bincount())
        num_batches = max(pruned_batch)+1
        batched_z = t_zeros(num_batches, N, dtype=t_int32, device=z.device)
        batched_hgs = t_zeros(num_batches, N, dtype=t_int32, device=z.device)
        batched_dist = t_zeros(num_batches, N, N, device=z.device)
        
        # Populate the batched_tensor
        for i in range(num_batches):
            indices_h = (batch == i).nonzero(as_tuple=True)[0]
            indices = (pruned_batch == i).nonzero(as_tuple=True)[0]
            batched_z[i, :len(indices)] = pruned_z[indices] + 3
            
            pos_i_h = pos[indices_h,:]
            pos_i = pruned_pos[indices,:]
            
            dist_matrix = torch.cdist(pos_i, pos_i_h)
            hydrogen_mask = (z[indices_h] == 0)
            hgs = torch.sum((dist_matrix < 1.2) & hydrogen_mask, dim=1)
            batched_hgs[i, :len(indices)] = hgs + 2
            
            diff = pos_i[None, :, :] - pos_i[:, None, :]  # NxNx3
            dist = (diff ** 2).sum(dim=-1).sqrt()  # BxNxN
            batched_dist[i, :len(indices), :len(indices)] = dist

        #add 1 cls
        atoms = t_ones(batched_z.shape[0], batched_z.shape[1] + 1, dtype=t_int32, device=z.device)
        neighbors = t_zeros(batched_hgs.shape[0], batched_hgs.shape[1] + 1, dtype=t_int32, device=z.device)
        #sum3 from default MAECEL
        atoms[:,1:] = batched_z.int()
        neighbors[:,1:] = batched_hgs.int()

        short_cutoff =.9
        long_cutoff = 5.
        precision = .05
        _bins = arange(short_cutoff - 3 * precision, long_cutoff, precision)
        _bins[:3] = [-1, 0, .01]  # trick for self-loop coding
        self.max_distance = len(_bins) - 2  # param for MoleculeEncoder

        dist = digitize(batched_dist.cpu(), _bins)
        tmp = ones((num_batches, atoms.shape[1], atoms.shape[1]), dtype=int32)
        tmp[:,1:, 1:] = dist
        distances = tensor(tmp, dtype=t_int32, device = z.device)

        # cls token in neighbors coded by 0
        #x = self.atoms_encoder(atoms.int()) #+ self.neighbors_encoder(neighbors)  BxNxEMB
        x = self.embed_sum(atoms,neighbors)

        if self.perturbation and self.training:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)

        if self.shared_attention_bias:
            d_mask = self.distance_encoders[0](distances.int()).permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN
            x = self.access_layer(x, d_mask)
            for lr in self.layers[self.access_layer.layer_i:-1]:
                x, _ = lr(x, d_mask)
            x, _ = self.layers[-1](x, d_mask)
        
        else:
            d_mask = self.distance_encoders[0](distances.int()).permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN
            x = self.access_layer(x, distances.int())

            for lr, d in zip(self.layers[self.access_layer.layer_i:], self.distance_encoders[self.access_layer.layer_i:]):
                if d is not None:
                    d_mask = d(distances.int()).permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN
                x, _ = lr(x, d_mask)
            #x, _ = self.layers[-1](x, d_mask)

        if self.post_norm:
            x = self.norm(x)
            
        #x = scatter(x, batch, dim=0, reduce='mean')

        return x[:,0]

    def merge_lora(self):
        """
        Transform LoRA layers to normal
        """
        self.atoms_encoder.merge_lora()
        self.neighbors_encoder.merge_lora()
        for layer in self.layers:
            layer.merge_lora()

    @property
    def centrality_encoder(self):
        warn('centrality_encoder renamed to neighbors_encoder in chytorch 1.37', DeprecationWarning)
        return self.neighbors_encoder

    @property
    def spatial_encoder(self):
        warn('spatial_encoder renamed to distance_encoder in chytorch 1.37', DeprecationWarning)
        return self.distance_encoder
        
        
def _update(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'centrality_encoder.weight' in state_dict:
        warn('fixed chytorch<1.37 checkpoint', DeprecationWarning)
        state_dict[prefix + 'neighbors_encoder.weight'] = state_dict.pop(prefix + 'centrality_encoder.weight')
        state_dict[prefix + 'distance_encoder.weight'] = state_dict.pop(prefix + 'spatial_encoder.weight')


def positional_init(distance_encoder):
    # from: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    max_distance, nhead = distance_encoder.weight.shape
    angle = arange(max_distance - 2).unsqueeze_(-1) * exp(arange(0, nhead, 2) * -log(100) / nhead)
    with no_grad():
        distance_encoder.weight[2:, ::2] = sin(angle)
        distance_encoder.weight[2:, 1::2] = cos(angle)

        
class LayerModule_with_dist(Module):
    def __init__(self, layers, distance_encoders, layer_i=1):
        super().__init__()

        self.layers = layers
        self.distance_encoders = distance_encoders
        self.layer_i = layer_i


    def forward(self, x, distances):

        for lr, d in zip(self.layers[:self.layer_i], self.distance_encoders[:self.layer_i]):  # noqa
            if d is not None:
                d_mask = d(distances).permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN
            x, _ = lr(x, d_mask)
            #x, _ = lr(x, d)

        # x, _ = self.layers[-1](x, d_mask)
        return x
        
class LayerModule(Module):
    def __init__(self, layers, layer_i=1):
        super().__init__()

        self.layers = layers
        self.layer_i = layer_i


    def forward(self, x, d_mask):

        for lr in self.layers[:self.layer_i]:  # noqa
            x, _ = lr(x, d_mask)

        # x, _ = self.layers[-1](x, d_mask)
        return x
        
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
        self.num_embeddings = num_embeddings
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
        assert x.min() >= 0 and x.max() < self.num_embeddings
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
                 norm_first: bool = False, lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.linear1 = Linear(d_model, dim_feedforward, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear2 = Linear(dim_feedforward, d_model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()
        self.norm_first = norm_first

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], *, cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_embedding: bool = True, need_weights: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        nx = self.norm1(x) if self.norm_first else x  # pre-norm or post-norm
        e, a = self.self_attn(nx, attn_mask, cache=cache, need_weights=need_weights)

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
    def __init__(self, embed_dim, num_heads, dropout=0., separate_proj: bool = False,
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

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], *, cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = x.shape
        x = x.transpose(1, 0)  # switch batch and sequence dims

        # do projection
        if self.separate_proj:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:  # optimized
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        if cache is not None:
            # inference caching. batch should be left padded. shape should be SxBxE
            ck, cv = cache
            ck[-tgt_len:] = k
            cv[-tgt_len:] = v
            k, v = ck, cv

        q = q.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # B*HxSxE
        k = k.reshape(-1, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)  # B*HxExS
        v = v.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # B*HxSxE

        if attn_mask is None:
            a = bmm(q, k) * self._scale
        else:
            a = baddbmm(attn_mask, q, k, alpha=self._scale)  # scaled dot-product with bias
        a = softmax(a, dim=-1)
        if self.training and self.dropout:
            a = dropout(a, self.dropout)

        o = bmm(a, v).transpose(0, 1).contiguous().view(-1, self.embed_dim)
        o = self.o_proj(o).view(tgt_len, bsz, -1).transpose(0, 1)  # switch dimensions back

        if need_weights:
            a = a.view(bsz, -1, tgt_len, tgt_len)
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

class EmbeddingSum(Module):
    def __init__(self,atoms, neighbors):
        super().__init__()
        self.neighbors = neighbors
        self.atoms = atoms
    def forward(self,atoms, neighbors):
        x = self.atoms(atoms) + self.neighbors(neighbors)
        return x
