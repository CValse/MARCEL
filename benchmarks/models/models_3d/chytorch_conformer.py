from math import sqrt, pi, inf
from torch import empty, exp, square, isinf, nan_to_num, sigmoid, stack, empty_like, Tensor, LongTensor
from torch.nn import Module, GELU, Embedding, Parameter, ModuleList, Sequential, Linear, LayerNorm, Dropout, MultiheadAttention
from torch.nn.init import uniform_, constant_
from torchtyping import TensorType
from typing import Tuple, Union, List, Optional, NamedTuple, NamedTupleMeta
#from .transformer import EncoderLayer
from numpy import clip, zeros as n_zeros, int32

#from numpy import ndarray, sqrt, ones, digitize, arange
from torch import IntTensor, Size, zeros as t_zeros, ones as t_ones, int32 as t_int32, LongTensor, FloatTensor, vstack, empty as tempty, arange as t_arange, tensor, zeros_like, long, where

from torch.nn.functional import embedding, dropout
import torch

class ChytorchConformer(Module):
    def __init__(self, *, 
                implicit: bool = False, 
                nkernel: int = 128, 
                shared_layers: bool = True,
                d_model: int = 1024, 
                nhead: int = 16, 
                num_layers: int = 8, 
                dim_feedforward: int = 3072,
                dropout: float = 0.1, 
                activation=GELU, 
                layer_norm_eps: float = 1e-5, 
                norm_first: bool = False,
                post_norm: bool = False, 
                perturbation: float = 0.):
        """
        Reimplemented Graphormer3D <https://github.com/microsoft/Graphormer>

        :param implicit: use hydrogens count embedding instead explicitly presented.
        :param nkernel: number of Gaussian functions.
        :param shared_layers: ALBERT-like encoder layer sharing.
        :param norm_first: do pre-normalization in encoder layers.
        :param post_norm: do normalization of output. Works only when norm_first=True.
        :param perturbation: add perturbation to embedding (https://aclanthology.org/2021.naacl-main.460.pdf).
            Disabled by default
        """
        super(ChytorchConformer, self).__init__()
        self.atoms_encoder = Embedding(121, d_model, 0)
        self.implicit = implicit
        if implicit:
            self.hydrogens_encoder = Embedding(7, d_model, 0)

        self.gaussian = GaussianLayer(nkernel)

        self.spatial_encoder = Sequential(Linear(nkernel, nkernel),
                                          LayerNorm(nkernel, eps=layer_norm_eps),
                                          activation(),
                                          Dropout(dropout),
                                          Linear(nkernel, nhead))

        self.perturbation = perturbation and perturbation / sqrt(d_model)
        self.post_norm = post_norm
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        if shared_layers:
            self.layer = layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                              norm_first)
            self.layers = [layer] * num_layers
        else:
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first) for _ in range(num_layers))

    def forward(self, 
        z: Tensor, 
        pos: Tensor,
        batch: Tensor = None,*, need_embedding: bool = True, need_weights: bool = False,
        averaged_weights: bool = False, intermediate_embeddings: bool = False):
        assert z.dim() == 1 and z.dtype == long

        if not need_weights:
            assert not averaged_weights, 'averaging without need_weights'
            assert need_embedding, 'at least weights or embeddings should be returned'
        elif intermediate_embeddings:
            assert need_embedding, 'need_embedding should be active for intermediate_embeddings option'

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
        hydrogens = t_zeros(batched_hgs.shape[0], batched_hgs.shape[1] + 1, dtype=t_int32, device=z.device)
        #sum3 from default MAECEL
        atoms[:,1:] = batched_z.int()
        hydrogens[:,1:] = batched_hgs.int()
        #######################################
        #cache = repeat(None) if cache is None else iter(cache)

        distance_cutoff = 5.
        #clip(batched_dist.cpu().numpy(), None, distance_cutoff, out=batched_dist)
        batched_dist = clip(batched_dist.cpu(), a_min=None, a_max=distance_cutoff)
        tmp = n_zeros((num_batches, atoms.shape[1], atoms.shape[1]), dtype=int32)
        tmp[:,1:, 1:] = batched_dist
        
        distances = tensor(tmp, dtype=long, device = z.device)
        #distances = LongTensor(tmp)

        gf = self.gaussian(atoms.int(), distances)  # BxNxNxK
        d_mask = self.spatial_encoder(gf).masked_fill_(isinf(distances).unsqueeze_(-1), -inf)  # BxNxNxK > BxNxNxH
        d_mask = d_mask.permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN

        x = self.atoms_encoder(atoms.int())
        if self.implicit:  # cls token in hydrogens coded by 0
            x = x + self.hydrogens_encoder(hydrogens)
        if self.perturbation:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)

        if intermediate_embeddings:
            embeddings = [x]

        if averaged_weights:  # average attention weights from each layer
            w = []
            for lr in self.layers[:-1]:  # noqa
                x, a = lr(x, d_mask, need_weights=True)
                w.append(a)
                if intermediate_embeddings:
                    embeddings.append(x)  # noqa
            x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=True)
            w.append(a)
            w = stack(w, dim=-1).mean(-1)
            if need_embedding:
                if self.post_norm:
                    x = self.norm(x)
                if intermediate_embeddings:
                    return x, w, embeddings
                return x, w
            return w

        for lr in self.layers[:-1]:  # noqa
            x, _ = lr(x, d_mask)
            if intermediate_embeddings:
                embeddings.append(x)  # noqa
        x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if self.post_norm:
                x = self.norm(x)
            #if intermediate_embeddings:
            #    if need_weights:
            #        return x, a, embeddings
            #   return x[:,0], embeddings
            #elif need_weights:
            #    return x[:,0], a[:,0]
            return x[:,0]
        return a[:,0]


class GaussianLayer(Module):
    """
    x = a * d + b
    g = exp(-.5 * ((x - u) ** 2 / s ** 2)) / (s * sqrt(2 * pi))
    """

    def __init__(self, nkernel: int, posinf: float = 10., eps: float = 1e-5):
        super().__init__()
        self.nkernel = nkernel
        self.posinf = posinf
        self.eps = eps
        self.mu = Parameter(empty(nkernel))
        self.sigma = Parameter(empty(nkernel))
        self.a = Parameter(empty(121, 121))
        self.b = Parameter(empty(121, 121))
        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.mu, 0, 3)
        uniform_(self.sigma, 1, 3)
        constant_(self.a, 1)
        constant_(self.b, 0)

    def forward(self, atoms, distances):
        a = self.a[atoms.unsqueeze(-1), atoms.unsqueeze(1)]  # [BxNx1, Bx1xN] > BxNxN
        b = self.b[atoms.unsqueeze(-1), atoms.unsqueeze(1)]
        # exp(-inf) > nan
        d = nan_to_num(distances, posinf=self.posinf)
        # BxNxN > BxNxNx1 > BxNxNxK
        x = (a * d + b).unsqueeze(-1).expand(-1, -1, -1, self.nkernel)
        return exp(-.5 * square((x - self.mu) / self.sigma)) / ((self.sigma.abs() + self.eps) * sqrt(2 * pi))


class SigmoidLayer(Module):
    def __init__(self, nkernel: int = 128, d_model: int = 1024, dropout: float = .1, layer_norm_eps: float = 1e-5):
        super().__init__()
        # atom-type-pair set of linear layers
        self.a = Parameter(empty(121, 121, nkernel))
        self.b = Parameter(empty(121, 121, nkernel))

        self.layer_norm = LayerNorm(nkernel, layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.head = Linear(nkernel, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.a, -1, 1)
        uniform_(self.b, 0, 5)

    def forward(self, atoms, distances):
        a = self.a[atoms.unsqueeze(-1), atoms.unsqueeze(1)]  # [BxNx1, Bx1xN] > BxNxNxK
        b = self.b[atoms.unsqueeze(-1), atoms.unsqueeze(1)]

        # BxNxN > BxNxNxK
        x = a * (distances + b)
        return self.head(self.dropout(sigmoid(self.layer_norm(x))))


class EncoderLayer(Module):
    r"""EncoderLayer based on torch.nn.TransformerEncoderLayer, but batch always first and returns also attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer. Default: GELU.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise, it's done after.
            Default: ``False`` (after).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5,
                 norm_first: bool = False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()
        self.norm_first = norm_first

    def forward(self, x: Tensor, attn_mask: Tensor, *,
                need_embedding: bool = True, need_weights: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        tx = x.transpose(1, 0)  # switch Batch and Sequence. torch-1.8 compatible.
        nx = self.norm1(tx) if self.norm_first else tx  # pre-norm or post-norm
        e, a = self.self_attn(nx, nx, nx, attn_mask=attn_mask, need_weights=need_weights)

        if need_embedding:
            x = (tx + self.dropout1(e)).transpose(1, 0)  # switch Sequence and Batch back
            if self.norm_first:
                return x + self._ff(self.norm2(x)), a
            # else: post-norm
            x = self.norm1(x)
            return self.norm2(x + self._ff(x)), a
        return None, a

    def _ff(self, x):
        return self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))

'''
class ConformerDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch', 'atoms', int]
    hydrogens: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', float]


class MultipleInheritanceNamedTupleMeta(NamedTupleMeta):
    def __new__(mcls, typename, bases, ns):
        if NamedTuple in bases:
            base = super().__new__(mcls, '_base_' + typename, bases, ns)
            bases = (base, *(b for b in bases if not isinstance(b, NamedTuple)))
        return super(NamedTupleMeta, mcls).__new__(mcls, typename, bases, ns)
        
class DataTypeMixin(metaclass=MultipleInheritanceNamedTupleMeta):
    def to(self, *args, **kwargs):
        return type(self)(*(x.to(*args, **kwargs) for x in self))

    def cpu(self, *args, **kwargs):
        return type(self)(*(x.cpu(*args, **kwargs) for x in self))

    def cuda(self, *args, **kwargs):
        return type(self)(*(x.cuda(*args, **kwargs) for x in self))

class ChytorchConformer(Module):
    def __init__(self, *, 
                implicit: bool = False, 
                nkernel: int = 128, 
                shared_layers: bool = True,
                d_model: int = 1024, 
                nhead: int = 16, 
                num_layers: int = 8, 
                dim_feedforward: int = 3072,
                dropout: float = 0.1, 
                activation=GELU, 
                layer_norm_eps: float = 1e-5, 
                norm_first: bool = False,
                post_norm: bool = False, 
                perturbation: float = 0.):
        """
        Reimplemented Graphormer3D <https://github.com/microsoft/Graphormer>

        :param implicit: use hydrogens count embedding instead explicitly presented.
        :param nkernel: number of Gaussian functions.
        :param shared_layers: ALBERT-like encoder layer sharing.
        :param norm_first: do pre-normalization in encoder layers.
        :param post_norm: do normalization of output. Works only when norm_first=True.
        :param perturbation: add perturbation to embedding (https://aclanthology.org/2021.naacl-main.460.pdf).
            Disabled by default
        """
        super(ChytorchConformer, self).__init__()
        self.atoms_encoder = Embedding(121, d_model, 0)
        self.implicit = implicit
        if implicit:
            self.hydrogens_encoder = Embedding(7, d_model, 0)

        self.gaussian = GaussianLayer(nkernel)

        self.spatial_encoder = Sequential(Linear(nkernel, nkernel),
                                          LayerNorm(nkernel, eps=layer_norm_eps),
                                          activation(),
                                          Dropout(dropout),
                                          Linear(nkernel, nhead))

        self.perturbation = perturbation and perturbation / sqrt(d_model)
        self.post_norm = post_norm
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        if shared_layers:
            self.layer = layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                              norm_first)
            self.layers = [layer] * num_layers
        else:
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first) for _ in range(num_layers))

    def forward(self, batch, /, *, need_embedding: bool = True, need_weights: bool = False,
                averaged_weights: bool = False, intermediate_embeddings: bool = False) -> \
            Union[TensorType['batch', 'atoms', 'embedding'], TensorType['batch', 'atoms', 'atoms'],
                  Tuple[TensorType['batch', 'atoms', 'embedding'], TensorType['batch', 'atoms', 'atoms']],
                  Tuple[TensorType['batch', 'atoms', 'embedding'], List[TensorType['batch', 'atoms', 'embedding']]],
                  Tuple[TensorType['batch', 'atoms', 'embedding'],
                        TensorType['batch', 'atoms', 'atoms'],
                        List[TensorType['batch', 'atoms', 'embedding']]]]:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for reaction cls or training tricks like MLM.
        Hydrogens should be coded from 2 (means no neighbors) to max neighbors + 2.
        Hydrogens equal to 1 reserved for training tricks like MLM. Use 0 for cls.

        :param need_embedding: return atoms embeddings
        :param need_weights: return attention weights
        :param averaged_weights: return averaged attentions from each layer, otherwise only last layer
        :param intermediate_embeddings: return embedding of each layer including initial weights but last
        """
        if not need_weights:
            assert not averaged_weights, 'averaging without need_weights'
            assert need_embedding, 'at least weights or embeddings should be returned'
        elif intermediate_embeddings:
            assert need_embedding, 'need_embedding should be active for intermediate_embeddings option'

        atoms, hydrogens, distances = batch

        gf = self.gaussian(atoms, distances)  # BxNxNxK
        d_mask = self.spatial_encoder(gf).masked_fill_(isinf(distances).unsqueeze_(-1), -inf)  # BxNxNxK > BxNxNxH
        d_mask = d_mask.permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN

        x = self.atoms_encoder(atoms)
        if self.implicit:  # cls token in hydrogens coded by 0
            x = x + self.hydrogens_encoder(hydrogens)
        if self.perturbation:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)

        if intermediate_embeddings:
            embeddings = [x]

        if averaged_weights:  # average attention weights from each layer
            w = []
            for lr in self.layers[:-1]:  # noqa
                x, a = lr(x, d_mask, need_weights=True)
                w.append(a)
                if intermediate_embeddings:
                    embeddings.append(x)  # noqa
            x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=True)
            w.append(a)
            w = stack(w, dim=-1).mean(-1)
            if need_embedding:
                if self.post_norm:
                    x = self.norm(x)
                if intermediate_embeddings:
                    return x, w, embeddings
                return x, w
            return w

        for lr in self.layers[:-1]:  # noqa
            x, _ = lr(x, d_mask)
            if intermediate_embeddings:
                embeddings.append(x)  # noqa
        x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if self.post_norm:
                x = self.norm(x)
            if intermediate_embeddings:
                if need_weights:
                    return x, a, embeddings
                return x, embeddings
            elif need_weights:
                return x, a
            return x
        return a


class GaussianLayer(Module):
    """
    x = a * d + b
    g = exp(-.5 * ((x - u) ** 2 / s ** 2)) / (s * sqrt(2 * pi))
    """

    def __init__(self, nkernel: int, posinf: float = 10., eps: float = 1e-5):
        super().__init__()
        self.nkernel = nkernel
        self.posinf = posinf
        self.eps = eps
        self.mu = Parameter(empty(nkernel))
        self.sigma = Parameter(empty(nkernel))
        self.a = Parameter(empty(121, 121))
        self.b = Parameter(empty(121, 121))
        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.mu, 0, 3)
        uniform_(self.sigma, 1, 3)
        constant_(self.a, 1)
        constant_(self.b, 0)

    def forward(self, atoms, distances):
        a = self.a[atoms.unsqueeze(-1), atoms.unsqueeze(1)]  # [BxNx1, Bx1xN] > BxNxN
        b = self.b[atoms.unsqueeze(-1), atoms.unsqueeze(1)]
        # exp(-inf) > nan
        d = nan_to_num(distances, posinf=self.posinf)
        # BxNxN > BxNxNx1 > BxNxNxK
        x = (a * d + b).unsqueeze(-1).expand(-1, -1, -1, self.nkernel)
        return exp(-.5 * square((x - self.mu) / self.sigma)) / ((self.sigma.abs() + self.eps) * sqrt(2 * pi))


class SigmoidLayer(Module):
    def __init__(self, nkernel: int = 128, d_model: int = 1024, dropout: float = .1, layer_norm_eps: float = 1e-5):
        super().__init__()
        # atom-type-pair set of linear layers
        self.a = Parameter(empty(121, 121, nkernel))
        self.b = Parameter(empty(121, 121, nkernel))

        self.layer_norm = LayerNorm(nkernel, layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.head = Linear(nkernel, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.a, -1, 1)
        uniform_(self.b, 0, 5)

    def forward(self, atoms, distances):
        a = self.a[atoms.unsqueeze(-1), atoms.unsqueeze(1)]  # [BxNx1, Bx1xN] > BxNxNxK
        b = self.b[atoms.unsqueeze(-1), atoms.unsqueeze(1)]

        # BxNxN > BxNxNxK
        x = a * (distances + b)
        return self.head(self.dropout(sigmoid(self.layer_norm(x))))


class EncoderLayer(Module):
    r"""EncoderLayer based on torch.nn.TransformerEncoderLayer, but batch always first and returns also attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer. Default: GELU.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise, it's done after.
            Default: ``False`` (after).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5,
                 norm_first: bool = False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()
        self.norm_first = norm_first

    def forward(self, x: Tensor, attn_mask: Tensor, *,
                need_embedding: bool = True, need_weights: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        tx = x.transpose(1, 0)  # switch Batch and Sequence. torch-1.8 compatible.
        nx = self.norm1(tx) if self.norm_first else tx  # pre-norm or post-norm
        e, a = self.self_attn(nx, nx, nx, attn_mask=attn_mask, need_weights=need_weights)

        if need_embedding:
            x = (tx + self.dropout1(e)).transpose(1, 0)  # switch Sequence and Batch back
            if self.norm_first:
                return x + self._ff(self.norm2(x)), a
            # else: post-norm
            x = self.norm1(x)
            return self.norm2(x + self._ff(x)), a
        return None, a

    def _ff(self, x):
        return self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))
        
'''