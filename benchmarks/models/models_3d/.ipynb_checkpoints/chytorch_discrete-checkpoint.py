from torch.nn import GELU, Module, ModuleList, LayerNorm
from torchtyping import TensorType
from warnings import warn
from .embedding import EmbeddingBag
from ..lora import Embedding
from ..transformer import EncoderLayer
#from ...utils.data import MoleculeDataBatch
import torch
import numpy as np

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
            max_tokens: int = 121+10,
            lora_r: int = 0,
            lora_alpha: float = 1.,
            lora_dropout: float = 0.):
        #super(ChytorchDiscrete, self).__init__()
        super().__init__()
        
        self.embedding = EmbeddingBag(max_neighbors, d_model, perturbation, max_tokens, lora_r, lora_alpha)

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
        z: torch.Tensor,
        hgs: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor = None, 
        tokens: torch.Tensor = None):
        assert z.dim() == 1 and z.dtype == torch.long

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
        batch = torch.zeros_like(z) if batch is None else batch
        
        N = max(batch.bincount())
        num_batches = max(batch)+1
        batched_z = torch.zeros(num_batches, N, dtype=torch.int32, device=z.device)
        batched_hgs =torch.zeros(num_batches, N, dtype=torch.int32, device=z.device)
        batched_dist = torch.zeros(num_batches, N+1, N+1, dtype=torch.int32, device=z.device)

        short_cutoff =.9
        long_cutoff = 5.
        precision = .05
        _bins = np.arange(short_cutoff - 3 * precision, long_cutoff, precision)
        _bins[:3] = [-1, 0, .01]  # trick for self-loop coding
        self.max_distance = len(_bins) - 2  # param for MoleculeEncoder
        
        # Populate the batched_tensor
        for i in range(num_batches):
            indices = (batch == i).nonzero(as_tuple=True)[0]
            batched_z[i, :len(indices)] = z[indices] + 2
            batched_hgs[i, :len(indices)] = hgs[indices] + 2
            pos_i = pos[indices,:]
            diff = pos_i[None, :, :] - pos_i[:, None, :]  # NxNx3
            dist = (diff ** 2).sum(dim=-1).sqrt()  # BxNxN

            dist = np.digitize(dist.cpu(), _bins)
            dist = torch.tensor(dist, dtype=torch.int32, device = z.device)
            
            tmp = torch.ones((len(indices)+1, len(indices)+1), dtype=torch.int32)
            tmp[1:, 1:] = dist
            
            batched_dist[i, :len(indices)+1, :len(indices)+1] = tmp
            diagonal = torch.diag(batched_dist[i, :, :])
            zero_diagonal_indices = torch.where(diagonal == 0)[0]
            batched_dist[i, zero_diagonal_indices, zero_diagonal_indices] = 1

        #add 1 cls
        atoms = torch.ones(batched_z.shape[0], batched_z.shape[1] + 1, dtype=torch.int32, device=z.device)
        neighbors = torch.zeros(batched_hgs.shape[0], batched_hgs.shape[1] + 1, dtype=torch.int32, device=z.device)
        #sum3 from default MAECEL
        atoms[:,1:] = batched_z.int()
        neighbors[:,1:] = batched_hgs.int()
        
        if tokens is None:
            pass
        else:
            atoms[:,0]=tokens+121
            
        #dist = np.digitize(batched_dist.cpu(), _bins)
        #tmp = torch.ones((num_batches, atoms.shape[1], atoms.shape[1]), dtype=torch.int32)
        #tmp[:,1:, 1:] = dist
        distances = batched_dist.to(z.device)#torch.tensor(batched_dist, dtype=torch.int32, device = z.device)
        #distances = torch.tensor(tmp, dtype=torch.int32, device = z.device)

        # cls token in neighbors coded by 0
        x = self.embedding(atoms, neighbors)

        for lr, d in zip(self.layers, self.distance_encoders):
            if d is not None:
                d_mask = d(distances).permute(0, 3, 1, 2)  # BxNxNxH > BxHxNxN
            # else: reuse previously calculated mask
            x, _ = lr(x, d_mask)  # noqa

        if self.post_norm:
            return self.norm(x)[:,0]
        return x[:,0]

    def merge_lora(self):
        """
        Transform LoRA layers to normal
        """
        self.embedding.merge_lora()
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

    @property
    def atoms_encoder(self):
        warn('neighbors_encoder moved to embedding submodule in chytorch 1.61', DeprecationWarning)
        return self.embedding.atoms_encoder

    @property
    def neighbors_encoder(self):
        warn('neighbors_encoder moved to embedding submodule in chytorch 1.61', DeprecationWarning)
        return self.embedding.neighbors_encoder
        
        
def _update(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'centrality_encoder.weight' in state_dict:
        warn('fixed chytorch<1.37 checkpoint', DeprecationWarning)
        state_dict[prefix + 'neighbors_encoder.weight'] = state_dict.pop(prefix + 'centrality_encoder.weight')
        state_dict[prefix + 'distance_encoder.weight'] = state_dict.pop(prefix + 'spatial_encoder.weight')
    if prefix + 'atoms_encoder.weight' in state_dict:
        warn('fixed chytorch<1.61 checkpoint', DeprecationWarning)
        state_dict[prefix + 'embedding.atoms_encoder.weight'] = state_dict.pop(prefix + 'atoms_encoder.weight')
        state_dict[prefix + 'embedding.neighbors_encoder.weight'] = state_dict.pop(prefix + 'neighbors_encoder.weight')