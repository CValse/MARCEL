import torch

from torch.nn import Linear, ReLU, Sequential, TransformerEncoderLayer, TransformerEncoder, Module, Parameter, MSELoss
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_mean_pool

from itertools import repeat
from torch import Tensor, cat, long, float32, zeros_like, exp, zeros, ones

from torch.nn.functional import l1_loss
from torch.nn.modules.loss import _Loss
from torchtyping import TensorType
from typing import Optional
from collections import OrderedDict


class SumPooling(torch.nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x, molecule_idx):
        x = global_add_pool(x, molecule_idx)
        return x


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, molecule_idx):
        x = global_mean_pool(x, molecule_idx)
        return x


class DeepSets(torch.nn.Module):
    def __init__(self, hidden_dim, reduce='mean'):
        super(DeepSets, self).__init__()
        self.phi = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.reduce = reduce

    def forward(self, x, molecule_idx):
        x = self.phi(x)
        x = scatter(x, molecule_idx, dim=0, reduce=self.reduce)
        x = self.rho(x)
        return x


class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = Linear(hidden_dim, hidden_dim)
        self.phi = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

    def forward(self, x, batch):
        x = self.phi(x)

        attention_scores = self.attention(x)
        dot_product = torch.matmul(attention_scores, attention_scores.transpose(1, 0))
        # attention_weights = scatter_softmax(dot_product, batch, dim=0)

        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
        max_values = (dot_product * mask).max(dim=1, keepdim=True).values
        masked_dot_product = (dot_product - max_values) * mask
        attention_weights = masked_dot_product.exp() / (masked_dot_product.exp() * mask).sum(dim=1, keepdim=True)
        attention_weights = attention_weights * mask

        x_weighted = torch.matmul(attention_weights, x)
        x_aggregated = scatter(x_weighted, batch, dim=0, reduce='sum')

        x_encoded = self.rho(x_aggregated)
        return x_encoded


class TransformerPooling(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TransformerPooling, self).__init__()
        self.phi = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        transformer_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

    def forward(self, x, molecule_idx):
        x_transformed = self.phi(x)

        # Create a mask to prevent attention to padding tokens
        batch_size = len(molecule_idx.unique())
        max_set_size = molecule_idx.bincount().max()
        padding_mask = torch.zeros(batch_size, max_set_size, dtype=torch.bool)
        for batch_idx, count in enumerate(molecule_idx.bincount()):
            padding_mask[batch_idx, count:] = 1

        # Create a padded version of x_transformed to match the shape required by the Transformer encoder
        x_padded = torch.zeros(batch_size, max_set_size, x_transformed.size(1))
        batch_elements = (molecule_idx == torch.arange(batch_size).unsqueeze(1)).float()
        x_padded = torch.matmul(batch_elements, x_transformed)

        # Apply the Transformer encoder
        x_encoded_padded = self.transformer_encoder(x_padded, src_key_padding_mask=padding_mask)

        # Remove padding
        x_encoded = scatter(x_encoded_padded, molecule_idx, dim=0, reduce="sum")
        x_encoded = self.rho(x_encoded)

        return x_encoded


class Model4D(torch.nn.Module):
    def __init__(
            self, hidden_dim, out_dim,
            graph_model_factory, set_model_factory,
            device, unique_variables=1, multitask=False):
        super().__init__()
        self.graph_encoders = torch.nn.ModuleList(
            [graph_model_factory() for _ in range(unique_variables)])
        self.set_encoders = torch.nn.ModuleList(
            [set_model_factory() for _ in range(unique_variables)])
        #self.linear = torch.nn.Linear(hidden_dim * unique_variables, out_dim)
        self.linear = Symbolic(hidden_dim * unique_variables, 1, bias = False)  #(hidden_dim * unique_variables, out_dim)#to symbolic
        self.device = device
        self.multitask = multitask

    def forward(self, batched_data, molecule_indices):
        outs = []
        for graph_encoder, set_encoder, data, molecule_index in zip(
                self.graph_encoders, self.set_encoders, batched_data, molecule_indices):

            data = data.to(self.device)
            if graph_encoder.__class__.__name__ in ['ChytorchConformer','ChytorchDiscrete','ChytorchRotary']:
                if self.multitask:
                    z, hgs, pos, bat, tokens = data.x[:, 0]+1, data.x[:, 4], data.pos, data.batch, data.tokens
                    out = graph_encoder(z, hgs, pos, bat, tokens)
                else:
                    z, hgs, pos, bat = data.x[:, 0]+1, data.x[:, 4], data.pos, data.batch
                    out = graph_encoder(z, hgs, pos, bat)
            elif graph_encoder.__class__.__name__ in ['ChIRo']:
                #z, hgs, pos, bat = data.x[:, 0]+1, data.x[:, 4], data.pos, data.batch
                #from models_3d.chiro import get_local_structure_map
                LS_map, alpha_indices = get_local_structure_map(data.dihedral_angle_index)
                data = data.to(self.device)
                LS_map = LS_map.to(self.device)
                alpha_indices = alpha_indices.to(self.device)
                out = graph_encoder(data, LS_map, alpha_indices)
            else:
                z, pos, bat = data.x[:, 0], data.pos, data.batch
                out = graph_encoder(z, pos, bat)
                if graph_encoder.__class__.__name__ == 'LEFTNet':
                    out = out[0]

            out = set_encoder(out, molecule_index)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = self.linear(outs).squeeze(-1)
        return outs

def _ident(x: Tensor) -> Tensor:
    return x

def _exponent10(x: Tensor) -> Tensor:
    x = torch.clamp(x, min=-10, max=10)
    result = 10 ** x
    return result

def _neg_exponent10(x: Tensor) -> Tensor:
    x = torch.clamp(x, min=-10, max=10)
    result = 10 ** -x
    return result

def _scale10(x: Tensor) -> Tensor:
    return x * 10

def _scale100(x: Tensor) -> Tensor:
    return x * 100

def _scale1000(x: Tensor) -> Tensor:
    return x * 1000

class Symbolic(Module):
    def __init__(self, in_features: int, hidden: int = 1, bias: bool = False, *,
                 ident: bool = True, exponent10: bool = False, neg_exponent10: bool = False,
                 scale10: bool = False, scale100: bool = False, scale1000: bool = False):
        super().__init__()
        self.hidden = hidden
        self.ident = ident
        self.exponent10 = exponent10
        self.neg_exponent10 = neg_exponent10
        self.scale10 = scale10
        self.scale100 = scale100
        self.scale1000 = scale1000

        self.ops = ops = []
        if ident:
            ops.append(_ident)
        if exponent10:
            ops.append(_exponent10)
        if neg_exponent10:
            ops.append(_neg_exponent10)
        if scale10:
            ops.append(_scale10)
        if scale100:
            ops.append(_scale100)
        if scale1000:
            ops.append(_scale1000)

        self.head = Linear(in_features, hidden, bias)
        self.scaler = Linear(in_features, len(ops), False)

    def forward(self, x: Tensor) -> Tensor:
        s = self.scaler(x)  # BxO
        x = self.head(x)  # BxO or Bx1

        if self.hidden == 1:
            hid = repeat(x)
        else:
            hid = [h.unsqueeze(-1) for h in x.unbind(dim=1)]

        # O(Bx1) > BxO * BxO > B > B*1
        return (s * cat([op(h) for op, h in zip(self.ops, hid)], dim=1)).sum(1).unsqueeze(-1)


class GroupedScaledMAELoss(_Loss):
    """
    Mean Absolute Error (MAE) scaled to target value and averaged per group.

    Calculates internally moving average of target value. Rescales only targets out of 0-1 range.
    """
    def __init__(self, targets_coefficients: Tensor, eps: float = 1e-5):
        """
        :param targets_coefficients: weights of each target group
        """
        super().__init__()
        self.register_buffer('targets_coefficients', targets_coefficients)
        # start from 1 to avoid zero div
        # in limit gives the same, but for the first points average unbalanced
        self.register_buffer('mean_targets', zeros(len(targets_coefficients)))
        self.register_buffer('mean_counts', ones(len(targets_coefficients), dtype=long))
        self.eps = eps
        self._max_targets = len(targets_coefficients)

    def forward(self, input: Tensor, target: Tensor, idx: Tensor, qualifier: Optional[Tensor] = None) -> Tensor:
        """
        :param input: vector of predicted values
        :param target: vector of target values
        :param idx: vector of target group identifiers. should be in [0, len(targets_coefficients)-1] range
        :param qualifier: vector of qualifiers (-1, 0, 1)
        """
        cnt = idx.bincount(minlength=self._max_targets)
        if self.training:
            # calculate moving average of target values
            cnt0 = self.mean_counts.clone()  # keep old counts
            cnt1 = self.mean_counts.add_(cnt)  # update counts
            # rescale current average and add new data
            self.mean_targets.mul_(cnt0 / cnt1).add_(idx.bincount(target, self._max_targets) / cnt1)

        if qualifier is not None:
            # censored mask
            mask = (((qualifier >= 0) | (input >= target)) & ((qualifier <= 0) | (input <= target))).to(target)
            # uncensored count. eps to avoid infinity
            masked_cnt = idx.bincount(mask, self._max_targets) + self.eps
            scaler = (self.targets_coefficients / self.mean_targets.abs().clamp(1) / masked_cnt)[idx] * mask
        else:
            # (A1+...+An) / N * C = A1 * C/N + ... + An * C/N
            scaler = (self.targets_coefficients / self.mean_targets.abs().clamp(1) / cnt)[idx]

        return l1_loss(input, target, reduction='none') @ scaler

def get_local_structure_map(psi_indices):
    LS_dict = OrderedDict()
    LS_map = torch.zeros(psi_indices.shape[1], dtype = torch.long)
    v = 0
    for i, indices in enumerate(psi_indices.T):
        tupl = (int(indices[1]), int(indices[2]))
        if tupl not in LS_dict:
            LS_dict[tupl] = v
            v += 1
        LS_map[i] = LS_dict[tupl]

    alpha_indices = torch.zeros((2, len(LS_dict)), dtype = torch.long)
    for i, tupl in enumerate(LS_dict):
        alpha_indices[:,i] = torch.LongTensor(tupl)

    return LS_map, alpha_indices


if __name__ == '__main__':
    model = SelfAttentionPooling(hidden_dim=16)
    x = torch.randn(11, 16)
    batch = torch.tensor([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4])

    x_encoded = model(x, batch)

    x_encoded_manual_list = []
    batch_indices = batch.unique()

    for batch_idx in batch_indices:
        batch_elements = (batch == batch_idx).nonzero(as_tuple=True)[0]
        x_transformed_batch = model.phi(x[batch_elements])
        attention_scores_batch = model.attention(x_transformed_batch)
        dot_product_batch = torch.matmul(attention_scores_batch, attention_scores_batch.transpose(1, 0))

        softmax_batch = torch.exp(dot_product_batch) / torch.sum(torch.exp(dot_product_batch), dim=1, keepdim=True)
        x_weighted_batch = torch.matmul(softmax_batch, x_transformed_batch)
        x_aggregated_batch = x_weighted_batch.sum(dim=0, keepdim=True)
        x_encoded_manual_batch = model.rho(x_aggregated_batch)

        x_encoded_manual_list.append(x_encoded_manual_batch)

    x_encoded_manual = torch.cat(x_encoded_manual_list, dim=0)
    torch.testing.assert_close(x_encoded, x_encoded_manual, rtol=1e-5, atol=1e-5)
