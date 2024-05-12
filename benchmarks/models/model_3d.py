import torch
from itertools import repeat
from torch import Tensor, cat, long, float32, zeros_like, exp, zeros, ones
from torch.nn import Module, Linear, Parameter, MSELoss

from torch.nn.functional import l1_loss
from torch.nn.modules.loss import _Loss
from torchtyping import TensorType
from typing import Optional

class Model3D(torch.nn.Module):
    def __init__(self, model_factory, hidden_dim, out_dim, device, unique_variables=1):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_factory() for _ in range(unique_variables)])
        #self.linear = torch.nn.Linear(hidden_dim * unique_variables, out_dim)#to symbolic
        self.linear = Symbolic(hidden_dim * unique_variables, 1, bias = False)  #(hidden_dim * unique_variables, out_dim)#to symbolic
        self.device = device

    def forward(self, batched_data):
        outs = []
        for model, data in zip(self.models, batched_data):
            data = data.to(self.device)
            if model.__class__.__name__ in ['ChytorchConformer','ChytorchDiscrete','ChytorchRotary']:
                z, hgs, pos, bat = data.x[:, 0]+1, data.x[:, 4], data.pos, data.batch
                out = model(z, hgs, pos, bat)
            else:
                z, pos, bat = data.x[:, 0], data.pos, data.batch
                out = model(z, pos, bat)
                if model.__class__.__name__ == 'LEFTNet':
                    out = out[0]
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