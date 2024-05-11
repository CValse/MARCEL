import torch


class Model3D(torch.nn.Module):
    def __init__(self, model_factory, hidden_dim, out_dim, device, unique_variables=1):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_factory() for _ in range(unique_variables)])
        self.linear = torch.nn.Linear(hidden_dim * unique_variables, out_dim)#to symbolic
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
