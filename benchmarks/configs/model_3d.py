from dataclasses import dataclass


@dataclass
class SchNet:
    hidden_dim: int = 128
    num_filters: int = 5
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: int = 10
    readout: str = 'mean'
    dipole: bool = False

@dataclass
class ChytorchDiscrete:
    max_distance: int = 83
    d_model: int = 128
    nhead: int = 16
    num_layers: int = 8
    dim_feedforward: int = 512
    shared_weights: bool = False
    dropout: float = 0.1
    norm_first: bool = True
    post_norm: bool = True
    zero_bias: bool = True
    
@dataclass
class ChytorchConformer:
    nkernel: int = 128
    shared_layers: bool = True
    d_model: int = 128
    nhead: int = 16
    num_layers: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    norm_first: bool = True
    post_norm: bool = True

@dataclass
class ChytorchRotary:
    d_model: int = 128
    nhead: int = 16
    num_layers: int = 8
    dim_feedforward: int = 512
    shared_weights: bool = False
    dropout: float = 0.1
    norm_first: bool = False
    post_norm: bool = False


@dataclass
class DimeNet:
    hidden_channels: int = 128
    out_channels: int = 128
    num_blocks: int = 6
    num_bilinear: int = 8
    num_spherical: int = 7
    num_radial: int = 6


@dataclass
class DimeNetPlusPlus:
    hidden_channels: int = 128
    out_channels: int = 128
    num_blocks: int = 4
    int_emb_size: int = 64
    basis_emb_size: int = 8
    out_emb_channels: int = 256
    num_spherical: int = 7
    num_radial: int = 6


@dataclass
class GemNet:
    num_spherical: int = 7
    num_radial: int = 6
    num_blocks: int = 4
    emb_size_atom: int = 128
    emb_size_edge: int = 128
    emb_size_trip: int = 64
    emb_size_rbf: int = 16
    emb_size_cbf: int = 16
    emb_size_bil_trip: int = 64
    num_before_skip: int = 1
    num_after_skip: int = 1
    num_concat: int = 1
    num_atoms: int = 1
    num_atom: int = 2
    bond_feat_dim: int = 0  # unused_argument


@dataclass
class ClofNet:
    cutoff: int = 6.5
    num_layers: int = 6
    hidden_channels: int = 128
    num_radial: int = 32


@dataclass
class LEFTNet:
    cutoff: int = 6.5
    num_layers: int = 6
    hidden_channels: int = 128
    num_radial: int = 32


@dataclass
class PaiNN:
    hidden_dim: int = 128
    num_interactions: int = 6  # 3
    num_rbf: int = 64  # 20
    cutoff: float = 12.0  # 5.0
    readout: str = 'add'  # 'add' or 'mean'
    # activation: Optional[Callable] = F.silu
    shared_interactions: bool = False
    shared_filters: bool = False


@dataclass
class Model3D:
    model: str = 'PaiNN'
    augmentation: bool = True

    schnet: SchNet = SchNet()
    dimenet: DimeNet = DimeNet()
    dimenetplusplus: DimeNetPlusPlus = DimeNetPlusPlus()
    gemnet: GemNet = GemNet()
    painn: PaiNN = PaiNN()
    clofnet: ClofNet = ClofNet()
    leftnet: LEFTNet = LEFTNet()
    chytorch_discrete: ChytorchDiscrete = ChytorchDiscrete()
    chytorch_conformer: ChytorchConformer = ChytorchConformer()
    chytorch_rotary: ChytorchRotary = ChytorchRotary()

