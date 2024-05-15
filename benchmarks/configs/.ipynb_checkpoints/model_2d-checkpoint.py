from dataclasses import dataclass


@dataclass
class GIN:
    num_layers: int = 6
    virtual_node: bool = False


@dataclass
class GPS:
    num_layers: int = 3
    walk_length: int = 20
    num_heads: int = 4


@dataclass
class ChemProp:
    num_layers: int = 6

@dataclass
class Chytorch2D:
    max_neighbors: int = 80
    max_distance: int = 83
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 8
    dim_feedforward: int = 128
    shared_weights: bool = False
    dropout: float = 0.1
    norm_first: bool = True
    post_norm: bool = True
    zero_bias: bool = True


@dataclass
class Model2D:
    model: str = 'GIN'
    gin: GIN = GIN()
    gps: GPS = GPS()
    chemprop: ChemProp = ChemProp()
    chytorch2d: Chytorch2D = Chytorch2D()

