from torch import nn


def create_mapper(in_dim, out_dim, k, neck):
    if neck is None:
        neck = out_dim

    modules = [nn.Linear(in_dim, neck),
               nn.GELU(),
               nn.Linear(neck, out_dim * k),
               nn.Unflatten(-1, (k, out_dim))]
    return nn.Sequential(*modules)

