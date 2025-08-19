from torch import nn


def create_mapper(in_dim, out_dim, k):
    modules = [nn.Linear(in_dim, out_dim),
               nn.GELU(),
               nn.Linear(out_dim, out_dim * k),
               nn.Unflatten(-1, (k, out_dim))]
    return nn.Sequential(*modules)