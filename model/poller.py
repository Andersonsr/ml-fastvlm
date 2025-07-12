from torch import nn


class Poller(nn.Module):
    def __init__(self, queries, input_dim, hidden_dim):
        super(Poller, self).__init__()

