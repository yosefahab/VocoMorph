import torch.nn as nn


class FiLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        self.gamma = nn.Linear(input_dim, output_dim)
        self.beta = nn.Linear(input_dim, output_dim)

    def forward(self, embedding):
        scale = self.gamma(embedding)
        shift = self.beta(embedding)

        return scale, shift
