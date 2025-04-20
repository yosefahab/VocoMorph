import torch
import torch.nn as nn


class RNNFiLM(nn.Module):
    """
    Time-varying scale and shift per frequency
    """

    def __init__(self, config):
        super().__init__()
        embedding_dim = config["embedding_dim"]
        num_frequencies = config["n_fft"] // 2 + 1
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]

        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.scale_fc = nn.Linear(hidden_size, num_frequencies)
        self.shift_fc = nn.Linear(hidden_size, num_frequencies)

    def forward(self, x, embedding):
        B, _, _, T = x.shape
        input_rnn = embedding.unsqueeze(1).repeat(1, T, 1)
        h0 = torch.zeros(
            self.rnn.num_layers, B, self.rnn.hidden_size, device=embedding.device
        )
        rnn_out, _ = self.rnn(input_rnn, h0)
        scale = self.scale_fc(rnn_out).unsqueeze(1).transpose(-1, -2)
        shift = self.shift_fc(rnn_out).unsqueeze(1).transpose(-1, -2)

        return x * scale + shift
