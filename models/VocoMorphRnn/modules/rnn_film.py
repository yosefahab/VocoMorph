import torch
import torch.nn as nn


class RNNFiLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config["embedding_dim"]
        num_frequencies = config["n_fft"] // 2 + 1
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]

        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        # Time-varying scale and shift per frequency
        self.scale_fc = nn.Linear(hidden_size, num_frequencies)
        self.shift_fc = nn.Linear(hidden_size, num_frequencies)

    def forward(self, x, embedding):
        # x shape: (B, C, F, T) -> (B * C * F, T, 1) to process time
        B, _, _, T = x.shape
        # Repeat embedding along time
        input_rnn = embedding.unsqueeze(1).repeat(1, T, 1)

        # Initialize hidden state (optional, can be zeros)
        h0 = torch.zeros(
            self.rnn.num_layers,
            B,
            self.rnn.hidden_size,
            device=embedding.device,
        )

        # RNN processes the embedding over time
        rnn_out, _ = self.rnn(input_rnn, h0)  # rnn_out: (B, T, rnn_hidden_size)

        # Predict time-varying scale and shift for each frequency
        # (B, 1, 1, T, F) -> (B, 1, F, T) after transpose
        scale = self.scale_fc(rnn_out).unsqueeze(1).unsqueeze(1)
        # (B, 1, 1, T, F) -> (B, 1, F, T) after transpose
        shift = self.shift_fc(rnn_out).unsqueeze(1).unsqueeze(1)

        # Apply modulation
        return x * scale.transpose(-1, -2) + shift.transpose(-1, -2)
