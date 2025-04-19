import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.
    Frequency-Selective Modulation:

    Pros:
    - Allows for different scaling and shifting of different frequency bands, enabling basic equalization-like effects and more nuanced timbre modifications.

    Cons:
    - linear and static across time. It cannot capture dynamic effects like tremolo or chorus.
    - not sufficient for highly non-linear effects like distortion or complex spectral manipulations needed for robotic voices.
    """

    def __init__(self, config):
        super().__init__()
        embedding_dim = config["embedding_dim"]
        num_frequencies = config["n_fft"] // 2 + 1
        self.scale_fc = nn.Linear(embedding_dim, num_frequencies)
        self.shift_fc = nn.Linear(embedding_dim, num_frequencies)

    def forward(self, x, embedding):
        # x shape: (B, C, F, T)
        B = x.shape[0]
        assert x.ndim == 4, f"Unexpected input dimensions in FiLM {x.shape}"

        # affine transformation: output = x * scale + shift.
        scale = self.scale_fc(embedding)
        shift = self.shift_fc(embedding)

        # broadcast across channels, freq, time
        scale = scale.view(B, 1, -1, 1)
        shift = shift.view(B, 1, -1, 1)

        return x * scale + shift
