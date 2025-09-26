import torch.nn as nn


class CrossAttention1D(nn.Module):
    def __init__(self, channels, cond_dim, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(cond_dim, channels)
        self.to_v = nn.Linear(cond_dim, channels)
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, cond):
        t = x.shape[-1]
        x_seq = x.permute(0, 2, 1)
        x_norm = self.norm(x_seq)
        q = self.to_q(x_norm)
        k = self.to_k(cond).unsqueeze(1)
        v = self.to_v(cond).unsqueeze(1)
        k = k.expand(-1, t, -1)
        v = v.expand(-1, t, -1)
        out, _ = self.mha(q, k, v)
        out = self.proj(out)
        out = out.permute(0, 2, 1)
        return x + out
