import torch
import torch.nn as nn


class SequenceAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        # x: (batch, channels, length)
        x_seq = x.permute(0, 2, 1)  # (batch, length, channels)
        x_seq = self.norm(x_seq)
        out, _ = self.attn(x_seq, x_seq, x_seq)
        return out.permute(0, 2, 1)  # back to (batch, channels, length)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, length)
        squeeze = x.mean(dim=2)  # (batch, channels)
        excite = torch.relu(self.fc1(squeeze))  # (batch, channels//reduction)
        excite = self.sigmoid(self.fc2(excite))  # (batch, channels)
        return x * excite.unsqueeze(-1)  # (batch, channels, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, use_sequence=True, use_channel=True, num_heads=4):
        super().__init__()
        self.sequence_attn = (
            SequenceAttention(channels, num_heads) if use_sequence else nn.Identity()
        )
        self.channel_attn = ChannelAttention(channels) if use_channel else nn.Identity()

    def forward(self, x):
        x = self.sequence_attn(x)
        x = self.channel_attn(x)
        return x
