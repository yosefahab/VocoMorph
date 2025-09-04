import torch.nn as nn


class EffectCrossAttention(nn.Module):
    def __init__(self, feature_channels, embedding_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_channels = feature_channels

        self.embedding_proj = nn.Linear(embedding_dim, feature_channels)
        self.layernorm = nn.LayerNorm(feature_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, features, effect_embedding):
        """
        Args:
            features: (B, C, T)
            effect_embedding: (B, embedding_dim)
        Returns:
            Modulated features: (B, C, T)
        """
        T = features.shape[-1]

        # project embedding to feature_channels
        effect_proj = self.embedding_proj(effect_embedding)  # (B, C)
        # repeat along temporal dimension
        effect_proj = effect_proj.unsqueeze(1).repeat(1, T, 1)  # (B, T, C)

        # features: (B, C, T) -> (B, T, C)
        x_seq = features.permute(0, 2, 1)
        x_norm = self.layernorm(x_seq)

        # cross-attention: query=features, key=value=embedding
        out, _ = self.attn(query=x_norm, key=effect_proj, value=effect_proj)

        # back to (B, C, T)
        return out.permute(0, 2, 1)
