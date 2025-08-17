import torch.nn as nn


class EffectEncoder(nn.Module):
    def __init__(self, num_effects, embedding_dim):
        super().__init__()
        self.num_effects = num_effects
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_effects, self.embedding_dim)

    def forward(self, effect_id):
        if effect_id.ndim == 1:
            effect_id = effect_id.unsqueeze(1)
        assert effect_id.ndim == 2 and effect_id.shape[1] == 1, (
            "effect_id must have shape (B, 1)"
        )
        return self.embedding(effect_id).squeeze(1)
