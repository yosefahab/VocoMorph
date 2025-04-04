import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.num_effects = len(config["effects"])
        self.embedding_dim = config["embedding_dim"]
        self.embedding = nn.Embedding(self.num_effects, self.embedding_dim)

    def forward(self, effect_id):
        assert (
            effect_id.dim() == 2 and effect_id.shape[1] == 1
        ), "effect_id must have shape (B, 1)"
        embed = self.embedding(effect_id).squeeze(1)
        return embed  # Shape (B, embedding_dim)
