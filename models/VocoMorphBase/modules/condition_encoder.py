import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.num_effects = len(config["effects"])
        embedding_dim = config["embedding_dim"]
        self.embedding = nn.Embedding(self.num_effects, embedding_dim)
        assert embedding_dim % 2 == 0, "Embedding dim should be a multiple of 2"
        self.film_layer = nn.Linear(embedding_dim, 2)

    def forward(self, effect_id):
        assert effect_id.dim() == 2 and effect_id.shape[1] == 1, "effect_id must have shape (B, 1)"

        # shape (B, E)
        embed = self.embedding(effect_id).squeeze(1)

        # shape (B, C)
        gamma, beta = self.film_layer(embed).chunk(2, dim=-1)

        # shape (B, C, 1)
        gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)  
        return gamma, beta
