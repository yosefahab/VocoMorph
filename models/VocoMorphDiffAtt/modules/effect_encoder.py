import torch.nn as nn
import torch


class EffectEncoder(nn.Module):
    def __init__(self, num_effects, embedding_dim):
        super().__init__()
        self.num_effects = num_effects
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_effects + 1, self.embedding_dim)
        with torch.no_grad():
            self.embedding.weight[-1].fill_(0.0)

    def forward(self, effect_id):
        if effect_id.ndim == 1:
            effect_id = effect_id.unsqueeze(1)
        if effect_id.ndim == 2 and effect_id.shape[1] == 1:
            return self.embedding(effect_id).squeeze(1)
        flat = effect_id.view(-1)
        return self.embedding(flat).view(effect_id.shape[0], -1)
