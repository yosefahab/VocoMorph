import mlx.core as mx
import mlx.nn as nn


class EffectEncoder(nn.Module):
    """
    Encodes a one-hot `effect_id` into a dense embedding vector.
    """

    def __init__(self, num_effects: int, embedding_dim: int):
        super().__init__()
        self.num_effects = num_effects
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_effects, self.embedding_dim)

    def __call__(self, effect_id: mx.array):
        # ensure the effect_id has the correct shape for embedding lookup.
        if effect_id.ndim == 1:
            effect_id = mx.expand_dims(effect_id, axis=1)

        # squeeze to remove the extra dimension added for the embedding lookup.
        return self.embedding(effect_id).squeeze(1)
