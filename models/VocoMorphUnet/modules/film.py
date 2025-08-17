import mlx.core as mx
import mlx.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    This layer modulates a feature map (e.g., from a convolutional layer)
    using learned scaling (gamma) and shifting (beta) parameters that are
    conditioned on a given embedding.
    """

    def __init__(self, feature_channels: int, embedding_dim: int):
        super().__init__()
        self.gamma_generator = nn.Linear(embedding_dim, feature_channels)
        self.beta_generator = nn.Linear(embedding_dim, feature_channels)

    def __call__(self, feature_map: mx.array, embedding: mx.array):
        """
        Args:
        - feature_map (mx.array): Input feature map with shape (B, T, C).
        - embedding (mx.array): Conditioning embedding with shape (B, embedding_dim).
        Returns:
            mx.array: The modulated feature map.
        """
        # shape (B, C).
        gamma = self.gamma_generator(embedding)
        beta = self.beta_generator(embedding)

        # the input feature_map has shape (B, L, C).
        # we need to reshape gamma and beta to (B, 1, C) to enable broadcasting for element-wise multiplication and addition.
        gamma = mx.expand_dims(gamma, axis=1)
        beta = mx.expand_dims(beta, axis=1)

        modulated_feature_map = gamma * feature_map + beta
        return modulated_feature_map
