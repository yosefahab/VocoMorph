import torch.nn as nn
import torch


# class FiLM(nn.Module):
#     """
#     Feature-wise Linear Modulation (FiLM) layer.
#     Modulates a feature map using gamma (scaling) and beta (shifting) parameters
#     generated from a conditioning embedding.
#     """
#
#     def __init__(self, feature_channels, embedding_dim):
#         super().__init__()
#         self.gamma_generator = nn.Linear(embedding_dim, feature_channels)
#         self.beta_generator = nn.Linear(embedding_dim, feature_channels)
#
#     def forward(self, feature_map, embedding):
#         """
#         Args:
#             feature_map (torch.Tensor): Input feature map (B, C, T).
#             embedding (torch.Tensor): Conditioning embedding (B, embedding_dim).
#         Returns:
#             torch.Tensor: Modulated feature map.
#         """
#         # (B, C)
#         gamma = self.gamma_generator(embedding)
#         beta = self.beta_generator(embedding)
#
#         # reshape gamma and beta to (B, C, 1) to enable broadcasting for element-wise multiplication/addition
#         gamma = gamma.unsqueeze(-1)
#         beta = beta.unsqueeze(-1)
#
#         modulated_feature_map = gamma * feature_map + beta
#         return modulated_feature_map


class FiLM(nn.Module):
    def __init__(self, feature_channels, embedding_dim):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * feature_channels),
        )

    def forward(self, feature_map, embedding):
        params = self.net(embedding)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * feature_map + beta
