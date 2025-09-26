import torch
import torch.nn.functional as F
import math
import numpy as np


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


def q_sample(
    x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None
):
    if noise is None:
        noise = torch.randn_like(x_start)
    return (
        sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_start
        + sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
    )


def get_alphas(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
