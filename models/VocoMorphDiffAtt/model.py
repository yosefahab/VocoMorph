import torch.nn as nn

from .modules.decoder import Decoder
from .modules.effect_encoder import EffectEncoder
from .modules.encoder import Encoder
from .modules.time_embedding import TimeEmbedding


class VocoMorphDiffAtt(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        embedding_dim = config["embedding_dim"]
        self.num_channels = config["num_channels"]
        encoder_filters = config["encoder_filters"]
        bottleneck_filters = config["bottleneck_filters"]
        decoder_filters = config["decoder_filters"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]
        attn_layers = config["attn_layers"]
        time_dim = config["time_dim"]

        self.effect_encoder = EffectEncoder(config["num_effects"], embedding_dim)
        self.time_embed = TimeEmbedding(time_dim)

        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_filters)):
            use_attn = i in attn_layers
            in_ch = self.num_channels if i == 0 else encoder_filters[i - 1]
            self.encoder_blocks.append(
                Encoder(
                    in_ch,
                    encoder_filters[i],
                    kernel_size,
                    padding,
                    embedding_dim,
                    apply_pool=True,
                    use_attn=use_attn,
                    time_dim=time_dim,
                )
            )

        self.bottleneck_blocks = nn.ModuleList()
        for i in range(len(bottleneck_filters)):
            in_ch = encoder_filters[-1] if i == 0 else bottleneck_filters[i - 1]
            use_attn = ("bottleneck" in attn_layers) or (i in attn_layers)
            self.bottleneck_blocks.append(
                Encoder(
                    in_ch,
                    bottleneck_filters[i],
                    kernel_size,
                    padding,
                    embedding_dim,
                    apply_pool=False,
                    use_attn=use_attn,
                    time_dim=time_dim,
                )
            )

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_filters)):
            use_attn = i in attn_layers
            in_ch = bottleneck_filters[-1] if i == 0 else decoder_filters[i - 1]
            skip_ch = encoder_filters[len(encoder_filters) - 1 - i]
            self.decoder_blocks.append(
                Decoder(
                    in_ch,
                    decoder_filters[i],
                    kernel_size,
                    padding,
                    embedding_dim,
                    skip_ch,
                    use_attn=use_attn,
                    time_dim=time_dim,
                )
            )

        self.final_conv = nn.Conv1d(
            decoder_filters[-1], self.num_channels, kernel_size=1
        )
        self.final_activation = nn.Tanh()

    def forward(self, x):
        effect_id, (noisy_chunk, t) = x
        # encode t to sinusoidal embedding
        # condition the UNet on effect_id and t

        effect_embedding = self.effect_encoder(effect_id)
        time_embedding = self.time_embed(t)
        x = noisy_chunk

        skip_connections = []
        for encoder in self.encoder_blocks:
            x, skip = encoder(x, effect_embedding, time_embedding)
            skip_connections.append(skip)
        for bottleneck in self.bottleneck_blocks:
            x, _ = bottleneck(x, effect_embedding, time_embedding)
        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(x, skip, effect_embedding, time_embedding)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x


# training_loop.py skeleton
import torch
import torch.nn.functional as F
from torch import optim

from .modules.diffusion_utils import get_alphas, q_sample


def p_losses(
    model,
    x_start,
    effect_id,
    t,
    noise,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    null_effect_id,
    device,
):
    x_noisy = q_sample(
        x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise
    )
    logits = model(x_noisy, effect_id, t)
    target = noise
    loss = F.mse_loss(logits, target)
    return loss


def train_one_epoch(
    model, dataloader, optimizer, betas, device, p_uncond=0.1, null_id=None
):
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_alphas(betas)
    timesteps = betas.shape[0]
    model.train()
    for batch in dataloader:
        audio, effect_id = batch
        audio = audio.to(device)
        effect_id = effect_id.to(device)
        b = audio.shape[0]
        t = torch.randint(0, timesteps, (b,), device=device).long()
        noise = torch.randn_like(audio)
        mask = torch.rand(b, device=device) < p_uncond
        effect_id_train = effect_id.clone()
        effect_id_train[mask] = null_id
        loss = p_losses(
            model,
            audio,
            effect_id_train,
            t,
            noise,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            null_id,
            device,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# inference.py guidance sampler
import numpy as np
import torch


@torch.no_grad()
def sample_with_guidance(
    model, x_init, effect_id, null_id, betas, guidance_scale=1.5, steps=50, device="cpu"
):
    model.eval()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    timesteps = betas.shape[0]
    seq = np.linspace(0, timesteps - 1, steps, dtype=int)[::-1]
    x = x_init
    for i in seq:
        t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
        e_t_cond = model(x, effect_id, t)
        e_t_uncond = model(x, torch.full_like(effect_id, null_id), t)
        e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta_t = betas[i]
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
        mean = coef1 * (x - coef2 * e_t)
        if i > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean
    return x
