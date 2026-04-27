import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Applies scale (gamma) and shift (beta) derived from a conditioning signal.
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film_scale = nn.Linear(cond_dim, out_dim)
        self.film_shift = nn.Linear(cond_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        gamma = self.film_scale(cond)
        beta = self.film_shift(cond)
        return self.act(h * (1 + gamma) + beta)

# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        t: (B,) integer timesteps.
        dim: Embedding dimension.

    Returns:
        (B, dim) embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class LaneDenoiser(nn.Module):
    """MLP denoiser with FiLM conditioning for lane geometry diffusion.

    Args:
        geom_dim: Flattened geometry dimension (K * 2).
        t_dim: Timestep embedding dimension.
        cond_dim: Behavioral embedding dimension.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        geom_dim: int = 32,
        t_dim: int = 64,
        cond_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.geom_dim = geom_dim
        self.t_dim = t_dim

        input_dim = geom_dim + t_dim

        self.layer1 = FiLMLayer(input_dim, hidden_dim, cond_dim)
        self.layer2 = FiLMLayer(hidden_dim, hidden_dim, cond_dim)
        self.layer3 = FiLMLayer(hidden_dim, hidden_dim // 2, cond_dim)
        self.out = nn.Linear(hidden_dim // 2, geom_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy geometry.

        Args:
            x_t: (B, geom_dim) noisy flattened geometry.
            t: (B,) integer timesteps.
            cond: (B, cond_dim) behavioral embedding.

        Returns:
            (B, geom_dim) predicted noise.
        """
        t_emb = sinusoidal_embedding(t, self.t_dim)
        inp = torch.cat([x_t, t_emb], dim=-1)
        h = self.layer1(inp, cond)
        h = self.layer2(h, cond)
        h = self.layer3(h, cond)
        return self.out(h)

# ---------------------------------------------------------------------------
# DDPM schedules and sampling
# ---------------------------------------------------------------------------

class DDPMSchedule:
    """Linear beta schedule for DDPM."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.1, device: str = "cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to x_0 at timestep t.

        Returns:
            (x_t, noise) — noisy sample and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

# ---------------------------------------------------------------------------
# Diffusion trainer
# ---------------------------------------------------------------------------

class LaneDiffusionTrainer:
    """Train a LaneDenoiser with DDPM on lane geometries.

    Supports two modes:
    - Stage 1 (unconditional pretraining): train on geometry alone with
      zero conditioning vectors.
    - Stage 2 (conditional fine-tuning): train with behavioral embeddings
      from the lane encoder, optionally starting from retrieved warm starts.
    """

    def __init__(
        self,
        model: LaneDenoiser,
        schedule: DDPMSchedule,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.schedule = schedule.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_epoch(
        self,
        geometries: torch.Tensor,
        cond_embeddings: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> float:
        """Train one epoch with MSE noise prediction loss.

        Args:
            geometries: (N, geom_dim) flattened lane geometries.
            cond_embeddings: (N, cond_dim) behavioral embeddings.
                If None, uses zeros (unconditional).

        Returns:
            Mean loss for the epoch.
        """
        self.model.train()
        N = geometries.shape[0]

        if cond_embeddings is None:
            cond_embeddings = torch.zeros(N, 128, device=self.device)

        geometries = geometries.to(self.device)
        cond_embeddings = cond_embeddings.to(self.device)

        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            x_0 = geometries[idx]
            cond = cond_embeddings[idx]
            B = x_0.shape[0]

            # Sample random timesteps
            t = torch.randint(0, self.schedule.T, (B,), device=self.device)

            # Forward diffusion
            x_t, noise = self.schedule.q_sample(x_0, t)

            # Predict noise
            pred_noise = self.model(x_t, t, cond)
            loss = nn.functional.mse_loss(pred_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        n_samples: int = 1,
        warm_start: Optional[torch.Tensor] = None,
        warm_start_t: int = 50,
    ) -> torch.Tensor:
        """Generate lane geometries via reverse diffusion.

        Args:
            cond: (cond_dim,) or (n_samples, cond_dim) conditioning embedding.
            n_samples: Number of samples to generate.
            warm_start: (geom_dim,) or (n_samples, geom_dim) initial geometry.
            warm_start_t: Timestep to start from when using warm start.

        Returns:
            (n_samples, geom_dim) generated geometries.
        """
        self.model.eval()

        if cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(n_samples, -1)
        cond = cond.to(self.device)

        if warm_start is not None:
            if warm_start.dim() == 1:
                warm_start = warm_start.unsqueeze(0).expand(n_samples, -1)
            warm_start = warm_start.to(self.device)
            t_tensor = torch.full((n_samples,), warm_start_t,
                                  device=self.device, dtype=torch.long)
            x_t, _ = self.schedule.q_sample(warm_start, t_tensor)
            start_t = warm_start_t
        else:
            geom_dim = self.model.geom_dim
            x_t = torch.randn(n_samples, geom_dim, device=self.device)
            start_t = self.schedule.T - 1

        # Reverse diffusion
        for t_val in reversed(range(0, start_t + 1)):
            t = torch.full((n_samples,), t_val, device=self.device, dtype=torch.long)

            pred_noise = self.model(x_t, t, cond)

            alpha = self.schedule.alphas[t_val]
            beta = self.schedule.betas[t_val]
            coeff = beta / self.schedule.sqrt_one_minus_alpha_bar[t_val]
            mean = (x_t - coeff * pred_noise) / torch.sqrt(alpha)

            if t_val > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta) * noise
            else:
                x_t = mean

        return x_t

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "schedule_T": self.schedule.T,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
