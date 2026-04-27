import logging
from typing import Optional

import torch
import torch.nn as nn

from src.generation.diffusion import (
    DDPMSchedule,
    FiLMLayer,
    sinusoidal_embedding,
)

logger = logging.getLogger(__name__)

class RelationalEncoder(nn.Module):
    """Encode relational context into a fixed-size vector.

    Inputs:
        neighbor_geom (B, geom_dim=32): flattened canonical neighbor geometry.
        merge_point (B, 1): where along the lane the merge/diverge occurs.
        offset (B, 1): starting lateral distance from the neighbor.
        has_relation (B, 1): binary flag (1=relation present, 0=ignore).

    Output:
        (B, out_dim) relational embedding.
    """

    def __init__(
        self,
        geom_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 64,
    ):
        super().__init__()
        # neighbor_geom(32) + merge_point(1) + offset(1) + has_relation(1) = 35
        input_dim = geom_dim + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        neighbor_geom: torch.Tensor,
        merge_point: torch.Tensor,
        offset: torch.Tensor,
        has_relation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            neighbor_geom: (B, geom_dim) flattened neighbor geometry.
            merge_point: (B, 1) topology change location.
            offset: (B, 1) lateral distance.
            has_relation: (B, 1) binary mask.

        Returns:
            (B, out_dim) relational embedding.
        """
        x = torch.cat([neighbor_geom, merge_point, offset, has_relation], dim=-1)
        return self.net(x)

# ---------------------------------------------------------------------------
# Relational denoiser
# ---------------------------------------------------------------------------

class RelationalLaneDenoiser(nn.Module):
    """MLP denoiser with FiLM conditioning on behavioral + relational context.

    Same architecture as LaneDenoiser but with wider conditioning dimension
    (cond_dim + rel_dim) to incorporate neighbor geometry context.

    Args:
        geom_dim: Flattened geometry dimension (K * 2).
        t_dim: Timestep embedding dimension.
        cond_dim: Behavioral embedding dimension.
        rel_dim: Relational encoder output dimension.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        geom_dim: int = 32,
        t_dim: int = 64,
        cond_dim: int = 128,
        rel_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.geom_dim = geom_dim
        self.t_dim = t_dim

        input_dim = geom_dim + t_dim
        aug_cond_dim = cond_dim + rel_dim

        self.rel_encoder = RelationalEncoder(
            geom_dim=geom_dim,
            hidden_dim=rel_dim,
            out_dim=rel_dim,
        )

        self.layer1 = FiLMLayer(input_dim, hidden_dim, aug_cond_dim)
        self.layer2 = FiLMLayer(hidden_dim, hidden_dim, aug_cond_dim)
        self.layer3 = FiLMLayer(hidden_dim, hidden_dim // 2, aug_cond_dim)
        self.out = nn.Linear(hidden_dim // 2, geom_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        neighbor_geom: torch.Tensor,
        merge_point: torch.Tensor,
        offset: torch.Tensor,
        has_relation: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy geometry with relational context.

        Args:
            x_t: (B, geom_dim) noisy flattened geometry.
            t: (B,) integer timesteps.
            cond: (B, cond_dim) behavioral embedding.
            neighbor_geom: (B, geom_dim) flattened neighbor geometry.
            merge_point: (B, 1) topology change point.
            offset: (B, 1) lateral distance.
            has_relation: (B, 1) binary flag.

        Returns:
            (B, geom_dim) predicted noise.
        """
        t_emb = sinusoidal_embedding(t, self.t_dim)
        inp = torch.cat([x_t, t_emb], dim=-1)

        # Relational encoding
        rel_emb = self.rel_encoder(neighbor_geom, merge_point, offset, has_relation)

        # Augmented conditioning: behavioral + relational
        cond_aug = torch.cat([cond, rel_emb], dim=-1)

        h = self.layer1(inp, cond_aug)
        h = self.layer2(h, cond_aug)
        h = self.layer3(h, cond_aug)
        return self.out(h)

# ---------------------------------------------------------------------------
# Relational diffusion trainer
# ---------------------------------------------------------------------------

class RelationalDiffusionTrainer:
    """Train a RelationalLaneDenoiser with DDPM.

    Same training loop as LaneDiffusionTrainer but passes relational
    context through to the denoiser at each step.
    """

    def __init__(
        self,
        model: RelationalLaneDenoiser,
        schedule: DDPMSchedule,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.schedule = schedule.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_epoch(
        self,
        geometries: torch.Tensor,
        cond_embeddings: torch.Tensor,
        neighbor_geoms: torch.Tensor,
        merge_points: torch.Tensor,
        offsets: torch.Tensor,
        has_relations: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        """Train one epoch on relational pairs.

        Args:
            geometries: (N, geom_dim) flattened target lane geometries.
            cond_embeddings: (N, cond_dim) behavioral embeddings.
            neighbor_geoms: (N, geom_dim) flattened neighbor geometries.
            merge_points: (N, 1) topology change locations.
            offsets: (N, 1) lateral distances.
            has_relations: (N, 1) binary flags.
            batch_size: Training batch size.

        Returns:
            Mean loss for the epoch.
        """
        self.model.train()
        N = geometries.shape[0]

        geometries = geometries.to(self.device)
        cond_embeddings = cond_embeddings.to(self.device)
        neighbor_geoms = neighbor_geoms.to(self.device)
        merge_points = merge_points.to(self.device)
        offsets = offsets.to(self.device)
        has_relations = has_relations.to(self.device)

        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            x_0 = geometries[idx]
            cond = cond_embeddings[idx]
            nb_geom = neighbor_geoms[idx]
            mp = merge_points[idx]
            off = offsets[idx]
            hr = has_relations[idx]
            B = x_0.shape[0]

            t = torch.randint(0, self.schedule.T, (B,), device=self.device)
            x_t, noise = self.schedule.q_sample(x_0, t)

            pred_noise = self.model(x_t, t, cond, nb_geom, mp, off, hr)
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
        neighbor_geom: torch.Tensor,
        merge_point: torch.Tensor,
        offset: torch.Tensor,
        has_relation: torch.Tensor,
        n_samples: int = 1,
        warm_start: Optional[torch.Tensor] = None,
        warm_start_t: int = 50,
    ) -> torch.Tensor:
        """Generate lane geometries via reverse diffusion with relational context.

        Args:
            cond: (cond_dim,) or (n_samples, cond_dim) conditioning embedding.
            neighbor_geom: (geom_dim,) or (n_samples, geom_dim) neighbor geometry.
            merge_point: (1,) or (n_samples, 1) merge point.
            offset: (1,) or (n_samples, 1) lateral offset.
            has_relation: (1,) or (n_samples, 1) relation flag.
            n_samples: Number of samples.
            warm_start: Optional initial geometry.
            warm_start_t: Timestep to start from for warm start.

        Returns:
            (n_samples, geom_dim) generated geometries.
        """
        self.model.eval()

        # Expand all inputs to (n_samples, dim)
        def _expand(x, dim):
            if x.dim() == 1:
                x = x.unsqueeze(0).expand(n_samples, -1)
            return x.to(self.device)

        cond = _expand(cond, -1)
        neighbor_geom = _expand(neighbor_geom, -1)
        merge_point = _expand(merge_point, -1)
        offset = _expand(offset, -1)
        has_relation = _expand(has_relation, -1)

        if warm_start is not None:
            if warm_start.dim() == 1:
                warm_start = warm_start.unsqueeze(0).expand(n_samples, -1)
            warm_start = warm_start.to(self.device)
            t_tensor = torch.full(
                (n_samples,), warm_start_t,
                device=self.device, dtype=torch.long,
            )
            x_t, _ = self.schedule.q_sample(warm_start, t_tensor)
            start_t = warm_start_t
        else:
            geom_dim = self.model.geom_dim
            x_t = torch.randn(n_samples, geom_dim, device=self.device)
            start_t = self.schedule.T - 1

        # Reverse diffusion
        for t_val in reversed(range(0, start_t + 1)):
            t = torch.full(
                (n_samples,), t_val, device=self.device, dtype=torch.long,
            )

            pred_noise = self.model(
                x_t, t, cond, neighbor_geom, merge_point, offset,
                has_relation,
            )

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
