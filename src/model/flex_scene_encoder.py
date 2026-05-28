"""FLEX-style scene token encoder for compressed visual prefixes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class FlexSceneConfig:
    enabled: bool = False
    tokens_per_image: int = 32
    expected_images_per_sample: int = 16
    input_hidden_size: int = 2048
    hidden_size: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_camera_time_embeddings: bool = False
    max_camera_types: int = 16

    @property
    def scene_tokens(self) -> int:
        return max(int(self.tokens_per_image), 0) * max(int(self.expected_images_per_sample), 0)


class FlexSceneEncoder(nn.Module):
    """Compress all image tokens in a sample into a fixed set of scene tokens."""

    def __init__(self, config: FlexSceneConfig) -> None:
        super().__init__()
        if config.scene_tokens <= 0:
            raise ValueError("FlexSceneEncoder requires a positive scene token count.")
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_hidden_size)
        self.input_proj = nn.Linear(config.input_hidden_size, config.hidden_size, bias=False)
        self.camera_embed = (
            nn.Embedding(config.max_camera_types, config.hidden_size)
            if config.use_camera_time_embeddings
            else None
        )
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(1, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            if config.use_camera_time_embeddings
            else None
        )
        self.scene_tokens = nn.Parameter(torch.empty(config.scene_tokens, config.hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_size * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.input_hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.scene_tokens, mean=0.0, std=0.02)
        if self.camera_embed is not None:
            nn.init.zeros_(self.camera_embed.weight)
        if self.time_mlp is not None:
            final = self.time_mlp[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        camera_ids: torch.Tensor | None = None,
        relative_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return compressed scene tokens from ``[batch, tokens, input_hidden_size]`` inputs."""
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be rank-3, got shape={tuple(visual_tokens.shape)}")
        batch_size = int(visual_tokens.shape[0])
        projected = self.input_proj(self.input_norm(visual_tokens))
        if self.config.use_camera_time_embeddings:
            if self.camera_embed is None or self.time_mlp is None:
                raise RuntimeError("FLEX camera/time embeddings are enabled but modules are missing.")
            if camera_ids is None or relative_times is None:
                raise ValueError("FLEX camera/time embeddings require camera_ids and relative_times.")
            if tuple(camera_ids.shape) != tuple(projected.shape[:2]):
                raise ValueError(
                    "camera_ids must match visual token batch/length; "
                    f"got {tuple(camera_ids.shape)} vs {tuple(projected.shape[:2])}."
                )
            if relative_times.ndim == 2:
                relative_times = relative_times.unsqueeze(-1)
            if tuple(relative_times.shape[:2]) != tuple(projected.shape[:2]) or int(relative_times.shape[-1]) != 1:
                raise ValueError(
                    "relative_times must be [batch, tokens, 1]; "
                    f"got {tuple(relative_times.shape)} for projected {tuple(projected.shape)}."
                )
            clamped_camera_ids = camera_ids.to(device=projected.device, dtype=torch.long).clamp(
                min=0,
                max=max(int(self.config.max_camera_types) - 1, 0),
            )
            projected = projected + self.camera_embed(clamped_camera_ids).to(dtype=projected.dtype)
            projected = projected + self.time_mlp(relative_times.to(device=projected.device, dtype=projected.dtype))
        queries = self.scene_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([queries, projected], dim=1))
        scene = encoded[:, : self.config.scene_tokens, :]
        return self.output_proj(self.output_norm(scene))
