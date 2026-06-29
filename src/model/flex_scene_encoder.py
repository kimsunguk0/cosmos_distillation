"""FLEX-style scene token encoder for compressed visual prefixes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class FlexSceneConfig:
    enabled: bool = False
    architecture: str = "single_level"
    tokens_per_image: int = 32
    expected_images_per_sample: int = 16
    input_hidden_size: int = 2048
    hidden_size: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_camera_time_embeddings: bool = False
    use_local_slot_embeddings: bool = True
    max_camera_types: int = 16
    compression_mode: str = "global"
    selection_strategy: str = "first"
    num_deepstack_levels: int = 3

    @property
    def scene_tokens(self) -> int:
        return max(int(self.tokens_per_image), 0) * max(int(self.expected_images_per_sample), 0)


class FlexCrossAttentionBlock(nn.Module):
    """Cross-attention block used by ML-FLEX level encoders."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        ffn_hidden = int(hidden_size * float(mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_hidden, hidden_size),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        queries: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        visual_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.query_norm(queries)
        kv = self.visual_norm(visual_tokens)
        attended, _ = self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=visual_padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class FlexLevelEncoder(nn.Module):
    """Level-specific cross-attention encoder for one ViT feature depth."""

    def __init__(self, config: FlexSceneConfig) -> None:
        super().__init__()
        layer_count = max(int(config.num_layers), 1)
        self.layers = nn.ModuleList(
            FlexCrossAttentionBlock(
                hidden_size=int(config.hidden_size),
                num_heads=int(config.num_heads),
                mlp_ratio=float(config.mlp_ratio),
                dropout=float(config.dropout),
            )
            for _ in range(layer_count)
        )

    def forward(
        self,
        queries: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        visual_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = queries
        for layer in self.layers:
            x = layer(x, visual_tokens, visual_padding_mask=visual_padding_mask)
        return x


class MultiLevelFlexEncoder(nn.Module):
    """Compress final and DeepStack visual streams into an aligned FLEX slot grid."""

    def __init__(self, config: FlexSceneConfig) -> None:
        super().__init__()
        if config.scene_tokens <= 0:
            raise ValueError("MultiLevelFlexEncoder requires a positive scene token count.")
        architecture = str(config.architecture or "multi_level").strip().lower()
        if architecture not in {"multi_level", "ml_flex", "ml-flex"}:
            raise ValueError(f"Unsupported ML-FLEX architecture={config.architecture!r}")
        config.architecture = "multi_level"
        mode = str(config.compression_mode or "per_image").strip().lower()
        if mode != "per_image":
            raise ValueError("ML-FLEX requires compression_mode='per_image' to preserve camera/frame slot order.")
        config.compression_mode = mode
        strategy = str(config.selection_strategy or "first").strip().lower()
        if strategy not in {"first", "uniform"}:
            raise ValueError(f"Unsupported FLEX selection_strategy={config.selection_strategy!r}")
        config.selection_strategy = strategy
        level_count = int(config.num_deepstack_levels) + 1
        if level_count <= 1:
            raise ValueError("MultiLevelFlexEncoder requires at least one DeepStack level plus final level.")
        self.config = config
        self.level_count = level_count
        self.scene_tokens = nn.Parameter(torch.empty(config.scene_tokens, config.hidden_size))
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
        self.local_slot_embed = (
            nn.Embedding(config.tokens_per_image, config.hidden_size)
            if config.use_local_slot_embeddings
            else None
        )
        self.input_norms = nn.ModuleList(
            nn.LayerNorm(config.input_hidden_size) for _ in range(level_count)
        )
        self.input_projs = nn.ModuleList(
            nn.Linear(config.input_hidden_size, config.hidden_size, bias=False)
            for _ in range(level_count)
        )
        self.level_encoders = nn.ModuleList(FlexLevelEncoder(config) for _ in range(level_count))
        self.output_norms = nn.ModuleList(nn.LayerNorm(config.hidden_size) for _ in range(level_count))
        self.output_projs = nn.ModuleList(
            nn.Linear(config.hidden_size, config.input_hidden_size, bias=False)
            for _ in range(level_count)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.scene_tokens, mean=0.0, std=0.02)
        if self.camera_embed is not None:
            nn.init.zeros_(self.camera_embed.weight)
        if self.local_slot_embed is not None:
            nn.init.normal_(self.local_slot_embed.weight, mean=0.0, std=0.02)
        if self.time_mlp is not None:
            final = self.time_mlp[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _resolve_image_token_lengths(
        self,
        visual_tokens: torch.Tensor,
        image_token_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, total_tokens = int(visual_tokens.shape[0]), int(visual_tokens.shape[1])
        image_count = max(int(self.config.expected_images_per_sample), 1)
        if image_token_lengths is None:
            if total_tokens % image_count != 0:
                raise ValueError(
                    "ML-FLEX needs image_token_lengths when visual token count is not uniform; "
                    f"total_tokens={total_tokens}, images={image_count}."
                )
            return torch.full(
                (batch_size, image_count),
                total_tokens // image_count,
                dtype=torch.long,
                device=visual_tokens.device,
            )
        lengths = image_token_lengths.to(device=visual_tokens.device, dtype=torch.long)
        if tuple(lengths.shape) != (batch_size, image_count):
            raise ValueError(
                "image_token_lengths must be [batch, expected_images_per_sample]; "
                f"got {tuple(lengths.shape)} vs {(batch_size, image_count)}."
            )
        row_sums = lengths.sum(dim=1)
        if not bool(torch.all(row_sums == total_tokens).item()):
            raise ValueError(
                "image_token_lengths row sums must match visual token count; "
                f"sums={row_sums.detach().cpu().tolist()}, total={total_tokens}."
            )
        return lengths

    def _query_metadata(
        self,
        *,
        camera_ids: torch.Tensor,
        relative_times: torch.Tensor,
        image_token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, image_count = int(image_token_lengths.shape[0]), int(image_token_lengths.shape[1])
        tokens_per_image = max(int(self.config.tokens_per_image), 1)
        query_camera_ids = torch.zeros(
            (batch_size, image_count * tokens_per_image),
            dtype=torch.long,
            device=camera_ids.device,
        )
        query_times = torch.zeros(
            (batch_size, image_count * tokens_per_image, 1),
            dtype=relative_times.dtype,
            device=relative_times.device,
        )
        local_slots = torch.arange(tokens_per_image, dtype=torch.long, device=camera_ids.device)
        query_local_slots = local_slots.view(1, 1, -1).expand(batch_size, image_count, -1)
        query_local_slots = query_local_slots.reshape(batch_size, image_count * tokens_per_image)
        offsets = torch.zeros_like(image_token_lengths)
        offsets[:, 1:] = torch.cumsum(image_token_lengths[:, :-1], dim=1)
        for row_index in range(batch_size):
            for image_index in range(image_count):
                length = int(image_token_lengths[row_index, image_index].item())
                slot_start = image_index * tokens_per_image
                slot_end = slot_start + tokens_per_image
                if length <= 0:
                    continue
                token_index = int(offsets[row_index, image_index].item())
                query_camera_ids[row_index, slot_start:slot_end] = camera_ids[row_index, token_index]
                query_times[row_index, slot_start:slot_end] = relative_times[row_index, token_index]
        return query_camera_ids, query_times, query_local_slots

    def _base_queries(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        camera_ids: torch.Tensor | None,
        relative_times: torch.Tensor | None,
        image_token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.scene_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        if self.local_slot_embed is not None:
            tokens_per_image = max(int(self.config.tokens_per_image), 1)
            image_count = max(int(self.config.expected_images_per_sample), 1)
            local_slots = torch.arange(tokens_per_image, dtype=torch.long, device=device)
            local_slots = local_slots.view(1, -1).expand(image_count, -1).reshape(-1)
            queries = queries + self.local_slot_embed(local_slots).to(dtype=dtype).unsqueeze(0)
        if self.config.use_camera_time_embeddings:
            if self.camera_embed is None or self.time_mlp is None:
                raise RuntimeError("ML-FLEX camera/time embeddings are enabled but modules are missing.")
            if camera_ids is None or relative_times is None:
                raise ValueError("ML-FLEX camera/time embeddings require camera_ids and relative_times.")
            if relative_times.ndim == 2:
                relative_times = relative_times.unsqueeze(-1)
            query_camera_ids, query_times, _ = self._query_metadata(
                camera_ids=camera_ids.to(device=device, dtype=torch.long),
                relative_times=relative_times.to(device=device, dtype=dtype),
                image_token_lengths=image_token_lengths,
            )
            clamped = query_camera_ids.clamp(min=0, max=max(int(self.config.max_camera_types) - 1, 0))
            queries = queries + self.camera_embed(clamped).to(dtype=dtype)
            queries = queries + self.time_mlp(query_times).to(dtype=dtype)
        return queries

    def _project_visual_tokens(
        self,
        level_idx: int,
        tokens: torch.Tensor,
        *,
        camera_ids: torch.Tensor | None,
        relative_times: torch.Tensor | None,
    ) -> torch.Tensor:
        projected = self.input_projs[level_idx](self.input_norms[level_idx](tokens))
        if self.config.use_camera_time_embeddings:
            if self.camera_embed is None or self.time_mlp is None:
                raise RuntimeError("ML-FLEX camera/time embeddings are enabled but modules are missing.")
            if camera_ids is None or relative_times is None:
                raise ValueError("ML-FLEX camera/time embeddings require camera_ids and relative_times.")
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
        return projected

    def forward(
        self,
        final_visual_tokens: torch.Tensor,
        *,
        deepstack_visual_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...],
        camera_ids: torch.Tensor | None = None,
        relative_times: torch.Tensor | None = None,
        image_token_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if final_visual_tokens.ndim != 3:
            raise ValueError(
                f"final_visual_tokens must be rank-3, got shape={tuple(final_visual_tokens.shape)}"
            )
        deepstack_tokens = list(deepstack_visual_tokens or [])
        expected_deepstack = int(self.config.num_deepstack_levels)
        if len(deepstack_tokens) != expected_deepstack:
            raise ValueError(
                f"ML-FLEX expected {expected_deepstack} DeepStack tensors, got {len(deepstack_tokens)}."
            )
        for level_idx, tokens in enumerate(deepstack_tokens):
            if tuple(tokens.shape) != tuple(final_visual_tokens.shape):
                raise ValueError(
                    "ML-FLEX requires DeepStack tensors to match final visual token shape; "
                    f"level={level_idx}, deepstack={tuple(tokens.shape)}, final={tuple(final_visual_tokens.shape)}."
                )
        lengths = self._resolve_image_token_lengths(final_visual_tokens, image_token_lengths)
        queries = self._base_queries(
            batch_size=int(final_visual_tokens.shape[0]),
            dtype=final_visual_tokens.dtype,
            device=final_visual_tokens.device,
            camera_ids=camera_ids,
            relative_times=relative_times,
            image_token_lengths=lengths,
        )
        all_tokens = deepstack_tokens + [final_visual_tokens]
        compressed_levels: list[torch.Tensor] = []
        for level_idx, tokens in enumerate(all_tokens):
            projected = self._project_visual_tokens(
                level_idx,
                tokens,
                camera_ids=camera_ids,
                relative_times=relative_times,
            )
            compressed = self.level_encoders[level_idx](queries, projected)
            compressed = self.output_projs[level_idx](self.output_norms[level_idx](compressed))
            compressed_levels.append(compressed)
        return compressed_levels[-1], compressed_levels[:-1]


class FlexSceneEncoder(nn.Module):
    """Compress all image tokens in a sample into a fixed set of scene tokens."""

    def __init__(self, config: FlexSceneConfig) -> None:
        super().__init__()
        if config.scene_tokens <= 0:
            raise ValueError("FlexSceneEncoder requires a positive scene token count.")
        mode = str(config.compression_mode or "global").strip().lower()
        if mode not in {"global", "per_image", "anchored_per_image"}:
            raise ValueError(f"Unsupported FLEX compression_mode={config.compression_mode!r}")
        config.compression_mode = mode
        strategy = str(config.selection_strategy or "first").strip().lower()
        if strategy not in {"first", "uniform"}:
            raise ValueError(f"Unsupported FLEX selection_strategy={config.selection_strategy!r}")
        config.selection_strategy = strategy
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
        if self.config.compression_mode == "anchored_per_image":
            nn.init.zeros_(self.output_proj.weight)
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
        image_token_lengths: torch.Tensor | None = None,
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
        if self.config.compression_mode == "anchored_per_image":
            return self._forward_anchored_per_image(
                projected,
                visual_tokens=visual_tokens,
                image_token_lengths=image_token_lengths,
            )
        if self.config.compression_mode == "per_image":
            scene = self._forward_per_image(projected, image_token_lengths=image_token_lengths)
        else:
            queries = self.scene_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            encoded = self.encoder(torch.cat([queries, projected], dim=1))
            scene = encoded[:, : self.config.scene_tokens, :]
        return self.output_proj(self.output_norm(scene))

    @staticmethod
    def _select_offsets(
        *,
        length: int,
        count: int,
        strategy: str,
        device: torch.device,
    ) -> torch.Tensor:
        length = max(int(length), 0)
        keep_count = min(max(int(count), 0), length)
        if keep_count <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        strategy = str(strategy or "first").lower()
        if strategy == "first":
            return torch.arange(keep_count, dtype=torch.long, device=device)
        if strategy == "uniform":
            if keep_count == length:
                return torch.arange(length, dtype=torch.long, device=device)
            offsets = torch.div(
                (torch.arange(keep_count, dtype=torch.long, device=device) * 2 + 1) * length,
                2 * keep_count,
                rounding_mode="floor",
            )
            return offsets.clamp_(0, length - 1)
        raise ValueError(f"Unsupported FLEX selection strategy: {strategy!r}.")

    def _resolve_image_token_lengths(
        self,
        projected: torch.Tensor,
        image_token_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, total_tokens = int(projected.shape[0]), int(projected.shape[1])
        image_count = max(int(self.config.expected_images_per_sample), 1)
        if image_token_lengths is None:
            if total_tokens % image_count != 0:
                raise ValueError(
                    "per_image FLEX needs image_token_lengths when visual token count is not uniform; "
                    f"total_tokens={total_tokens}, images={image_count}."
                )
            return torch.full(
                (batch_size, image_count),
                total_tokens // image_count,
                dtype=torch.long,
                device=projected.device,
            )
        lengths = image_token_lengths.to(device=projected.device, dtype=torch.long)
        if tuple(lengths.shape) != (batch_size, image_count):
            raise ValueError(
                "image_token_lengths must be [batch, expected_images_per_sample]; "
                f"got {tuple(lengths.shape)} vs {(batch_size, image_count)}."
            )
        row_sums = lengths.sum(dim=1)
        if not bool(torch.all(row_sums == total_tokens).item()):
            raise ValueError(
                "image_token_lengths row sums must match visual token count; "
                f"sums={row_sums.detach().cpu().tolist()}, total={total_tokens}."
            )
        return lengths

    def _forward_per_image(
        self,
        projected: torch.Tensor,
        *,
        image_token_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compress each image independently so output token blocks keep camera/frame order."""
        lengths = self._resolve_image_token_lengths(projected, image_token_lengths)
        batch_size = int(projected.shape[0])
        image_count = int(lengths.shape[1])
        tokens_per_image = max(int(self.config.tokens_per_image), 1)
        hidden_size = int(projected.shape[-1])
        queries = self.scene_tokens.view(image_count, tokens_per_image, hidden_size)

        if bool(torch.all(lengths == lengths.flatten()[0]).item()):
            token_count = int(lengths.flatten()[0].item())
            projected_images = projected.view(batch_size, image_count, token_count, hidden_size)
            flat_visual = projected_images.reshape(batch_size * image_count, token_count, hidden_size)
            flat_queries = (
                queries.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .reshape(batch_size * image_count, tokens_per_image, hidden_size)
            )
            encoded = self.encoder(torch.cat([flat_queries, flat_visual], dim=1))
            return encoded[:, :tokens_per_image, :].reshape(batch_size, image_count * tokens_per_image, hidden_size)

        offsets = torch.zeros_like(lengths)
        offsets[:, 1:] = torch.cumsum(lengths[:, :-1], dim=1)
        scene_parts: list[torch.Tensor] = []
        for image_index in range(image_count):
            max_len = int(lengths[:, image_index].max().item())
            image_tokens = projected.new_zeros((batch_size, max_len, hidden_size))
            padding_mask = torch.zeros(
                (batch_size, tokens_per_image + max_len),
                dtype=torch.bool,
                device=projected.device,
            )
            for row_index in range(batch_size):
                start = int(offsets[row_index, image_index].item())
                length = int(lengths[row_index, image_index].item())
                if length > 0:
                    image_tokens[row_index, :length] = projected[row_index, start : start + length]
                if length < max_len:
                    padding_mask[row_index, tokens_per_image + length :] = True
            row_queries = queries[image_index].unsqueeze(0).expand(batch_size, -1, -1)
            encoded = self.encoder(
                torch.cat([row_queries, image_tokens], dim=1),
                src_key_padding_mask=padding_mask if max_len > 0 else None,
            )
            scene_parts.append(encoded[:, :tokens_per_image, :])
        return torch.cat(scene_parts, dim=1)

    def _forward_anchored_per_image(
        self,
        projected: torch.Tensor,
        *,
        visual_tokens: torch.Tensor,
        image_token_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-image compression initialized around selected original visual features.

        The selected visual tokens are anchors, while the Transformer learns a
        residual update.  With the zero-initialized output projection this starts
        as a deterministic selector baseline, then becomes learned compression as
        the residual path trains.
        """
        lengths = self._resolve_image_token_lengths(projected, image_token_lengths)
        batch_size = int(projected.shape[0])
        image_count = int(lengths.shape[1])
        tokens_per_image = max(int(self.config.tokens_per_image), 1)
        hidden_size = int(projected.shape[-1])
        input_hidden_size = int(visual_tokens.shape[-1])
        queries = self.scene_tokens.view(image_count, tokens_per_image, hidden_size)
        offsets_by_image = torch.zeros_like(lengths)
        offsets_by_image[:, 1:] = torch.cumsum(lengths[:, :-1], dim=1)

        scene_parts: list[torch.Tensor] = []
        strategy = str(self.config.selection_strategy or "first")
        for image_index in range(image_count):
            max_len = int(lengths[:, image_index].max().item())
            image_tokens = projected.new_zeros((batch_size, max_len, hidden_size))
            anchor_tokens = visual_tokens.new_zeros((batch_size, tokens_per_image, input_hidden_size))
            anchor_queries = projected.new_zeros((batch_size, tokens_per_image, hidden_size))
            padding_mask = torch.zeros(
                (batch_size, tokens_per_image + max_len),
                dtype=torch.bool,
                device=projected.device,
            )
            for row_index in range(batch_size):
                start = int(offsets_by_image[row_index, image_index].item())
                length = int(lengths[row_index, image_index].item())
                if length > 0:
                    image_tokens[row_index, :length] = projected[row_index, start : start + length]
                if length < max_len:
                    padding_mask[row_index, tokens_per_image + length :] = True
                keep_offsets = self._select_offsets(
                    length=length,
                    count=tokens_per_image,
                    strategy=strategy,
                    device=projected.device,
                )
                keep_count = int(keep_offsets.numel())
                if keep_count > 0:
                    token_positions = start + keep_offsets
                    anchor_tokens[row_index, :keep_count] = visual_tokens[row_index].index_select(0, token_positions)
                    anchor_queries[row_index, :keep_count] = projected[row_index].index_select(0, token_positions)
                if keep_count < tokens_per_image:
                    padding_mask[row_index, keep_count:tokens_per_image] = True

            row_queries = anchor_queries + queries[image_index].unsqueeze(0)
            encoded = self.encoder(
                torch.cat([row_queries, image_tokens], dim=1),
                src_key_padding_mask=padding_mask if max_len > 0 else None,
            )
            residual = self.output_proj(self.output_norm(encoded[:, :tokens_per_image, :]))
            scene_parts.append(anchor_tokens + residual)
        return torch.cat(scene_parts, dim=1)
