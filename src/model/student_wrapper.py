"""Student wrapper contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - older transformers fallback
    Qwen3VLForConditionalGeneration = None

from src.data.consistency import ACTION_CLASSES
from src.model.flex_scene_encoder import FlexSceneConfig, FlexSceneEncoder, MultiLevelFlexEncoder
from src.model.tokenizer_ext import REQUIRED_SPECIAL_TOKENS, ensure_special_tokens


@dataclass(slots=True)
class StudentWrapperConfig:
    student_model_name: str = "nvidia/Cosmos-Reason2-2B"
    teacher_model_name: str = "nvidia/Alpamayo-1.5-10B"
    max_length: int = 4096
    min_pixels: int = 49152
    max_pixels: int = 196608
    torch_dtype: torch.dtype | None = None
    special_tokens: tuple[str, ...] = field(default_factory=lambda: tuple(REQUIRED_SPECIAL_TOKENS))
    trust_remote_code: bool = True
    local_files_only: bool = False
    attn_implementation: str | None = None
    traj_teacher_hidden_size: int | None = None
    traj_aux_num_buckets: int = 1
    traj_hidden_bridge_size: int | None = None
    boundary_action_head_hidden_size: int = 1024
    boundary_action_head_dropout: float = 0.05
    flex_scene: FlexSceneConfig | None = None
    vit_in_dim: int = 4096  # teacher ViT output dim
    use_vit_projection: bool = False  # whether to use teacher ViT + projection


def _effective_local_files_only(config: StudentWrapperConfig) -> bool:
    return config.local_files_only or Path(config.student_model_name).expanduser().exists()


def _checkpoint_artifact_source(config: StudentWrapperConfig, artifact_dir: str) -> str:
    """Resolve tokenizer/processor subdirs saved beside a full HF model checkpoint."""
    root = Path(config.student_model_name).expanduser()
    if root.exists():
        nested = root / artifact_dir
        if nested.exists():
            return str(nested)
    return config.student_model_name


class BoundaryActionHead(nn.Module):
    """Small readout from CoT/action boundary hidden states to teacher trajectory xyz."""

    def __init__(self, hidden_size: int, head_hidden_size: int = 1024, dropout: float = 0.05) -> None:
        super().__init__()
        input_dim = int(hidden_size) * 3
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(head_hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_size), int(head_hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_size), 64 * 3),
        )

    def forward(self, boundary_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(boundary_hidden.reshape(boundary_hidden.shape[0], -1)).view(-1, 64, 3)


class FlexDeepStackProjector(nn.Module):
    """Layer-specific low-rank adapters for compressed DeepStack tokens.

    The adapters are zero-initialized at the output projection, so enabling the
    module initially matches the no-compressed-DeepStack behavior instead of the
    harmful "repeat final scene embeddings at every DeepStack layer" baseline.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        *,
        rank: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        rank = int(rank)
        if hidden_size <= 0:
            raise ValueError("FlexDeepStackProjector requires hidden_size > 0.")
        if num_layers <= 0:
            raise ValueError("FlexDeepStackProjector requires num_layers > 0.")
        if rank <= 0:
            raise ValueError("FlexDeepStackProjector requires rank > 0.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rank = rank
        self.dropout_p = float(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.down = nn.ModuleList(nn.Linear(hidden_size, rank, bias=False) for _ in range(num_layers))
        self.up = nn.ModuleList(nn.Linear(rank, hidden_size, bias=False) for _ in range(num_layers))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for down, up in zip(self.down, self.up, strict=True):
            nn.init.normal_(down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(up.weight)

    def forward(self, scene_embeds: torch.Tensor) -> list[torch.Tensor]:
        flat_scene = scene_embeds.reshape(-1, scene_embeds.shape[-1])
        normalized = self.norm(flat_scene.float())
        outputs: list[torch.Tensor] = []
        for down, up in zip(self.down, self.up, strict=True):
            residual = up(self.dropout(down(normalized)))
            outputs.append(residual.to(dtype=flat_scene.dtype))
        return outputs


class DistillStudentModel(nn.Module):
    """Thin wrapper around a causal LM plus a meta-action classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_action_classes: int,
        *,
        traj_teacher_hidden_size: int | None = None,
        traj_aux_num_buckets: int = 1,
        traj_hidden_bridge_size: int | None = None,
        boundary_action_head_hidden_size: int = 1024,
        boundary_action_head_dropout: float = 0.05,
        flex_scene: FlexSceneConfig | None = None,
        image_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.flex_scene_config: FlexSceneConfig | None = None
        self.flex_scene_encoder: FlexSceneEncoder | MultiLevelFlexEncoder | None = None
        self.flex_deepstack_projector: FlexDeepStackProjector | None = None
        self.flex_deepstack_projector_config: dict[str, int | float] | None = None
        self.configure_flex_scene(flex_scene)
        self.meta_action_head = nn.Linear(hidden_size, num_action_classes)
        self.traj_aux_num_buckets: int = 1
        self.traj_aux_head: nn.Linear | None = None
        self.configure_traj_aux_head(traj_aux_num_buckets)
        self.boundary_action_head = BoundaryActionHead(
            hidden_size,
            head_hidden_size=boundary_action_head_hidden_size,
            dropout=boundary_action_head_dropout,
        )
        for parameter in self.boundary_action_head.parameters():
            parameter.requires_grad = False
        self.num_action_classes = num_action_classes
        self.traj_teacher_hidden_size: int | None = None
        self.traj_hidden_projector: nn.Linear | None = None
        self.traj_hidden_bridge_size: int | None = None
        self.traj_hidden_bridge_student: nn.Module | None = None
        self.traj_hidden_bridge_teacher: nn.Module | None = None
        self.configure_traj_hidden_bridge(
            teacher_hidden_size=traj_teacher_hidden_size,
            bridge_size=traj_hidden_bridge_size,
        )
        self.configure_traj_hidden_projector(
            None if traj_hidden_bridge_size not in (None, 0) else traj_teacher_hidden_size
        )
        if traj_teacher_hidden_size not in (None, 0):
            self.traj_teacher_hidden_size = int(traj_teacher_hidden_size)
        # ViT projection for teacher-ViT-features distillation
        self.vit_projection: nn.Linear | None = None

    def configure_flex_scene(self, flex_scene: FlexSceneConfig | None) -> None:
        """Attach or remove the optional FLEX scene encoder."""
        if flex_scene is None or not bool(flex_scene.enabled):
            self.flex_scene_config = None
            self.flex_scene_encoder = None
            self.flex_deepstack_projector = None
            self.flex_deepstack_projector_config = None
            return
        architecture = str(getattr(flex_scene, "architecture", "single_level") or "single_level").strip().lower()
        encoder_cls = MultiLevelFlexEncoder if architecture in {"multi_level", "ml_flex", "ml-flex"} else FlexSceneEncoder
        if self.flex_scene_config == flex_scene and isinstance(self.flex_scene_encoder, encoder_cls):
            return
        self.flex_scene_config = flex_scene
        self.flex_scene_encoder = encoder_cls(flex_scene)

    def configure_flex_deepstack_projector(
        self,
        *,
        num_layers: int,
        rank: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Attach layer-specific DeepStack adapters for compressed FLEX scene tokens."""
        if self.flex_scene_config is None:
            raise RuntimeError("FLEX scene must be configured before flex_deepstack_projector.")
        config = {
            "hidden_size": int(self.flex_scene_config.input_hidden_size),
            "num_layers": int(num_layers),
            "rank": int(rank),
            "dropout": float(dropout),
        }
        existing = self.flex_deepstack_projector
        if (
            isinstance(existing, FlexDeepStackProjector)
            and existing.hidden_size == int(config["hidden_size"])
            and existing.num_layers == int(config["num_layers"])
            and existing.rank == int(config["rank"])
            and abs(float(existing.dropout_p) - float(config["dropout"])) < 1e-12
        ):
            self.flex_deepstack_projector_config = config
            return
        self.flex_deepstack_projector_config = config
        self.flex_deepstack_projector = FlexDeepStackProjector(
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            rank=int(config["rank"]),
            dropout=float(config["dropout"]),
        )
        if self.flex_scene_encoder is not None:
            target_device = next(self.flex_scene_encoder.parameters()).device
            self.flex_deepstack_projector.to(device=target_device)

    def configure_vit_projection(self, in_dim: int, out_dim: int) -> None:
        """Attach a learnable linear projection from teacher ViT dim to student hidden dim."""
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        existing = self.vit_projection
        if (
            isinstance(existing, nn.Linear)
            and existing.in_features == in_dim
            and existing.out_features == out_dim
        ):
            return
        self.vit_projection = nn.Linear(in_dim, out_dim)

    def embed_with_teacher_vit_features(
        self,
        input_ids: torch.Tensor,
        teacher_image_embeds: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        """Build inputs_embeds by injecting projected teacher ViT features.

        Args:
            input_ids: [batch, seq_len] token ids.
            teacher_image_embeds: [N_total_image_tokens, vit_in_dim] teacher ViT features
                (concatenated across the batch in token order).
            image_token_id: token id used as placeholder for image tokens.

        Returns:
            inputs_embeds: [batch, seq_len, hidden_size] with image positions replaced
                by projected teacher features.
        """
        if self.vit_projection is None:
            raise RuntimeError("vit_projection is not configured. Call configure_vit_projection first.")
        projected = self.vit_projection(teacher_image_embeds)  # [N, hidden_size]
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)  # [batch, seq, hidden]
        image_mask = (input_ids == int(image_token_id))  # [batch, seq]
        n_image_tokens = int(image_mask.sum().item())
        if n_image_tokens != int(projected.shape[0]):
            raise ValueError(
                f"image_mask has {n_image_tokens} True positions but "
                f"teacher_image_embeds has {int(projected.shape[0])} rows."
            )
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            projected.reshape(-1),
        )
        return inputs_embeds

    def configure_traj_aux_head(self, num_buckets: int | None) -> None:
        """Attach or resize the training-time trajectory auxiliary head."""
        resolved_buckets = max(int(num_buckets or 1), 1)
        output_dim = resolved_buckets * 2
        self.traj_aux_num_buckets = resolved_buckets
        head = self.traj_aux_head
        if (
            isinstance(head, nn.Linear)
            and head.in_features == self.hidden_size
            and head.out_features == output_dim
        ):
            return
        self.traj_aux_head = nn.Linear(self.hidden_size, output_dim)

    def configure_traj_hidden_projector(self, output_dim: int | None) -> None:
        """Attach or remove a trainable projector for teacher trajectory hidden alignment."""
        if output_dim in (None, 0):
            self.traj_teacher_hidden_size = None
            self.traj_hidden_projector = None
            return
        output_dim = int(output_dim)
        self.traj_teacher_hidden_size = output_dim
        if output_dim == self.hidden_size:
            self.traj_hidden_projector = None
            return
        projector = self.traj_hidden_projector
        if (
            projector is not None
            and projector.in_features == self.hidden_size
            and projector.out_features == output_dim
        ):
            return
        self.traj_hidden_projector = nn.Linear(self.hidden_size, output_dim, bias=False)

    def configure_traj_hidden_bridge(
        self,
        *,
        teacher_hidden_size: int | None,
        bridge_size: int | None,
    ) -> None:
        """Attach or remove a shared bottleneck used for normalized traj hidden distillation."""
        if teacher_hidden_size in (None, 0) or bridge_size in (None, 0):
            self.traj_hidden_bridge_size = None
            self.traj_hidden_bridge_student = None
            self.traj_hidden_bridge_teacher = None
            return

        teacher_hidden_size = int(teacher_hidden_size)
        bridge_size = int(bridge_size)
        self.traj_teacher_hidden_size = teacher_hidden_size
        self.traj_hidden_bridge_size = bridge_size

        student_bridge = self.traj_hidden_bridge_student
        if not (
            isinstance(student_bridge, nn.Sequential)
            and isinstance(student_bridge[0], nn.Linear)
            and isinstance(student_bridge[1], nn.LayerNorm)
            and student_bridge[0].in_features == self.hidden_size
            and student_bridge[0].out_features == bridge_size
            and student_bridge[1].normalized_shape == (bridge_size,)
        ):
            self.traj_hidden_bridge_student = nn.Sequential(
                nn.Linear(self.hidden_size, bridge_size, bias=False),
                nn.LayerNorm(bridge_size),
            )

        teacher_bridge = self.traj_hidden_bridge_teacher
        if not (
            isinstance(teacher_bridge, nn.Sequential)
            and isinstance(teacher_bridge[0], nn.Linear)
            and isinstance(teacher_bridge[1], nn.LayerNorm)
            and teacher_bridge[0].in_features == teacher_hidden_size
            and teacher_bridge[0].out_features == bridge_size
            and teacher_bridge[1].normalized_shape == (bridge_size,)
        ):
            self.traj_hidden_bridge_teacher = nn.Sequential(
                nn.Linear(teacher_hidden_size, bridge_size, bias=False),
                nn.LayerNorm(bridge_size),
            )

    def project_teacher_traj_hidden(self, teacher_hidden: torch.Tensor | None) -> torch.Tensor | None:
        """Project cached teacher traj hidden states into the shared bridge space."""
        if teacher_hidden is None:
            return None
        if self.traj_hidden_bridge_teacher is None:
            return teacher_hidden
        return self.traj_hidden_bridge_teacher(teacher_hidden)

    def flex_enabled(self) -> bool:
        return self.flex_scene_encoder is not None and self.flex_scene_config is not None and self.flex_scene_config.enabled

    def _conditional_backbone(self) -> nn.Module:
        backbone = self.backbone
        base_model = getattr(backbone, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            return base_model.model
        return backbone

    def _output_head(self) -> nn.Module:
        from src.model.lm_head_adapter import get_output_lm_head

        return get_output_lm_head(self.backbone)

    @staticmethod
    def _position_ids_from_attention_mask(
        attention_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            positions = torch.arange(seq_len, device=device).view(1, -1)
            batch_size = 1
        else:
            positions = attention_mask.long().cumsum(-1) - 1
            positions = positions.masked_fill(attention_mask == 0, 1)
            if int(positions.shape[-1]) != int(seq_len):
                positions = positions[:, -int(seq_len) :]
            batch_size = int(attention_mask.shape[0])
        return positions.view(1, batch_size, seq_len).expand(3, -1, -1)

    def _qwen_visual_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        conditional = self._conditional_backbone()
        image_embeds, deepstack_image_embeds = conditional.get_image_features(pixel_values, image_grid_thw)
        return list(image_embeds), deepstack_image_embeds

    @staticmethod
    def _flex_block_keep_offsets(
        *,
        length: int,
        tokens_per_image: int,
        strategy: str,
        device: torch.device,
    ) -> torch.Tensor:
        length = max(int(length), 0)
        keep_count = min(max(int(tokens_per_image), 0), length)
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
        raise ValueError(f"Unsupported FLEX image-token selection strategy: {strategy!r}.")

    @staticmethod
    def _default_camera_metadata(
        *,
        batch_size: int,
        images_per_sample: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_count = 4 if images_per_sample % 4 == 0 else max(images_per_sample, 1)
        frames_per_camera = max(images_per_sample // max(camera_count, 1), 1)
        base_cameras = torch.arange(camera_count, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        if camera_count == 4:
            base_cameras = torch.tensor([0, 1, 2, 6], dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        frame_offsets = torch.arange(frames_per_camera, dtype=torch.float32, device=device)
        frame_offsets = (frame_offsets - float(frames_per_camera - 1)) * 0.1
        relative_times = frame_offsets.view(1, 1, frames_per_camera).expand(batch_size, camera_count, -1)
        camera_counts = torch.full((batch_size,), camera_count, dtype=torch.long, device=device)
        frames_per_camera_tensor = torch.full((batch_size,), frames_per_camera, dtype=torch.long, device=device)
        return base_cameras, relative_times, camera_counts, frames_per_camera_tensor

    def _expand_flex_token_metadata(
        self,
        image_features: list[torch.Tensor],
        *,
        batch_size: int,
        images_per_sample: int,
        camera_indices: torch.Tensor | None,
        relative_timestamps: torch.Tensor | None,
        camera_counts: torch.Tensor | None,
        frames_per_camera: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if camera_indices is None or relative_timestamps is None:
            camera_indices, relative_timestamps, camera_counts, frames_per_camera = self._default_camera_metadata(
                batch_size=batch_size,
                images_per_sample=images_per_sample,
                device=device,
            )
        else:
            camera_indices = camera_indices.to(device=device, dtype=torch.long)
            relative_timestamps = relative_timestamps.to(device=device, dtype=torch.float32)
            if camera_counts is None:
                camera_counts = torch.full(
                    (batch_size,),
                    int(camera_indices.shape[1]),
                    dtype=torch.long,
                    device=device,
                )
            else:
                camera_counts = camera_counts.to(device=device, dtype=torch.long)
            if frames_per_camera is None:
                inferred_frames = max(images_per_sample // max(int(camera_indices.shape[1]), 1), 1)
                frames_per_camera = torch.full((batch_size,), inferred_frames, dtype=torch.long, device=device)
            else:
                frames_per_camera = frames_per_camera.to(device=device, dtype=torch.long)

        camera_id_rows: list[torch.Tensor] = []
        relative_time_rows: list[torch.Tensor] = []
        for row_index in range(batch_size):
            camera_count = max(int(camera_counts[row_index].item()), 1)
            frame_count = max(int(frames_per_camera[row_index].item()), 1)
            row_camera_ids = []
            row_times = []
            for local_image_index in range(images_per_sample):
                global_image_index = row_index * images_per_sample + local_image_index
                feature_len = int(image_features[global_image_index].shape[0])
                camera_offset = min(local_image_index // frame_count, camera_count - 1)
                frame_index = min(local_image_index % frame_count, int(relative_timestamps.shape[-1]) - 1)
                camera_id = int(camera_indices[row_index, camera_offset].item())
                relative_time = float(relative_timestamps[row_index, camera_offset, frame_index].item())
                row_camera_ids.append(torch.full((feature_len,), camera_id, dtype=torch.long, device=device))
                row_times.append(torch.full((feature_len, 1), relative_time, dtype=torch.float32, device=device))
            camera_id_rows.append(torch.cat(row_camera_ids, dim=0))
            relative_time_rows.append(torch.cat(row_times, dim=0))
        return torch.stack(camera_id_rows, dim=0), torch.stack(relative_time_rows, dim=0)

    def _scene_deepstack_visual_embeds(self, scene_embeds: torch.Tensor) -> list[torch.Tensor] | None:
        """Use compressed scene tokens as DeepStack visual injections.

        Qwen3-VL feeds visual features into the decoder at several early layers.
        Compressed FLEX image slots otherwise lose that entire pathway.  When a
        FLEX DeepStack projector is configured this returns layer-specific
        adapter outputs; otherwise it falls back to the repeated-scene diagnostic
        baseline.
        """
        conditional = self._conditional_backbone()
        visual_model = getattr(conditional, "visual", None)
        layer_count = len(getattr(visual_model, "deepstack_visual_indexes", []) or [])
        if layer_count <= 0:
            layer_count = len(getattr(visual_model, "deepstack_merger_list", []) or [])
        language_model = getattr(getattr(conditional, "model", None), "language_model", None)
        if layer_count <= 0:
            layer_count = len(getattr(language_model, "deepstack_visual_indexes", []) or [])
        if layer_count <= 0:
            layer_count = len(getattr(language_model, "deepstack_merger_list", []) or [])
        if layer_count <= 0:
            return None
        projector = self.flex_deepstack_projector
        if projector is not None:
            if int(projector.num_layers) != int(layer_count):
                raise ValueError(
                    "FLEX DeepStack projector layer count does not match backbone hooks; "
                    f"projector={int(projector.num_layers)}, backbone={layer_count}."
                )
            return projector(scene_embeds)
        flat_scene = scene_embeds.reshape(-1, scene_embeds.shape[-1])
        return [flat_scene for _ in range(layer_count)]

    def _passthrough_deepstack_visual_embeds(
        self,
        deepstack_image_embeds: list[torch.Tensor] | None,
        image_token_lengths: torch.Tensor,
        *,
        tokens_per_image: int,
        selection_strategy: str,
    ) -> list[torch.Tensor] | None:
        if deepstack_image_embeds is None:
            return None
        layer_tensors = list(deepstack_image_embeds)
        if not layer_tensors:
            return None
        batch_size, images_per_sample = int(image_token_lengths.shape[0]), int(image_token_lengths.shape[1])
        flat_lengths = [int(value.item()) for value in image_token_lengths.reshape(-1)]
        image_offsets: list[int] = []
        cursor = 0
        for length in flat_lengths:
            image_offsets.append(cursor)
            cursor += int(length)
        selected_layers: list[torch.Tensor] = []
        for layer_tensor in layer_tensors:
            row_parts: list[torch.Tensor] = []
            layer_device = layer_tensor.device
            for row_index in range(batch_size):
                for local_image_index in range(images_per_sample):
                    flat_index = row_index * images_per_sample + local_image_index
                    length = int(flat_lengths[flat_index])
                    offsets = self._flex_block_keep_offsets(
                        length=length,
                        tokens_per_image=tokens_per_image,
                        strategy=selection_strategy,
                        device=layer_device,
                    )
                    take = int(offsets.numel())
                    start = int(image_offsets[flat_index])
                    if take > 0:
                        row_parts.append(layer_tensor[start : start + length].index_select(0, offsets))
                    if take < int(tokens_per_image):
                        row_parts.append(
                            layer_tensor.new_zeros((int(tokens_per_image) - take, int(layer_tensor.shape[-1])))
                        )
            if row_parts:
                selected_layers.append(torch.cat(row_parts, dim=0))
        return selected_layers or None

    @staticmethod
    def _batch_deepstack_visual_embeds(
        deepstack_image_embeds: list[torch.Tensor] | None,
        image_token_lengths: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        if deepstack_image_embeds is None:
            return None
        layer_tensors = list(deepstack_image_embeds)
        if not layer_tensors:
            return None
        batch_size, images_per_sample = int(image_token_lengths.shape[0]), int(image_token_lengths.shape[1])
        row_lengths = image_token_lengths.sum(dim=1)
        if not bool(torch.all(row_lengths == row_lengths[0]).item()):
            raise ValueError(
                "ML-FLEX currently requires equal visual token counts per sample; "
                f"row_lengths={row_lengths.detach().cpu().tolist()}."
            )
        tokens_per_sample = int(row_lengths[0].item())
        flat_lengths = [int(value.item()) for value in image_token_lengths.reshape(-1)]
        image_offsets: list[int] = []
        cursor = 0
        for length in flat_lengths:
            image_offsets.append(cursor)
            cursor += int(length)
        batched_layers: list[torch.Tensor] = []
        for layer_tensor in layer_tensors:
            row_parts: list[torch.Tensor] = []
            for row_index in range(batch_size):
                sample_parts: list[torch.Tensor] = []
                for local_image_index in range(images_per_sample):
                    flat_index = row_index * images_per_sample + local_image_index
                    start = int(image_offsets[flat_index])
                    length = int(flat_lengths[flat_index])
                    if length > 0:
                        sample_parts.append(layer_tensor[start : start + length])
                if sample_parts:
                    row_parts.append(torch.cat(sample_parts, dim=0))
                else:
                    row_parts.append(layer_tensor.new_zeros((tokens_per_sample, int(layer_tensor.shape[-1]))))
            batched_layers.append(torch.stack(row_parts, dim=0))
        return batched_layers

    def _flex_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        camera_indices: torch.Tensor | None = None,
        relative_timestamps: torch.Tensor | None = None,
        camera_counts: torch.Tensor | None = None,
        frames_per_camera: torch.Tensor | None = None,
        allow_dummy_image_slots: bool = False,
        residual_image_slots: bool = False,
        residual_scale: float = 1.0,
        passthrough_image_slots: bool = False,
        selection_strategy: str = "first",
        scene_deepstack: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if self.flex_scene_encoder is None or self.flex_scene_config is None:
            raise RuntimeError("FLEX scene encoder is not configured.")
        if self.image_token_id is None:
            raise RuntimeError("FLEX requires image_token_id to identify compressed placeholder positions.")
        batch_size = int(input_ids.shape[0])
        if int(image_grid_thw.shape[0]) % max(batch_size, 1) != 0:
            raise ValueError(
                "FLEX expects a uniform number of images per sample; "
                f"got image_grid rows={int(image_grid_thw.shape[0])}, batch={batch_size}."
            )
        images_per_sample = int(image_grid_thw.shape[0]) // max(batch_size, 1)
        expected_images = int(self.flex_scene_config.expected_images_per_sample)
        if expected_images > 0 and images_per_sample != expected_images:
            raise ValueError(f"FLEX expected {expected_images} images/sample, got {images_per_sample}.")

        image_features, deepstack_image_embeds = self._qwen_visual_features(pixel_values, image_grid_thw)
        sample_features = []
        image_token_lengths = torch.zeros(
            (batch_size, images_per_sample),
            dtype=torch.long,
            device=input_ids.device,
        )
        for row_index in range(batch_size):
            start = row_index * images_per_sample
            end = start + images_per_sample
            row_features = image_features[start:end]
            for local_image_index, image_feature in enumerate(row_features):
                image_token_lengths[row_index, local_image_index] = int(image_feature.shape[0])
            sample_features.append(torch.cat(row_features, dim=0))
        visual_tokens = torch.stack(sample_features, dim=0).to(
            device=input_ids.device,
            dtype=self.flex_scene_encoder.scene_tokens.dtype,
        )
        camera_ids = None
        relative_times = None
        if self.flex_scene_config.use_camera_time_embeddings:
            camera_ids, relative_times = self._expand_flex_token_metadata(
                image_features,
                batch_size=batch_size,
                images_per_sample=images_per_sample,
                camera_indices=camera_indices,
                relative_timestamps=relative_timestamps,
                camera_counts=camera_counts,
                frames_per_camera=frames_per_camera,
                device=input_ids.device,
            )
        ml_flex_deepstack = None
        if bool(passthrough_image_slots):
            scene_parts: list[torch.Tensor] = []
            tokens_per_image = int(self.flex_scene_config.tokens_per_image)
            for row_index in range(batch_size):
                offset = 0
                row_parts: list[torch.Tensor] = []
                for local_image_index in range(images_per_sample):
                    length = int(image_token_lengths[row_index, local_image_index].item())
                    offsets = self._flex_block_keep_offsets(
                        length=length,
                        tokens_per_image=tokens_per_image,
                        strategy=selection_strategy,
                        device=input_ids.device,
                    )
                    take = int(offsets.numel())
                    if take > 0:
                        row_parts.append(
                            visual_tokens[row_index, offset : offset + length].index_select(0, offsets)
                        )
                    if take < tokens_per_image:
                        row_parts.append(
                            visual_tokens.new_zeros((tokens_per_image - take, int(visual_tokens.shape[-1])))
                        )
                    offset += length
                scene_parts.append(torch.cat(row_parts, dim=0))
            scene_embeds = torch.stack(scene_parts, dim=0)
        elif isinstance(self.flex_scene_encoder, MultiLevelFlexEncoder):
            batched_deepstack = self._batch_deepstack_visual_embeds(deepstack_image_embeds, image_token_lengths)
            if batched_deepstack is None:
                raise RuntimeError("ML-FLEX requires Qwen DeepStack image embeddings from get_image_features().")
            batched_deepstack = [
                layer.to(device=input_ids.device, dtype=visual_tokens.dtype) for layer in batched_deepstack
            ]
            scene_embeds, ml_flex_deepstack = self.flex_scene_encoder(
                visual_tokens,
                deepstack_visual_tokens=batched_deepstack,
                camera_ids=camera_ids,
                relative_times=relative_times,
                image_token_lengths=image_token_lengths,
            )
        else:
            scene_embeds = self.flex_scene_encoder(
                visual_tokens,
                camera_ids=camera_ids,
                relative_times=relative_times,
                image_token_lengths=image_token_lengths,
            )
        scene_embeds = scene_embeds.to(
            device=input_ids.device,
            dtype=self.backbone.get_input_embeddings().weight.dtype,
        )
        if ml_flex_deepstack is not None:
            ml_flex_deepstack = [
                layer.to(device=input_ids.device, dtype=scene_embeds.dtype).reshape(-1, int(layer.shape[-1]))
                for layer in ml_flex_deepstack
            ]
        passthrough_deepstack = None
        if bool(passthrough_image_slots) and bool(scene_deepstack):
            passthrough_deepstack = self._passthrough_deepstack_visual_embeds(
                deepstack_image_embeds,
                image_token_lengths,
                tokens_per_image=int(self.flex_scene_config.tokens_per_image),
                selection_strategy=selection_strategy,
            )

        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        image_mask = input_ids == int(self.image_token_id)
        expected_scene_tokens = int(scene_embeds.shape[1])
        counts = image_mask.sum(dim=1)
        if bool(residual_image_slots):
            visual_embeds = visual_tokens.to(
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )
            if not bool(torch.all(counts == visual_embeds.shape[1]).item()):
                raise ValueError(
                    "Residual FLEX expects full original image placeholders; "
                    f"counts={counts.detach().cpu().tolist()}, visual_tokens={int(visual_embeds.shape[1])}."
                )
            out = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                visual_embeds.reshape(-1),
            )
            tokens_per_image = int(self.flex_scene_config.tokens_per_image)
            scaled_scene = scene_embeds * float(residual_scale)
            for row_index in range(batch_size):
                row_mask = image_mask[row_index]
                selected: list[int] = []
                cursor = 0
                seq_len = int(row_mask.shape[0])
                while cursor < seq_len:
                    if not bool(row_mask[cursor].item()):
                        cursor += 1
                        continue
                    end = cursor + 1
                    while end < seq_len and bool(row_mask[end].item()):
                        end += 1
                    offsets = self._flex_block_keep_offsets(
                        length=end - cursor,
                        tokens_per_image=tokens_per_image,
                        strategy=selection_strategy,
                        device=input_ids.device,
                    )
                    selected.extend(int(cursor + value.item()) for value in offsets)
                    cursor = end
                if len(selected) != expected_scene_tokens:
                    all_image_positions = torch.nonzero(row_mask, as_tuple=False).flatten()
                    selected = [int(value) for value in all_image_positions[:expected_scene_tokens].tolist()]
                if len(selected) != expected_scene_tokens:
                    raise ValueError(
                        "Residual FLEX could not map scene tokens to image placeholders; "
                        f"selected={len(selected)}, expected={expected_scene_tokens}."
                    )
                positions = torch.tensor(selected, dtype=torch.long, device=input_ids.device)
                out[row_index, positions] = out[row_index, positions] + scaled_scene[row_index]
            return out, image_mask, deepstack_image_embeds
        if bool(torch.all(counts == expected_scene_tokens).item()):
            if bool(scene_deepstack) and ml_flex_deepstack is not None:
                deepstack = ml_flex_deepstack
            elif bool(scene_deepstack) and passthrough_deepstack is not None:
                deepstack = passthrough_deepstack
            else:
                deepstack = self._scene_deepstack_visual_embeds(scene_embeds) if bool(scene_deepstack) else None
            return (
                inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    scene_embeds.reshape(-1),
                ),
                image_mask if deepstack is not None else None,
                deepstack,
            )
        if bool(allow_dummy_image_slots) and bool(torch.all(counts >= expected_scene_tokens).item()):
            tokens_per_image = int(self.flex_scene_config.tokens_per_image)
            out = inputs_embeds.clone()
            for row_index in range(batch_size):
                row_mask = image_mask[row_index]
                selected: list[int] = []
                cursor = 0
                seq_len = int(row_mask.shape[0])
                while cursor < seq_len:
                    if not bool(row_mask[cursor].item()):
                        cursor += 1
                        continue
                    end = cursor + 1
                    while end < seq_len and bool(row_mask[end].item()):
                        end += 1
                    offsets = self._flex_block_keep_offsets(
                        length=end - cursor,
                        tokens_per_image=tokens_per_image,
                        strategy=selection_strategy,
                        device=input_ids.device,
                    )
                    selected.extend(int(cursor + value.item()) for value in offsets)
                    cursor = end
                if len(selected) != expected_scene_tokens:
                    all_image_positions = torch.nonzero(row_mask, as_tuple=False).flatten()
                    selected = [int(value) for value in all_image_positions[:expected_scene_tokens].tolist()]
                if len(selected) != expected_scene_tokens:
                    raise ValueError(
                        "Dummy-slot FLEX could not map scene tokens to image placeholders; "
                        f"selected={len(selected)}, expected={expected_scene_tokens}."
                    )
                positions = torch.tensor(selected, dtype=torch.long, device=input_ids.device)
                out[row_index, positions] = scene_embeds[row_index]
            return out, None, None
        if not bool(torch.all(counts == expected_scene_tokens).item()):
            raise ValueError(
                "Compressed image placeholder count must match FLEX scene tokens; "
                f"counts={counts.detach().cpu().tolist()}, expected={expected_scene_tokens}."
            )
        raise AssertionError("unreachable FLEX image placeholder mapping state")

    def _forward_flex(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        need_hidden_states: bool,
        **kwargs: Any,
    ) -> Any:
        conditional = self._conditional_backbone()
        past_key_values = kwargs.get("past_key_values")
        position_ids_override = kwargs.pop("position_ids", None)
        cache_position = kwargs.get("cache_position")
        use_official_mrope_positions = False
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if past_key_values is None:
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError("FLEX prefill requires pixel_values and image_grid_thw.")
            allow_dummy_image_slots = bool(kwargs.pop("flex_allow_dummy_image_slots", False))
            residual_image_slots = bool(kwargs.pop("flex_residual_image_slots", False))
            residual_scale = float(kwargs.pop("flex_residual_scale", 1.0))
            passthrough_image_slots = bool(kwargs.pop("flex_passthrough_image_slots", False))
            default_selection = (
                getattr(self.flex_scene_config, "selection_strategy", "first")
                if self.flex_scene_config is not None
                else "first"
            )
            selection_strategy = str(kwargs.pop("flex_selection_strategy", default_selection) or default_selection)
            default_scene_deepstack = isinstance(self.flex_scene_encoder, MultiLevelFlexEncoder)
            scene_deepstack = bool(kwargs.pop("flex_scene_deepstack", default_scene_deepstack))
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._flex_inputs_embeds(
                input_ids,
                pixel_values,
                image_grid_thw,
                camera_indices=kwargs.pop("camera_indices", None),
                relative_timestamps=kwargs.pop("relative_timestamps", None),
                camera_counts=kwargs.pop("camera_counts", None),
                frames_per_camera=kwargs.pop("frames_per_camera", None),
                allow_dummy_image_slots=allow_dummy_image_slots,
                residual_image_slots=residual_image_slots,
                residual_scale=residual_scale,
                passthrough_image_slots=passthrough_image_slots,
                selection_strategy=selection_strategy,
                scene_deepstack=scene_deepstack,
            )
            use_official_mrope_positions = bool(allow_dummy_image_slots or residual_image_slots)
        else:
            inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        if position_ids_override is None:
            if use_official_mrope_positions:
                position_ids, rope_deltas = conditional.model.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                )
                conditional.model.rope_deltas = rope_deltas
            elif past_key_values is not None and isinstance(cache_position, torch.Tensor):
                batch_size = int(inputs_embeds.shape[0])
                base_positions = cache_position.to(device=inputs_embeds.device, dtype=torch.long)
                row_positions = base_positions.view(1, -1).expand(batch_size, -1)
                rope_deltas = getattr(conditional.model, "rope_deltas", None)
                if isinstance(rope_deltas, torch.Tensor):
                    delta = rope_deltas.reshape(-1).to(device=inputs_embeds.device, dtype=torch.long)
                    if int(delta.numel()) == 1 and batch_size > 1:
                        delta = delta.expand(batch_size)
                    if int(delta.numel()) == batch_size:
                        row_positions = row_positions + delta.view(batch_size, 1)
                position_ids = row_positions.view(1, batch_size, -1).expand(3, -1, -1)
            else:
                position_ids = self._position_ids_from_attention_mask(
                    attention_mask,
                    inputs_embeds.shape[1],
                    inputs_embeds.device,
                )
        else:
            position_ids = position_ids_override.to(device=inputs_embeds.device, dtype=torch.long)
            if position_ids.ndim == 2:
                batch_size, seq_len = int(position_ids.shape[0]), int(position_ids.shape[1])
                position_ids = position_ids.view(1, batch_size, seq_len).expand(3, -1, -1)
            if position_ids.ndim != 3:
                raise ValueError(f"position_ids override must be rank-2 or rank-3, got {tuple(position_ids.shape)}")
            if int(position_ids.shape[-1]) != int(inputs_embeds.shape[1]):
                raise ValueError(
                    "position_ids override length must match inputs_embeds length; "
                    f"got {tuple(position_ids.shape)} vs {tuple(inputs_embeds.shape)}."
                )
        pre_norm_hidden: torch.Tensor | None = None

        def _capture_pre_norm_hidden(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            nonlocal pre_norm_hidden
            if inputs:
                candidate = inputs[0]
                if isinstance(candidate, torch.Tensor):
                    pre_norm_hidden = candidate

        hook_handle = None
        if need_hidden_states:
            hook_handle = conditional.model.language_model.norm.register_forward_hook(_capture_pre_norm_hidden)
        try:
            language_outputs = conditional.model.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_hidden_states=False,
                return_dict=True,
                use_cache=bool(kwargs.get("use_cache", False)),
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        hidden_states = language_outputs.last_hidden_state
        logits = self._output_head()(hidden_states)

        class _FlexOutput:
            pass

        outputs = _FlexOutput()
        outputs.logits = logits
        outputs.past_key_values = getattr(language_outputs, "past_key_values", None)
        outputs.hidden_states = None
        if need_hidden_states:
            outputs.hidden_states = (pre_norm_hidden if pre_norm_hidden is not None else hidden_states,)
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = True,
        compute_meta_action: bool = True,
        compute_traj_aux: bool = True,
        compute_boundary_action: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | Any]:
        boundary_action_positions = kwargs.pop("boundary_action_positions", None)
        need_hidden_states = bool(return_hidden_states or compute_meta_action or compute_traj_aux or compute_boundary_action)
        if self.flex_enabled() and (
            kwargs.get("pixel_values") is not None or kwargs.get("past_key_values") is not None
        ):
            outputs = self._forward_flex(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=kwargs.pop("pixel_values", None),
                image_grid_thw=kwargs.pop("image_grid_thw", None),
                need_hidden_states=need_hidden_states,
                **kwargs,
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden_states,
                return_dict=True,
                **kwargs,
            )
        logits = getattr(outputs, "logits", None)
        if logits is None and hasattr(outputs, "language_model_outputs"):
            logits = getattr(outputs.language_model_outputs, "logits", None)
        if logits is None:
            raise ValueError("Student backbone did not return logits.")

        result: dict[str, torch.Tensor | Any] = {
            "backbone_outputs": outputs,
            "logits": logits,
        }
        if not need_hidden_states:
            return result

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and hasattr(outputs, "language_model_outputs"):
            hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Student backbone did not return hidden states.")
        hidden = hidden_states[-1]
        result["hidden_states"] = hidden

        if return_hidden_states:
            result["traj_hidden_states"] = (
                self.traj_hidden_projector(hidden) if self.traj_hidden_projector is not None else hidden
            )
            result["traj_hidden_bridge_states"] = (
                self.traj_hidden_bridge_student(hidden)
                if self.traj_hidden_bridge_student is not None
                else None
            )

        if compute_meta_action:
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = (hidden * mask).sum(dim=1) / denom
            result["pooled_hidden"] = pooled
            result["meta_action_logits"] = self.meta_action_head(pooled)

        if compute_traj_aux:
            if self.traj_aux_head is None:
                raise ValueError("Trajectory auxiliary head is not configured.")
            result["traj_aux_values"] = self.traj_aux_head(hidden)

        if compute_boundary_action:
            if boundary_action_positions is None:
                result["boundary_action_xyz"] = None
            else:
                positions = boundary_action_positions.to(device=hidden.device, dtype=torch.long)
                valid = (positions >= 0) & (positions < hidden.shape[1])
                safe_positions = positions.clamp(min=0, max=max(int(hidden.shape[1]) - 1, 0))
                gathered_rows = []
                for row_index in range(hidden.shape[0]):
                    row_hidden = hidden[row_index].index_select(0, safe_positions[row_index])
                    row_hidden = row_hidden * valid[row_index].to(dtype=hidden.dtype).unsqueeze(-1)
                    gathered_rows.append(row_hidden)
                boundary_hidden = torch.stack(gathered_rows, dim=0)
                result["boundary_action_xyz"] = self.boundary_action_head(boundary_hidden)

        return result

    def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate resize call to the backbone."""
        return self.backbone.resize_token_embeddings(*args, **kwargs)


def load_student_tokenizer(config: StudentWrapperConfig):
    """Load and extend the student tokenizer."""
    tokenizer_source = _checkpoint_artifact_source(config, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
    )
    ensure_special_tokens(tokenizer, list(config.special_tokens))
    return tokenizer


def load_student_processor(config: StudentWrapperConfig, tokenizer=None):
    """Load the student multimodal processor with bounded pixel budgets."""
    if tokenizer is None:
        tokenizer = load_student_tokenizer(config)
    processor_source = _checkpoint_artifact_source(config, "processor")
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    processor.tokenizer = tokenizer
    return processor


def _resolve_backbone_loader(config: StudentWrapperConfig):
    base_config = AutoConfig.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
    )
    if getattr(base_config, "model_type", "") == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration, base_config
        return AutoModelForVision2Seq, base_config
    return AutoModelForCausalLM, base_config


def _hidden_size_from_config(config: Any) -> int:
    for attr in ("hidden_size", "n_embd"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    for attr in ("hidden_size", "n_embd"):
        value = getattr(text_config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Could not infer hidden size from student config.")


def build_student_model(config: StudentWrapperConfig, tokenizer) -> DistillStudentModel:
    """Load the student model and resize embeddings for special tokens."""
    model_cls, resolved_config = _resolve_backbone_loader(config)
    load_kwargs: dict[str, Any] = {
        "dtype": config.torch_dtype,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": _effective_local_files_only(config),
    }
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    backbone = model_cls.from_pretrained(
        config.student_model_name,
        **load_kwargs,
    )
    backbone.resize_token_embeddings(len(tokenizer))
    hidden_size = _hidden_size_from_config(getattr(backbone, "config", resolved_config))
    student = DistillStudentModel(
        backbone=backbone,
        hidden_size=hidden_size,
        num_action_classes=len(ACTION_CLASSES),
        traj_teacher_hidden_size=config.traj_teacher_hidden_size,
        traj_aux_num_buckets=config.traj_aux_num_buckets,
        traj_hidden_bridge_size=config.traj_hidden_bridge_size,
        boundary_action_head_hidden_size=config.boundary_action_head_hidden_size,
        boundary_action_head_dropout=config.boundary_action_head_dropout,
        flex_scene=config.flex_scene,
        image_token_id=getattr(backbone.config, "image_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    if config.use_vit_projection:
        student.configure_vit_projection(config.vit_in_dim, hidden_size)
    return student
