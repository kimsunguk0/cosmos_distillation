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
from src.model.flex_scene_encoder import FlexSceneConfig, FlexSceneEncoder
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
        self.flex_scene_encoder: FlexSceneEncoder | None = None
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
            return
        if self.flex_scene_config == flex_scene and self.flex_scene_encoder is not None:
            return
        self.flex_scene_config = flex_scene
        self.flex_scene_encoder = FlexSceneEncoder(flex_scene)

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

    def _qwen_visual_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        conditional = self._conditional_backbone()
        image_embeds, _ = conditional.get_image_features(pixel_values, image_grid_thw)
        return list(image_embeds)

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
    ) -> torch.Tensor:
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

        image_features = self._qwen_visual_features(pixel_values, image_grid_thw)
        sample_features = []
        for row_index in range(batch_size):
            start = row_index * images_per_sample
            end = start + images_per_sample
            sample_features.append(torch.cat(image_features[start:end], dim=0))
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
        scene_embeds = self.flex_scene_encoder(
            visual_tokens,
            camera_ids=camera_ids,
            relative_times=relative_times,
        ).to(
            device=input_ids.device,
            dtype=self.backbone.get_input_embeddings().weight.dtype,
        )

        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        image_mask = input_ids == int(self.image_token_id)
        expected_scene_tokens = int(scene_embeds.shape[1])
        counts = image_mask.sum(dim=1)
        if not bool(torch.all(counts == expected_scene_tokens).item()):
            raise ValueError(
                "Compressed image placeholder count must match FLEX scene tokens; "
                f"counts={counts.detach().cpu().tolist()}, expected={expected_scene_tokens}."
            )
        return inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            scene_embeds.reshape(-1),
        )

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
        if past_key_values is None:
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError("FLEX prefill requires pixel_values and image_grid_thw.")
            inputs_embeds = self._flex_inputs_embeds(
                input_ids,
                pixel_values,
                image_grid_thw,
                camera_indices=kwargs.pop("camera_indices", None),
                relative_timestamps=kwargs.pop("relative_timestamps", None),
                camera_counts=kwargs.pop("camera_counts", None),
                frames_per_camera=kwargs.pop("frames_per_camera", None),
            )
        else:
            inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        position_ids = self._position_ids_from_attention_mask(attention_mask, inputs_embeds.shape[1], inputs_embeds.device)
        language_outputs = conditional.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_hidden_states=need_hidden_states,
            return_dict=True,
            use_cache=bool(kwargs.get("use_cache", False)),
            cache_position=kwargs.get("cache_position"),
        )
        hidden_states = language_outputs.last_hidden_state
        logits = self._output_head()(hidden_states)

        class _FlexOutput:
            pass

        outputs = _FlexOutput()
        outputs.logits = logits
        outputs.past_key_values = getattr(language_outputs, "past_key_values", None)
        outputs.hidden_states = getattr(language_outputs, "hidden_states", None)
        if outputs.hidden_states is None and need_hidden_states:
            outputs.hidden_states = (hidden_states,)
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
    )
    ensure_special_tokens(tokenizer, list(config.special_tokens))
    return tokenizer


def load_student_processor(config: StudentWrapperConfig, tokenizer=None):
    """Load the student multimodal processor with bounded pixel budgets."""
    if tokenizer is None:
        tokenizer = load_student_tokenizer(config)
    processor = AutoProcessor.from_pretrained(
        config.student_model_name,
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
