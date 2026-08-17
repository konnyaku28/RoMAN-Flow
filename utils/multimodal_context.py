from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SMOLVLM_MODEL_PATH = os.environ.get(
    'SMOLVLM_MODEL_PATH',
    '/path/to/SmolVLM-500M-Instruct',
)


class QuantileProprioceptionNormalizer(nn.Module):
    """Persist OpenVLA-style q01/q99 normalization with the checkpoint."""

    def __init__(self, q01, q99, eps: float = 1e-6):
        super().__init__()
        q01 = torch.as_tensor(q01, dtype=torch.float32)
        q99 = torch.as_tensor(q99, dtype=torch.float32)
        if q01.ndim != 1 or q99.shape != q01.shape:
            raise ValueError(
                f'Proprioception quantiles must be matching vectors, got {q01.shape}/{q99.shape}.'
            )
        if torch.any(q99 <= q01):
            raise ValueError('Every proprioception q99 value must be greater than q01.')
        self.register_buffer('q01', q01, persistent=True)
        self.register_buffer('q99', q99, persistent=True)
        self.eps = float(eps)

    def forward(self, proprioceptions: torch.Tensor) -> torch.Tensor:
        if proprioceptions.shape[-1] != self.q01.numel():
            raise ValueError(
                f'Expected {self.q01.numel()}-D proprioception, got {proprioceptions.shape[-1]}.'
            )
        q01 = self.q01.to(device=proprioceptions.device, dtype=proprioceptions.dtype)
        q99 = self.q99.to(device=proprioceptions.device, dtype=proprioceptions.dtype)
        normalized = 2.0 * (proprioceptions - q01) / (q99 - q01 + self.eps) - 1.0
        return normalized.clamp_(-1.0, 1.0)


class GatedProprioceptionFusion(nn.Module):
    """Fuse the latest normalized state into a vector visual representation."""

    def __init__(self, proprio_dim: int, feature_dim: int):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.proprio_dim, int(feature_dim)),
            nn.GELU(),
            nn.Linear(int(feature_dim), int(feature_dim)),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: torch.Tensor,
        proprioceptions: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f'Gated proprioception fusion expects features [B, D], got {features.shape}.'
            )
        if proprioceptions.ndim != 3 or proprioceptions.shape[0] != features.shape[0]:
            raise ValueError(
                'Gated proprioception fusion expects state [B, T, D] aligned with features; '
                f'got {proprioceptions.shape}/{features.shape}.'
            )
        if proprioceptions.shape[-1] != self.proprio_dim:
            raise ValueError(
                f'Expected {self.proprio_dim}-D state, got {proprioceptions.shape[-1]}.'
            )
        state_features = self.projector(proprioceptions[:, -1].to(dtype=features.dtype))
        return features + torch.tanh(self.gate) * state_features


def _enable_non_reentrant_gradient_checkpointing(model: nn.Module) -> None:
    """Enable DDP-compatible activation checkpointing for HuggingFace VLMs."""
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False},
        )
    except TypeError as exc:
        raise RuntimeError(
            'SmolVLM DDP training requires transformers gradient checkpointing '
            'with use_reentrant=False. Upgrade transformers or set '
            'vlm_gradient_checkpointing=False.'
        ) from exc


class SmolVLMContextEncoder(nn.Module):
    """SmolVLM visual-language fusion used as an actor context encoder."""

    def __init__(
        self,
        context_dim: int,
        model_path: str | None = None,
        image_size: int = 384,
        gradient_checkpointing: bool = True,
        proprio_dim: int | None = None,
        freeze_text_model: bool = False,
        image_normalization: str = 'processor',
        num_views: int = 2,
        max_horizon: int = 2,
        train_last_n_vision_layers: int = -1,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except (ImportError, RuntimeError) as exc:
            raise ImportError(
                'SmolVLM fusion requires compatible versions of transformers and '
                'huggingface-hub (transformers>=4.46,<5 and huggingface-hub<1).'
            ) from exc

        self.model_path = model_path or DEFAULT_SMOLVLM_MODEL_PATH
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f'SmolVLM model directory does not exist: {self.model_path}. '
                'Place the required SmolVLM-500M-Instruct files at that local path '
                'before enabling vlm_fuse.'
            )
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        self.max_horizon = int(max_horizon)
        if self.num_views <= 0 or self.max_horizon <= 0:
            raise ValueError(
                'SmolVLM view/time dimensions must be positive; '
                f'got num_views={self.num_views}, max_horizon={self.max_horizon}.'
            )
        self.backbone_frozen = False
        self.freeze_text_model = bool(freeze_text_model)
        self.train_last_n_vision_layers = int(train_last_n_vision_layers)
        if self.train_last_n_vision_layers < -1:
            raise ValueError(
                "train_last_n_vision_layers must be -1 (all) or a non-negative integer, "
                f"got {self.train_last_n_vision_layers}."
            )
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        if hasattr(self.vlm.config, 'use_cache'):
            self.vlm.config.use_cache = False
        if gradient_checkpointing and hasattr(self.vlm, 'gradient_checkpointing_enable'):
            _enable_non_reentrant_gradient_checkpointing(self.vlm)

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        image_processor = processor.image_processor
        self.image_normalization = str(image_normalization).strip().lower().replace('-', '_')
        if self.image_normalization == 'processor':
            image_mean = getattr(image_processor, 'image_mean', [0.5, 0.5, 0.5])
            image_std = getattr(image_processor, 'image_std', [0.5, 0.5, 0.5])
        elif self.image_normalization == 'imagenet':
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(
                'SmolVLM image_normalization must be `processor` or `imagenet`, '
                f'got {image_normalization!r}.'
            )
        self.register_buffer(
            'image_mean',
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'image_std',
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        text_config = getattr(self.vlm.config, 'text_config', None)
        hidden_size = int(getattr(text_config, 'hidden_size'))
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_projector = (
            nn.Sequential(
                nn.Linear(self.proprio_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.proprio_dim is not None
            else None
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, int(context_dim)),
        )
        self.set_backbone_frozen(False)

    @property
    def backbone(self):
        return self.vlm.model

    def set_backbone_frozen(self, frozen: bool) -> None:
        frozen = bool(frozen)
        self.backbone_frozen = frozen
        train_last_n = int(getattr(self, 'train_last_n_vision_layers', -1))
        for parameter in self.vlm.parameters():
            parameter.requires_grad_(not frozen and train_last_n == -1)
        if not frozen and train_last_n >= 0:
            connector = getattr(self.backbone, 'connector', None)
            if connector is None:
                connector = getattr(self.backbone, 'multi_modal_projector', None)
            if connector is None:
                raise AttributeError('SmolVLM backbone has no trainable visual connector.')
            for parameter in connector.parameters():
                parameter.requires_grad_(True)

            vision_model = self.backbone.vision_model
            encoder = getattr(vision_model, 'encoder', None)
            layers = getattr(encoder, 'layers', None)
            if layers is None:
                raise AttributeError(
                    'Partial SmolVLM tuning requires vision_model.encoder.layers.'
                )
            if train_last_n > len(layers):
                raise ValueError(
                    f'train_last_n_vision_layers={train_last_n} exceeds '
                    f'the {len(layers)} available vision layers.'
                )
            for layer in layers[len(layers) - train_last_n :]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            post_layernorm = getattr(vision_model, 'post_layernorm', None)
            if post_layernorm is not None:
                for parameter in post_layernorm.parameters():
                    parameter.requires_grad_(True)
            if not bool(getattr(self, 'freeze_text_model', False)):
                for parameter in self.backbone.text_model.parameters():
                    parameter.requires_grad_(True)
                lm_head = getattr(self.vlm, 'lm_head', None)
                if lm_head is not None:
                    for parameter in lm_head.parameters():
                        parameter.requires_grad_(True)
        if bool(getattr(self, 'freeze_text_model', False)):
            for parameter in self.backbone.text_model.parameters():
                parameter.requires_grad_(False)
            lm_head = getattr(self.vlm, 'lm_head', None)
            if lm_head is not None:
                for parameter in lm_head.parameters():
                    parameter.requires_grad_(False)

    def _prepare_images(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int]:
        if observations.ndim != 6:
            raise ValueError(
                f'SmolVLM expects observations [B, V, C, T, H, W], got {tuple(observations.shape)}.'
            )
        batch_size, num_views, channels, horizon, height, width = observations.shape
        if channels != 3:
            raise ValueError(f'SmolVLM requires RGB observations, got C={channels}.')
        images = observations.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size * num_views * horizon,
            channels,
            height,
            width,
        )
        images = images.float().div_(255.0)
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        images = (images - mean) / std
        return images, batch_size, num_views, horizon

    def forward(
        self,
        observations: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        proprioceptions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, batch_size, num_views, horizon = self._prepare_images(observations)
        backbone = self.backbone
        vision_model = backbone.vision_model
        vision_dtype = next(vision_model.parameters()).dtype
        vision_grad_context = (
            torch.no_grad() if self.backbone_frozen else contextlib.nullcontext()
        )
        with vision_grad_context:
            vision_outputs = vision_model(
                pixel_values=images.to(dtype=vision_dtype),
                output_hidden_states=False,
                return_dict=True,
            )
            image_features = vision_outputs.last_hidden_state
            if hasattr(backbone, 'connector'):
                image_features = backbone.connector(image_features)
            elif hasattr(backbone, 'multi_modal_projector'):
                image_features = backbone.multi_modal_projector(image_features)
            else:
                raise AttributeError('SmolVLM backbone has no connector or multi_modal_projector.')

        image_features = image_features.reshape(
            batch_size,
            num_views * horizon * image_features.shape[1],
            image_features.shape[2],
        )
        input_ids = input_ids.to(device=observations.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=observations.device, dtype=torch.bool)
        text_model = backbone.text_model
        text_embeddings = text_model.get_input_embeddings()(input_ids)
        image_features = image_features.to(dtype=text_embeddings.dtype)
        prefix_features = [image_features]
        prefix_masks = [
            torch.ones(
                image_features.shape[:2],
                device=observations.device,
                dtype=torch.bool,
            )
        ]
        if self.proprio_projector is not None:
            if proprioceptions is None:
                raise ValueError('State-conditioned SmolVLM requires proprioceptions.')
            if (
                proprioceptions.ndim != 3
                or proprioceptions.shape[0] != batch_size
                or proprioceptions.shape[-1] != self.proprio_dim
            ):
                raise ValueError(
                    'SmolVLM proprioceptions must have shape '
                    f'[B, T, {self.proprio_dim}], got {tuple(proprioceptions.shape)}.'
                )
            state_tokens = self.proprio_projector(
                proprioceptions[:, -1:].to(dtype=text_embeddings.dtype)
            )
            prefix_features.append(state_tokens)
            prefix_masks.append(
                torch.ones(
                    state_tokens.shape[:2],
                    device=observations.device,
                    dtype=torch.bool,
                )
            )
        elif proprioceptions is not None:
            raise ValueError('Proprioceptions were passed to a SmolVLM encoder without a state projector.')
        inputs_embeds = torch.cat([*prefix_features, text_embeddings], dim=1)
        context_mask = torch.cat([*prefix_masks, attention_mask], dim=1)
        # During the freeze window the VLM optimizer group has lr=0. Text-model
        # autograd remains enabled so gradients can reach the new state projector.
        text_outputs = text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=context_mask.long(),
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        fused_tokens = text_outputs.last_hidden_state
        context_tokens = self.output_projection(fused_tokens)
        return context_tokens, context_mask
