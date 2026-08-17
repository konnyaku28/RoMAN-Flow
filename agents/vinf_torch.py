from __future__ import annotations

import math
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import ml_collections
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from utils.encoders import (
    TorchImpalaEncoder,
    TorchRoboMimicSpatialResNet18,
)
from utils.networks_torch import (
    ChunkTransformerValue,
    DiagGaussianPrior,
    LanguageStateFusion,
    VectorObservationEncoder,
    Value,
    batch_flatten,
    default_device,
    to_tensor,
)
from utils.simflow_networks import SimFlow
from utils.reverse import InContextReverseModel


TARGET_ONLY_PARAMETER_PREFIXES = (
    'target_critic.',
    'target_critic_value_state_encoder.',
    'target_critic_value_state_fusion.',
    'target_critic_value_proprio_fusion.',
    'reverse_model_ema.',
)
ONLINE_CRITIC_VALUE_PARAMETER_PREFIXES = (
    'critic.',
    'value.',
    'critic_value_state_encoder.',
    'critic_value_state_fusion.',
    'critic_value_proprio_fusion.',
)
ACTOR_CONTEXT_ADAPTER_PARAMETER_PREFIXES = (
    'actor_context_encoder.output_projection.',
    'actor_context_encoder.proprio_projector.',
)


def uses_proprioception(config) -> bool:
    return bool(
        config.get('use_proprioception', False)
        or config.get('robomimic_use_proprioception', False)
    )


@dataclass(frozen=True)
class ActorConditioning:
    y: torch.Tensor | None
    context_tokens: torch.Tensor | None = None
    context_attention_mask: torch.Tensor | None = None

    def kwargs(self) -> Dict[str, torch.Tensor | None]:
        return {
            'y': self.y,
            'context_tokens': self.context_tokens,
            'context_attention_mask': self.context_attention_mask,
        }

    def detach(self) -> 'ActorConditioning':
        return ActorConditioning(
            y=self.y.detach() if self.y is not None else None,
            context_tokens=(
                self.context_tokens.detach() if self.context_tokens is not None else None
            ),
            context_attention_mask=(
                self.context_attention_mask.detach()
                if self.context_attention_mask is not None
                else None
            ),
        )

    def slice(self, item) -> 'ActorConditioning':
        return ActorConditioning(
            y=self.y[item] if self.y is not None else None,
            context_tokens=(
                self.context_tokens[item] if self.context_tokens is not None else None
            ),
            context_attention_mask=(
                self.context_attention_mask[item]
                if self.context_attention_mask is not None
                else None
            ),
        )

def _clip_grad_norm_finite_(named_parameters, max_norm: float) -> torch.Tensor:
    named_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not named_parameters:
        return torch.tensor(0.0)

    parameters = [parameter for _, parameter in named_parameters]
    try:
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            float(max_norm),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        nonfinite_names = [
            name
            for name, parameter in named_parameters
            if not bool(torch.isfinite(parameter.grad.detach()).all().item())
        ]
        if nonfinite_names:
            raise RuntimeError(
                "Non-finite gradient elements detected during gradient clipping: "
                f"{nonfinite_names[:8]}."
            ) from exc

        device = parameters[0].grad.device
        total_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
        max_abs_by_name = []
        for name, parameter in named_parameters:
            grad = parameter.grad.detach()
            max_abs = grad.abs().max()
            max_abs_by_name.append((float(max_abs.cpu()), name))
            if not bool((max_abs > 0).item()):
                continue
            scaled_norm = torch.linalg.vector_norm((grad / max_abs).float(), ord=2)
            parameter_norm = max_abs.double() * scaled_norm.double()
            total_norm_sq = total_norm_sq + parameter_norm.square()

        total_norm = total_norm_sq.sqrt()
        if not bool(torch.isfinite(total_norm).item()):
            raise RuntimeError(
                "Gradient elements are finite, but the stable float64 gradient norm is non-finite."
            ) from exc

        clip_coefficient = (
            torch.tensor(float(max_norm), device=device, dtype=torch.float64)
            / (total_norm + 1e-6)
        ).clamp(max=1.0)
        for _, parameter in named_parameters:
            parameter.grad.mul_(clip_coefficient.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            largest = sorted(max_abs_by_name, reverse=True)[:8]
            print(
                "[stable-grad-clip] PyTorch float32 norm overflowed while every gradient element "
                f"was finite. Stable pre-clip norm={float(total_norm.cpu()):.6e}; "
                f"largest gradients={largest}.",
                flush=True,
            )
        return total_norm


def _parse_hw_tuple(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected adaptive pool size as `H,W`, got `{value}`.")
        return (int(parts[0]), int(parts[1]))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return (int(value), int(value))
    raise ValueError(f"Unsupported adaptive pool size: {value!r}")


def build_torch_encoder(
    name: str,
    config=None,
) -> nn.Module:
    config = config or {}
    image_num_views = int(config.get('image_num_views', 2))
    impala_adaptive_pool_hw = _parse_hw_tuple(config.get('impala_adaptive_pool_hw', None))
    if name == 'robomimic_spatial_resnet18':
        return TorchRoboMimicSpatialResNet18(
            num_views=image_num_views,
            proprio_dim=int(config.get('robomimic_proprio_dim', 9)),
            use_proprioception=bool(
                config.get('robomimic_use_proprioception', False)
            ),
            use_augmentation=bool(
                config.get('robomimic_use_crop_augmentation', False)
            ),
            crop_size=int(config.get('robomimic_crop_size', 76)),
            output_size=int(config.get('robomimic_resnet_input_size', 224)),
            max_obs_horizon=int(config.get('robomimic_max_obs_horizon', 8)),
            pretrained_weights=config.get(
                'robomimic_resnet_pretrained_weights',
                'IMAGENET1K_V1',
            ),
        )
    if name == 'impala':
        return TorchImpalaEncoder(
            in_channels=3,
            adaptive_pool_hw=impala_adaptive_pool_hw,
        )
    raise ValueError(
        f'Unsupported encoder {name!r}. RoMAN-Flow supports only `impala` and '
        '`robomimic_spatial_resnet18`.'
    )


def validate_release_architecture(config) -> str:
    """Validate the two model architectures supported by this release."""
    conditioning_mode = str(config.get('conditioning_mode', '')).strip().lower()
    expected_encoder = {
        'context': 'impala',
        'vision_context': 'robomimic_spatial_resnet18',
    }.get(conditioning_mode)
    if expected_encoder is None:
        raise ValueError(
            f'Unsupported conditioning_mode={conditioning_mode!r}; expected '
            '`context` or `vision_context`.'
        )

    for key in ('actor_encoder', 'critic_encoder'):
        if str(config.get(key, '')).strip().lower() != expected_encoder:
            raise ValueError(
                f'{conditioning_mode} conditioning requires '
                f'{key}=`{expected_encoder}`.'
            )

    uses_actor_language = bool(config.get('use_language_conditioning', False))
    uses_state_language = (
        bool(config.get('critic_use_language_conditioning', False))
        or bool(config.get('value_use_language_conditioning', False))
    )
    vlm_fuse = bool(config.get('vlm_fuse', False))
    if conditioning_mode == 'context':
        if not vlm_fuse or not uses_actor_language:
            raise ValueError(
                'LIBERO context conditioning requires vlm_fuse=True and '
                'use_language_conditioning=True.'
            )
    elif vlm_fuse or uses_actor_language or uses_state_language:
        raise ValueError(
            'RoboMimic vision_context conditioning requires VLM and all language '
            'conditioning to be disabled.'
        )
    return conditioning_mode


class VINFModule(nn.Module):
    def __init__(
        self,
        ex_observations,
        ex_actions,
        config,
        ex_proprioceptions=None,
    ):
        super().__init__()
        action_dim = ex_actions.shape[-1]
        action_len = ex_actions.shape[-2] if ex_actions.ndim > 2 else int(config['action_len'])
        conditioning_mode = validate_release_architecture(config)
        vlm_fuse = bool(config.get('vlm_fuse', False))
        self.conditioning_mode = conditioning_mode
        self.vlm_fuse = vlm_fuse
        self.actor_context_obs_horizon = self._resolve_actor_context_obs_horizon(
            config,
            conditioning_mode,
            vlm_fuse,
        )
        self.use_context_proprioception = bool(config.get('use_proprioception', False))
        if self.use_context_proprioception:
            if conditioning_mode != 'context' or not vlm_fuse:
                raise ValueError(
                    'use_proprioception=True requires context conditioning with vlm_fuse=True.'
                )
            injection = str(
                config.get('proprio_injection', 'vlm_context_token')
            ).strip().lower()
            if injection != 'vlm_context_token':
                raise ValueError(
                    f'Unsupported proprio_injection={injection!r}; expected `vlm_context_token`.'
                )
            from utils.multimodal_context import QuantileProprioceptionNormalizer

            proprio_dim = int(config.get('proprio_dim', 8))
            self.proprio_normalizer = QuantileProprioceptionNormalizer(
                config.get('proprio_q01', [-1.0] * proprio_dim),
                config.get('proprio_q99', [1.0] * proprio_dim),
            )
        else:
            self.proprio_normalizer = None
        actor_encoder_name = str(config['actor_encoder']).strip().lower()
        critic_encoder_name = str(config['critic_encoder']).strip().lower()
        actor_encoder_module = build_torch_encoder(actor_encoder_name, config)
        critic_encoder_module = build_torch_encoder(critic_encoder_name, config)
        self.actor_encoder_name = actor_encoder_name
        self.critic_encoder_name = critic_encoder_name
        def infer_obs_dim(encoder_module):
            if encoder_module is None:
                return ex_observations.shape[-1] if ex_observations.ndim <= 2 else int(np.prod(ex_observations.shape[1:]))
            with torch.no_grad():
                _dummy_obs = to_tensor(ex_observations)
                _dummy_proprio = (
                    to_tensor(ex_proprioceptions)
                    if (
                        ex_proprioceptions is not None
                        and bool(config.get('robomimic_use_proprioception', False))
                    )
                    else None
                )
                # Invoke the real encoder so lazily constructed heads are registered.
                if _dummy_proprio is None:
                    return encoder_module(_dummy_obs).shape[-1]
                return encoder_module(
                    _dummy_obs,
                    proprioceptions=_dummy_proprio,
                ).shape[-1]

        actor_obs_dim = infer_obs_dim(actor_encoder_module)
        critic_obs_dim = infer_obs_dim(critic_encoder_module)
        critic_use_language_conditioning = bool(config.get('critic_use_language_conditioning', False))
        value_use_language_conditioning = bool(config.get('value_use_language_conditioning', False))
        critic_language_embed_dim = (
            int(config.get('critic_language_embed_dim', 512))
            if critic_use_language_conditioning
            else None
        )
        value_language_embed_dim = (
            int(config.get('value_language_embed_dim', 512))
            if value_use_language_conditioning
            else None
        )
        share_critic_value_state = bool(config.get('share_critic_value_state', False))
        self.share_critic_value_state = share_critic_value_state
        independent_proprio_dim = (
            int(config.get('proprio_dim', 8))
            if self.use_context_proprioception
            and not share_critic_value_state
            else None
        )
        self.critic_value_state_encoder = None
        self.critic_value_state_fusion = None
        self.target_critic_value_state_encoder = None
        self.target_critic_value_state_fusion = None
        self.critic_value_proprio_fusion = None
        self.target_critic_value_proprio_fusion = None
        if critic_use_language_conditioning or value_use_language_conditioning:
            from utils.language import CLIPTextFeatureExtractor

            self.state_text_encoder = CLIPTextFeatureExtractor(
                model_path=config.get('language_model_path', None),
            )
        else:
            self.state_text_encoder = None

        if share_critic_value_state:
            if not (critic_use_language_conditioning and value_use_language_conditioning):
                raise ValueError(
                    'share_critic_value_state=True requires both '
                    'critic_use_language_conditioning=True and '
                    'value_use_language_conditioning=True.'
                )
            if critic_language_embed_dim != value_language_embed_dim:
                raise ValueError(
                    'share_critic_value_state=True requires critic_language_embed_dim '
                    'to equal value_language_embed_dim.'
                )
            self.critic_value_state_encoder = critic_encoder_module
            self.critic_value_state_fusion = LanguageStateFusion(
                obs_dim=critic_obs_dim,
                language_dim=critic_language_embed_dim,
                layer_norm=config['layer_norm'],
            )
            if self.use_context_proprioception:
                from utils.multimodal_context import GatedProprioceptionFusion

                self.critic_value_proprio_fusion = GatedProprioceptionFusion(
                    proprio_dim=int(config.get('proprio_dim', 8)),
                    feature_dim=critic_obs_dim,
                )
        critic_encoder_for_head = None if share_critic_value_state else critic_encoder_module
        critic_language_embed_for_head = None if share_critic_value_state else critic_language_embed_dim
        self.critic = ChunkTransformerValue(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            action_len=action_len,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=critic_encoder_for_head,
            hidden_dim=int(config.get('critic_hidden_dim', 512)),
            num_layers=int(config.get('critic_num_layers', 2)),
            num_heads=int(config.get('critic_num_heads', 8)),
            dropout=float(config.get('critic_dropout', 0.0)),
            prefix_reduce=str(config.get('critic_prefix_reduce', 'last')),
            language_embed_dim=critic_language_embed_for_head,
            proprio_dim=independent_proprio_dim,
        )
        value_encoder_module = (
            None
            if share_critic_value_state
            else copy.deepcopy(critic_encoder_module) if critic_encoder_module is not None else None
        )
        value_language_embed_for_head = None if share_critic_value_state else value_language_embed_dim
        self.value = Value(
            input_dim=critic_obs_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=value_encoder_module,
            language_embed_dim=value_language_embed_for_head,
            language_obs_dim=critic_obs_dim,
            proprio_dim=independent_proprio_dim,
        )
        self.target_critic = copy.deepcopy(self.critic)
        if share_critic_value_state:
            self.target_critic_value_state_encoder = (
                copy.deepcopy(self.critic_value_state_encoder)
                if self.critic_value_state_encoder is not None
                else None
            )
            self.target_critic_value_state_fusion = copy.deepcopy(self.critic_value_state_fusion)
            self.target_critic_value_proprio_fusion = copy.deepcopy(
                self.critic_value_proprio_fusion
            )
        self.actor_token_projection = None
        if conditioning_mode == 'vision_context':
            if not isinstance(actor_encoder_module, TorchRoboMimicSpatialResNet18):
                raise ValueError(
                    'vision_context conditioning requires '
                    'actor_encoder=`robomimic_spatial_resnet18`.'
                )
            self.actor_encoder = actor_encoder_module
            self.actor_token_projection = nn.Sequential(
                nn.Linear(actor_obs_dim, int(config['bc_channels'])),
                nn.LayerNorm(int(config['bc_channels'])),
            )
        else:
            self.actor_encoder = VectorObservationEncoder(
                input_size=actor_obs_dim,
                rep_size=config['bc_rep_size'],
                encoder=actor_encoder_module,
            )

        self.actor_context_encoder = None
        if conditioning_mode == 'context':
            if not bool(config.get('use_language_conditioning', False)):
                raise ValueError(
                    'conditioning_mode=`context` requires use_language_conditioning=True.'
                )
            if not vlm_fuse:
                raise ValueError('LIBERO context conditioning requires vlm_fuse=True.')
            from utils.multimodal_context import SmolVLMContextEncoder

            self.actor_context_encoder = SmolVLMContextEncoder(
                context_dim=int(config['bc_channels']),
                model_path=config.get('smolvlm_model_path', None),
                image_size=int(config.get('vlm_image_size', 384)),
                gradient_checkpointing=bool(config.get('vlm_gradient_checkpointing', True)),
                freeze_text_model=bool(config.get('vlm_freeze_text_model', False)),
                image_normalization=str(
                    config.get('vlm_image_normalization', 'processor')
                ),
                num_views=int(ex_observations.shape[1]),
                max_horizon=int(ex_observations.shape[3]),
                train_last_n_vision_layers=int(
                    config.get('vlm_train_last_n_vision_layers', -1)
                ),
                proprio_dim=(
                    int(config.get('proprio_dim', 8))
                    if self.use_context_proprioception
                    else None
                ),
            )
        
        self.actor = SimFlow(
            in_channels=action_dim,
            patch_size=1,
            channels=config['bc_channels'],
            num_blocks=config['bc_num_blocks'],
            layers_per_block=config['layers_per_block'],
            num_heads=config['num_heads'],
            action_len=config['action_len'],
            label_drop_prob=config['label_drop_prob'],
            obs_cond_channels=int(config['bc_rep_size']),
            log_scale_clip=float(config.get('simflow_log_scale_clip', 5.0)),
        )
        if config.get('use_biflow', False):
            self.reverse_model = InContextReverseModel(
                action_len=action_len,
                in_channels=action_dim,
                cond_channels=config['bc_rep_size'],
                channels=int(config.get('biflow_channels', 512)),
                num_layers=int(config.get('biflow_num_layers', 2)),
                num_blocks=int(config['bc_num_blocks']) + 1,
                num_heads=int(config.get('biflow_num_heads', 8)),
                label_drop_rate=float(config.get('biflow_label_drop_rate', 0.0)),
                num_condition_tokens=int(config.get('biflow_num_condition_tokens', 1)),
                num_guidance_tokens=int(config.get('biflow_num_guidance_tokens', 1)),
                context_dim=int(config['bc_channels']),
            )
            if bool(config.get('biflow_use_ema', False)):
                self.reverse_model_ema = copy.deepcopy(self.reverse_model)
                for parameter in self.reverse_model_ema.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _resolve_actor_context_obs_horizon(
        config,
        conditioning_mode: str,
        vlm_fuse: bool,
    ) -> int:
        horizon = int(config.get('actor_context_obs_horizon', 0))
        if horizon < 0:
            raise ValueError(
                'actor_context_obs_horizon must be nonnegative; '
                f'got {horizon}.'
            )
        if horizon == 0:
            return 0
        if conditioning_mode != 'context':
            raise ValueError(
                'actor_context_obs_horizon is only supported for context conditioning.'
            )
        if not vlm_fuse:
            raise ValueError(
                'actor_context_obs_horizon is only supported for SmolVLM context '
                'conditioning because CLIP context checkpoints use a fixed visual-token count.'
            )
        obs_horizon = int(config.get('obs_horizon', 0))
        if horizon > obs_horizon:
            raise ValueError(
                'actor_context_obs_horizon cannot exceed obs_horizon; '
                f'got actor_context_obs_horizon={horizon}, obs_horizon={obs_horizon}.'
            )
        return horizon

    def _actor_context_observations(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        horizon = self.actor_context_obs_horizon
        if horizon == 0:
            return observations
        if observations.ndim != 6:
            raise ValueError(
                'Actor context frame selection expects image observations '
                f'[B, V, C, T, H, W], got {tuple(observations.shape)}.'
            )
        available_horizon = int(observations.shape[3])
        if horizon > available_horizon:
            raise ValueError(
                'actor_context_obs_horizon exceeds the available observation frames; '
                f'got {horizon} requested and {available_horizon} available.'
            )
        return observations[:, :, :, -horizon:, :, :]

    def build_actor_conditioning(
        self,
        observations: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
    ) -> ActorConditioning:
        if self.use_context_proprioception:
            if proprioceptions is None:
                raise ValueError('State-conditioned context requires proprioceptions.')
            proprioceptions = self.proprio_normalizer(proprioceptions.float())
        if self.conditioning_mode == 'vision_context':
            if self.actor_token_projection is None:
                raise RuntimeError('Vision-context projection is missing.')
            visual_tokens = self.actor_encoder.forward_tokens(
                observations,
                proprioceptions=proprioceptions,
            )
            context_tokens = self.actor_token_projection(visual_tokens)
            return ActorConditioning(
                y=None,
                context_tokens=context_tokens,
                context_attention_mask=torch.ones(
                    context_tokens.shape[:2],
                    device=context_tokens.device,
                    dtype=torch.bool,
                ),
            )
        if input_ids is None or attention_mask is None:
            raise ValueError('Context conditioning requires language input_ids and attention_mask.')
        if self.actor_context_encoder is None:
            raise RuntimeError('Context conditioning is enabled but actor_context_encoder is missing.')
        context_observations = self._actor_context_observations(observations)
        context_tokens, context_mask = self.actor_context_encoder(
            context_observations,
            input_ids,
            attention_mask,
            proprioceptions=proprioceptions,
        )
        return ActorConditioning(
            y=None,
            context_tokens=context_tokens,
            context_attention_mask=context_mask,
        )

    def encode_critic_value_state(
        self,
        observations: torch.Tensor,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        if not self.share_critic_value_state:
            return observations
        encoder = (
            self.target_critic_value_state_encoder
            if target
            else self.critic_value_state_encoder
        )
        fusion = (
            self.target_critic_value_state_fusion
            if target
            else self.critic_value_state_fusion
        )
        obs_feat = encoder(observations) if encoder is not None else batch_flatten(observations.float())
        obs_feat = fusion(obs_feat, language_embed) if fusion is not None else obs_feat
        proprio_fusion = (
            self.target_critic_value_proprio_fusion
            if target
            else self.critic_value_proprio_fusion
        )
        if proprio_fusion is not None:
            if proprioceptions is None:
                raise ValueError('State-conditioned critic/value requires proprioceptions.')
            normalized = self.proprio_normalizer(proprioceptions.float())
            obs_feat = proprio_fusion(obs_feat, normalized)
        return obs_feat

    @staticmethod
    def _soft_update_module(source: nn.Module | None, target: nn.Module | None, tau: float):
        if source is None or target is None:
            return
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)

    def soft_update_target(self, tau: float):
        with torch.no_grad():
            self._soft_update_module(self.critic, self.target_critic, tau)
            self._soft_update_module(
                self.critic_value_state_encoder,
                self.target_critic_value_state_encoder,
                tau,
            )
            self._soft_update_module(
                self.critic_value_state_fusion,
                self.target_critic_value_state_fusion,
                tau,
            )
            self._soft_update_module(
                self.critic_value_proprio_fusion,
                self.target_critic_value_proprio_fusion,
                tau,
            )

    def soft_update_reverse_model_ema(self, decay: float):
        if not hasattr(self, 'reverse_model') or not hasattr(self, 'reverse_model_ema'):
            return
        with torch.no_grad():
            for source, target in zip(self.reverse_model.parameters(), self.reverse_model_ema.parameters()):
                target.data.mul_(decay).add_(source.data, alpha=1.0 - decay)
            for source, target in zip(self.reverse_model.buffers(), self.reverse_model_ema.buffers()):
                target.copy_(source)

    def forward(self, loss_fn, *args, **kwargs):
        """Run the agent loss inside DDP.forward so reducer hooks are active."""
        return loss_fn(*args, **kwargs)


@dataclass
class VINFTorchAgent:
    model: nn.Module
    prior: DiagGaussianPrior
    config: Dict[str, Any]
    device: torch.device
    scaler: torch.amp.GradScaler | None = None
    actor_optimizer: torch.optim.Optimizer | None = None
    critic_optimizer: torch.optim.Optimizer | None = None
    update_step: int = 0

    @staticmethod
    def _to_model_obs_layout(obs_seq):
        if torch.is_tensor(obs_seq):
            return obs_seq.permute(0, 2, 3, 1, 4, 5)
        return np.transpose(obs_seq, (0, 2, 3, 1, 4, 5))

    @staticmethod
    def _get_sequence_spec(config) -> Tuple[int, int, int]:
        obs_horizon = int(config.get('obs_horizon', 0))
        action_horizon = int(config.get('action_len', 0))
        if obs_horizon <= 0 or action_horizon <= 0:
            raise ValueError(
                f'Both obs_horizon and action_len must be positive, got '
                f'obs_horizon={obs_horizon}, action_len={action_horizon}.'
            )
        sequence_len = obs_horizon + action_horizon
        return obs_horizon, action_horizon, sequence_len

    @staticmethod
    def _get_sequence_time_axis(observations, sequence_len: int) -> int | None:
        if getattr(observations, 'ndim', 0) != 6:
            return None
        matched_axes = [axis for axis in (1, 3) if int(observations.shape[axis]) == sequence_len]
        if len(matched_axes) > 1:
            raise ValueError(
                f'Ambiguous observation layout for shape {tuple(observations.shape)}: '
                f'both axis 1 and axis 3 match expected sequence length {sequence_len}.'
            )
        return matched_axes[0] if matched_axes else None

    @staticmethod
    def _slice_sequence_aux_fields(
        out: Dict[str, torch.Tensor],
        sequence_len: int,
        action_start: int,
        action_end: int,
        action_horizon: int,
    ):
        if 'actions' in out and out['actions'].ndim >= 2:
            if out['actions'].shape[1] == sequence_len:
                out['actions'] = out['actions'][:, action_start:action_end]
            elif out['actions'].shape[1] != action_horizon:
                raise ValueError(
                    f"actions shape {tuple(out['actions'].shape)} is inconsistent with "
                    f"sequence_len={sequence_len} and action_horizon={action_horizon}."
                )

        for key in (
            'rewards',
            'terminals',
            'masks',
            'hubl_rewards',
            'hubl_discounts',
            'action_valid_mask',
        ):
            if key not in out or out[key].ndim < 2:
                continue
            if out[key].shape[1] == sequence_len:
                out[key] = out[key][:, action_start:action_end]
            elif out[key].shape[1] not in (1, action_horizon):
                raise ValueError(
                    f"{key} shape {tuple(out[key].shape)} is inconsistent with "
                    f"sequence_len={sequence_len} and action_horizon={action_horizon}."
                )

        if 'mc_returns' in out and out['mc_returns'].ndim >= 2:
            if out['mc_returns'].shape[1] == sequence_len:
                out['mc_returns'] = out['mc_returns'][:, action_start]
            elif out['mc_returns'].shape[1] != 1:
                raise ValueError(
                    f"mc_returns shape {tuple(out['mc_returns'].shape)} is inconsistent with "
                    f"sequence_len={sequence_len}."
                )

        if 'hubl_lambda' in out and out['hubl_lambda'].ndim >= 2:
            if out['hubl_lambda'].shape[1] == sequence_len:
                out['hubl_lambda'] = out['hubl_lambda'][:, action_start]
            elif out['hubl_lambda'].shape[1] != 1:
                raise ValueError(
                    f"hubl_lambda shape {tuple(out['hubl_lambda'].shape)} is inconsistent with "
                    f"sequence_len={sequence_len}."
                )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(0)
        try:
            expanded_mask = mask.expand_as(values)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} mask shape {tuple(mask.shape)} cannot broadcast to {tuple(values.shape)}."
            ) from exc
        denominator = expanded_mask.sum()
        if not bool((denominator > 0).item()):
            raise ValueError(f"{name} has no valid elements after applying action_valid_mask.")
        return (values * expanded_mask).sum() / denominator

    @staticmethod
    def _get_action_valid_mask(batch, actions):
        mask = batch.get('action_valid_mask')
        if mask is None:
            return torch.ones(actions.shape[:2], device=actions.device, dtype=torch.float32)
        mask = mask.to(device=actions.device, dtype=torch.float32)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        if tuple(mask.shape) != tuple(actions.shape[:2]):
            raise ValueError(
                f"action_valid_mask shape {tuple(mask.shape)} must match action time axes "
                f"{tuple(actions.shape[:2])}."
            )
        return mask

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        device=None,
        ex_proprioceptions=None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = device or default_device()
        model_ex_observations = ex_observations
        model_ex_actions = ex_actions
        model_ex_proprioceptions = ex_proprioceptions
        obs_horizon, action_horizon, sequence_len = cls._get_sequence_spec(config)

        if getattr(ex_observations, 'ndim', 0) == 6 and getattr(ex_actions, 'ndim', 0) > 1:
            time_axis = cls._get_sequence_time_axis(ex_observations, sequence_len)
            if time_axis is not None:
                action_start = obs_horizon - 1
                action_end = action_start + action_horizon
                if time_axis == 1:
                    model_ex_observations = cls._to_model_obs_layout(ex_observations[:, :obs_horizon])
                else:
                    model_ex_observations = ex_observations[:, :, :, :obs_horizon]

                if ex_proprioceptions is not None:
                    if (
                        getattr(ex_proprioceptions, 'ndim', 0) != 3
                        or ex_proprioceptions.shape[1] != sequence_len
                    ):
                        raise ValueError(
                            'ex_proprioceptions must have shape [B, sequence, D] '
                            f'aligned with images, got {tuple(ex_proprioceptions.shape)}.'
                        )
                    model_ex_proprioceptions = ex_proprioceptions[:, :obs_horizon]

                if ex_actions.ndim >= 3:
                    if ex_actions.shape[1] == sequence_len:
                        model_ex_actions = ex_actions[:, action_start:action_end]
                    elif ex_actions.shape[1] != action_horizon:
                        raise ValueError(
                            f"ex_actions shape {tuple(ex_actions.shape)} is inconsistent with "
                            f"sequence_len={sequence_len} and action_horizon={action_horizon}."
                        )
            elif ex_actions.ndim >= 3 and ex_actions.shape[1] == sequence_len:
                raise ValueError(
                    f"Expected a contiguous observation sequence with time axis length {sequence_len}, "
                    f"but got observations shape {tuple(ex_observations.shape)}."
                )
        prior = DiagGaussianPrior(dim=ex_actions.shape[-1], device=device, action_len=config['action_len'])
        use_amp = bool(config.get('use_amp', False)) and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        if uses_proprioception(config) and model_ex_proprioceptions is None:
            raise ValueError(
                'State conditioning requires example proprioceptions.'
            )
        model = VINFModule(
            model_ex_observations,
            model_ex_actions,
            config,
            ex_proprioceptions=model_ex_proprioceptions,
        ).to(device)
        agent = cls(model=model, prior=prior, config=dict(config), device=device, scaler=scaler)
        agent.reset_optimizer_for_train_mode(config.get('train_mode', 'rl'))
        return agent

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                raise TypeError(f'Nested dict batch is not supported in torch agent for key {k}')
            tensor = to_tensor(v, device=self.device)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            out[k] = tensor

        observations = out.get('observations')
        if observations is not None and 'next_observations' not in out and observations.ndim == 6:
            obs_horizon, action_horizon, sequence_len = self._get_sequence_spec(self.config)
            time_axis = self._get_sequence_time_axis(observations, sequence_len)
            if time_axis is None:
                raise ValueError(
                    "Unable to build next_observations from observations without an explicit contiguous "
                    f"sequence. observations shape={tuple(observations.shape)}, "
                    f"actions shape={tuple(out['actions'].shape) if 'actions' in out else None}, "
                    f"obs_horizon={obs_horizon}, action_horizon={action_horizon}."
                )

            action_start = obs_horizon - 1
            action_end = action_start + action_horizon
            prefix_next_obs = []
            next_obs_start = action_horizon
            if time_axis == 1:
                for prefix_idx in range(action_horizon):
                    prefix_start = prefix_idx + 1
                    prefix_next_obs.append(
                        self._to_model_obs_layout(observations[:, prefix_start : prefix_start + obs_horizon])
                    )
                out['observations'] = self._to_model_obs_layout(observations[:, :obs_horizon])
                out['next_observations'] = self._to_model_obs_layout(
                    observations[:, next_obs_start : next_obs_start + obs_horizon]
                )
            else:
                for prefix_idx in range(action_horizon):
                    prefix_start = prefix_idx + 1
                    prefix_next_obs.append(
                        observations[:, :, :, prefix_start : prefix_start + obs_horizon]
                    )
                out['observations'] = observations[:, :, :, :obs_horizon]
                out['next_observations'] = observations[:, :, :, next_obs_start : next_obs_start + obs_horizon]
            out['prefix_next_observations'] = torch.stack(prefix_next_obs, dim=1)

            proprioceptions = out.get('proprioceptions')
            if proprioceptions is not None:
                if (
                    proprioceptions.ndim != 3
                    or proprioceptions.shape[1] != sequence_len
                ):
                    raise ValueError(
                        'proprioceptions must have shape [B, sequence, D] aligned '
                        f'with observations, got {tuple(proprioceptions.shape)}.'
                    )
                prefix_next_proprioceptions = []
                for prefix_idx in range(action_horizon):
                    prefix_start = prefix_idx + 1
                    prefix_next_proprioceptions.append(
                        proprioceptions[
                            :,
                            prefix_start : prefix_start + obs_horizon,
                        ]
                    )
                out['proprioceptions'] = proprioceptions[:, :obs_horizon]
                out['next_proprioceptions'] = proprioceptions[
                    :,
                    next_obs_start : next_obs_start + obs_horizon,
                ]
                out['prefix_next_proprioceptions'] = torch.stack(
                    prefix_next_proprioceptions,
                    dim=1,
                )
            self._slice_sequence_aux_fields(out, sequence_len, action_start, action_end, action_horizon)
        return out

    @property
    def raw_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _biflow_reverse_num_blocks(self) -> int:
        raw_model = self.raw_model
        if self._use_biflow() and hasattr(raw_model, 'reverse_model'):
            return int(raw_model.reverse_model.num_blocks)
        return int(self.config.get('bc_num_blocks', 0)) + 1

    def _biflow_sampling_reverse_model(self, raw_model: nn.Module | None = None) -> nn.Module:
        raw_model = self.raw_model if raw_model is None else raw_model
        if (
            bool(self.config.get('biflow_sample_use_ema', self.config.get('biflow_use_ema', False)))
            and hasattr(raw_model, 'reverse_model_ema')
        ):
            return raw_model.reverse_model_ema
        return raw_model.reverse_model

    def _make_biflow_guidance_schedule(
        self,
        guidance: float | torch.Tensor,
        batch_size: int,
        *,
        final_guidance: float | torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Build BiFlow's learned-inverse guidance vector.

        The original BiFlow sampler passes one guidance value per reverse
        block: `[guidance] * forward_num_blocks + [guidance_diff]`.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.float32
        num_blocks = self._biflow_reverse_num_blocks()
        if torch.is_tensor(guidance):
            base = guidance.to(device=device, dtype=dtype)
        else:
            base = torch.tensor(float(guidance), device=device, dtype=dtype)
        if final_guidance is None:
            final = base
        elif torch.is_tensor(final_guidance):
            final = final_guidance.to(device=device, dtype=dtype)
        else:
            final = torch.tensor(float(final_guidance), device=device, dtype=dtype)

        if base.ndim == 0:
            base = base.repeat(batch_size)
        elif base.ndim == 1 and base.shape[0] == 1 and batch_size != 1:
            base = base.repeat(batch_size)
        elif base.ndim != 1 or base.shape[0] != batch_size:
            raise ValueError(f'guidance must be scalar or shape [{batch_size}], got {tuple(base.shape)}.')

        if final.ndim == 0:
            final = final.repeat(batch_size)
        elif final.ndim == 1 and final.shape[0] == 1 and batch_size != 1:
            final = final.repeat(batch_size)
        elif final.ndim != 1 or final.shape[0] != batch_size:
            raise ValueError(f'final_guidance must be scalar or shape [{batch_size}], got {tuple(final.shape)}.')

        schedule = base[None, :].repeat(num_blocks, 1)
        schedule[-1] = final
        return schedule

    def _get_biflow_guidance_final(self, key: str, base_guidance: float) -> float:
        value = self.config.get(key, -1.0)
        if value is None:
            return float(base_guidance)
        value = float(value)
        return float(base_guidance) if value < 0.0 else value

    def _train_critic_during_biflow_align(self, train_mode: str | None = None) -> bool:
        train_mode = str(train_mode or self.config.get('train_mode', 'rl')).lower()
        return train_mode == 'biflow_align' and bool(self.config.get('biflow_align_train_critic', False))

    def _train_critic_value_during_il(self, train_mode: str | None = None) -> bool:
        train_mode = str(train_mode or self.config.get('train_mode', 'rl')).lower()
        return train_mode == 'il' and bool(self.config.get('il_train_critic_value', False))

    def _iter_trainable_parameters_for_mode(self, train_mode: str | None = None):
        train_mode = str(train_mode or self.config.get('train_mode', 'rl')).lower()
        raw_model = self.raw_model
        if train_mode == 'biflow_align':
            if not hasattr(raw_model, 'reverse_model'):
                raise ValueError('train_mode=`biflow_align` requires `reverse_model`.')
            for name, parameter in raw_model.reverse_model.named_parameters():
                if '.language_model.' in name:
                    continue
                yield parameter
            if self._train_critic_during_biflow_align(train_mode):
                yield from raw_model.critic.parameters()
                yield from raw_model.value.parameters()
                if getattr(raw_model, 'critic_value_state_encoder', None) is not None:
                    yield from raw_model.critic_value_state_encoder.parameters()
                if getattr(raw_model, 'critic_value_state_fusion', None) is not None:
                    yield from raw_model.critic_value_state_fusion.parameters()
            return

        for name, parameter in raw_model.named_parameters():
            if name.startswith(TARGET_ONLY_PARAMETER_PREFIXES):
                continue
            if '.language_model.' in name:
                continue
            yield parameter

    def configure_trainable_parameters(self, train_mode: str | None = None):
        """Set requires_grad to match the active training stage."""
        train_mode = str(train_mode or self.config.get('train_mode', 'rl')).lower()
        raw_model = self.raw_model
        for name, parameter in raw_model.named_parameters():
            parameter.requires_grad = (
                not name.startswith(TARGET_ONLY_PARAMETER_PREFIXES)
                and '.language_model.' not in name
            )
        context_encoder = getattr(raw_model, 'actor_context_encoder', None)
        if context_encoder is not None and hasattr(context_encoder, 'set_backbone_frozen'):
            context_encoder.set_backbone_frozen(False)

        if (
            train_mode == 'biflow_align'
            and bool(
                self.config.get(
                    'biflow_align_freeze_teacher_requires_grad',
                    self.config.get('biflow_freeze_teacher_in_align', True),
                )
            )
        ):
            if not hasattr(raw_model, 'reverse_model'):
                raise ValueError('train_mode=`biflow_align` requires `reverse_model`.')
            for parameter in raw_model.parameters():
                parameter.requires_grad = False
            for parameter in raw_model.reverse_model.parameters():
                parameter.requires_grad = True
            if self._train_critic_during_biflow_align(train_mode):
                for parameter in raw_model.critic.parameters():
                    parameter.requires_grad = True
                for parameter in raw_model.value.parameters():
                    parameter.requires_grad = True
                if getattr(raw_model, 'critic_value_state_encoder', None) is not None:
                    for parameter in raw_model.critic_value_state_encoder.parameters():
                        parameter.requires_grad = True
                if getattr(raw_model, 'critic_value_state_fusion', None) is not None:
                    for parameter in raw_model.critic_value_state_fusion.parameters():
                        parameter.requires_grad = True
                for parameter in raw_model.target_critic.parameters():
                    parameter.requires_grad = False
                if getattr(raw_model, 'target_critic_value_state_encoder', None) is not None:
                    for parameter in raw_model.target_critic_value_state_encoder.parameters():
                        parameter.requires_grad = False
                if getattr(raw_model, 'target_critic_value_state_fusion', None) is not None:
                    for parameter in raw_model.target_critic_value_state_fusion.parameters():
                        parameter.requires_grad = False
        elif train_mode == 'biflow_align':
            if not hasattr(raw_model, 'reverse_model'):
                raise ValueError('train_mode=`biflow_align` requires `reverse_model`.')
            if bool(self.config.get('biflow_align_freeze_actor', True)):
                for parameter in raw_model.actor.parameters():
                    parameter.requires_grad = False
                for parameter in raw_model.actor_encoder.parameters():
                    parameter.requires_grad = False
                if getattr(raw_model, 'actor_context_encoder', None) is not None:
                    for parameter in raw_model.actor_context_encoder.parameters():
                        parameter.requires_grad = False
            for parameter in raw_model.target_critic.parameters():
                parameter.requires_grad = False
            if getattr(raw_model, 'target_critic_value_state_encoder', None) is not None:
                for parameter in raw_model.target_critic_value_state_encoder.parameters():
                    parameter.requires_grad = False
            if getattr(raw_model, 'target_critic_value_state_fusion', None) is not None:
                for parameter in raw_model.target_critic_value_state_fusion.parameters():
                    parameter.requires_grad = False
        for name, parameter in raw_model.named_parameters():
            if '.language_model.' in name:
                parameter.requires_grad = False
            if (
                bool(self.config.get('vlm_freeze_text_model', False))
                and name.startswith((
                    'actor_context_encoder.vlm.model.text_model.',
                    'actor_context_encoder.vlm.lm_head.',
                ))
            ):
                parameter.requires_grad = False
        return self

    def reset_optimizer_for_train_mode(self, train_mode: str | None = None):
        self._validate_train_mode_config()
        self.configure_trainable_parameters(train_mode)
        active_parameter_ids = {
            id(parameter)
            for parameter in self._iter_trainable_parameters_for_mode(train_mode)
            if parameter.requires_grad
        }
        actor_parameters = []
        context_adapter_parameters = []
        vlm_parameters = []
        critic_parameters = []
        assigned_parameter_ids = set()
        for name, parameter in self.raw_model.named_parameters():
            if id(parameter) not in active_parameter_ids:
                continue
            if id(parameter) in assigned_parameter_ids:
                continue
            assigned_parameter_ids.add(id(parameter))
            if name.startswith(ONLINE_CRITIC_VALUE_PARAMETER_PREFIXES):
                critic_parameters.append(parameter)
            elif name.startswith('actor_context_encoder.vlm.'):
                vlm_parameters.append(parameter)
            elif (
                float(self.config.get('context_adapter_lr', -1.0)) >= 0.0
                or float(self.config.get('context_adapter_max_grad_norm', -1.0)) >= 0.0
            ) and name.startswith(ACTOR_CONTEXT_ADAPTER_PARAMETER_PREFIXES):
                context_adapter_parameters.append(parameter)
            else:
                actor_parameters.append(parameter)

        if assigned_parameter_ids != active_parameter_ids:
            missing = len(active_parameter_ids - assigned_parameter_ids)
            raise RuntimeError(f'Failed to assign {missing} trainable parameters to an optimizer group.')
        if not actor_parameters and not critic_parameters:
            raise ValueError(f'No trainable parameters are available for train_mode={train_mode}.')

        base_lr = float(self.config['lr'])
        actor_lr = float(self.config.get('actor_lr', -1.0))
        critic_lr = float(self.config.get('critic_lr', -1.0))
        actor_lr = base_lr if actor_lr < 0.0 else actor_lr
        critic_lr = base_lr if critic_lr < 0.0 else critic_lr
        context_adapter_lr = float(self.config.get('context_adapter_lr', -1.0))
        context_adapter_lr = actor_lr if context_adapter_lr < 0.0 else context_adapter_lr
        actor_param_groups = []
        if actor_parameters:
            actor_param_groups.append({
                'params': actor_parameters,
                'lr': actor_lr,
                'name': 'actor',
            })
        if context_adapter_parameters:
            actor_param_groups.append({
                'params': context_adapter_parameters,
                'lr': context_adapter_lr,
                'name': 'context_adapter',
            })
        if vlm_parameters:
            actor_param_groups.append({
                'params': vlm_parameters,
                'lr': 0.0,
                'name': 'vlm',
            })
        optimizer_type = str(self.config.get('optimizer_type', 'adam')).strip().lower()
        if optimizer_type not in ('adam', 'adamw'):
            raise ValueError(
                f'Unsupported optimizer_type={optimizer_type!r}; expected `adam` or `adamw`.'
            )
        optimizer_cls = torch.optim.AdamW if optimizer_type == 'adamw' else torch.optim.Adam
        weight_decay = float(self.config.get('optimizer_weight_decay', 0.0))
        if weight_decay < 0.0:
            raise ValueError(f'optimizer_weight_decay must be non-negative, got {weight_decay}.')
        beta1 = float(self.config.get('optimizer_beta1', 0.9))
        beta2 = float(self.config.get('optimizer_beta2', 0.999))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(
                'optimizer_beta1 and optimizer_beta2 must both be in [0, 1); '
                f'got ({beta1}, {beta2}).'
            )
        optimizer_kwargs = {
            'foreach': False,
            'weight_decay': weight_decay,
            'betas': (beta1, beta2),
        }
        self.actor_optimizer = (
            optimizer_cls(actor_param_groups, lr=actor_lr, **optimizer_kwargs)
            if actor_param_groups
            else None
        )
        self.critic_optimizer = (
            optimizer_cls(critic_parameters, lr=critic_lr, **optimizer_kwargs)
            if critic_parameters
            else None
        )
        return self

    def _apply_optimizer_lr_schedule(self, optimizer_name: str, step: int) -> float:
        optimizer = self.optimizers.get(optimizer_name)
        if optimizer is None:
            return 0.0

        schedule = str(
            self.config.get(f'{optimizer_name}_lr_schedule', 'constant')
        ).strip().lower()
        if schedule not in ('constant', 'cosine'):
            raise ValueError(
                f'Unsupported {optimizer_name}_lr_schedule={schedule!r}; '
                'expected `constant` or `cosine`.'
            )
        warmup_steps = max(
            0,
            int(self.config.get(f'{optimizer_name}_lr_warmup_steps', 0)),
        )
        delay_steps = max(
            0,
            int(self.config.get(f'{optimizer_name}_lr_schedule_delay_steps', 0)),
        )
        schedule_step = max(0, int(step) - delay_steps)
        start_factor = float(self.config.get('lr_warmup_start_factor', 0.0))
        if not 0.0 <= start_factor <= 1.0:
            raise ValueError(
                f'lr_warmup_start_factor must be in [0, 1], got {start_factor}.'
            )
        base_lr = float(self.config['lr'])
        target_lr = float(self.config.get(f'{optimizer_name}_lr', -1.0))
        target_lr = base_lr if target_lr < 0.0 else target_lr
        if warmup_steps > 0 and schedule_step <= warmup_steps:
            progress = min(max(float(schedule_step), 0.0) / float(warmup_steps), 1.0)
            schedule_factor = start_factor + (1.0 - start_factor) * progress
        elif schedule == 'cosine':
            schedule_steps = int(
                self.config.get(f'{optimizer_name}_lr_schedule_steps', 0)
            )
            if schedule_steps <= warmup_steps:
                raise ValueError(
                    f'{optimizer_name}_lr_schedule_steps must be greater than warmup steps '
                    f'for cosine decay, got {schedule_steps} <= {warmup_steps}.'
                )
            min_ratio = float(self.config.get('lr_cosine_min_ratio', 0.0))
            if not 0.0 <= min_ratio <= 1.0:
                raise ValueError(
                    f'lr_cosine_min_ratio must be in [0, 1], got {min_ratio}.'
                )
            decay_progress = min(
                max(float(schedule_step - warmup_steps), 0.0)
                / float(schedule_steps - warmup_steps),
                1.0,
            )
            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            schedule_factor = min_ratio + (1.0 - min_ratio) * cosine_ratio
        else:
            schedule_factor = 1.0
        current_lr = target_lr * schedule_factor
        context_adapter_lr = float(self.config.get('context_adapter_lr', -1.0))
        context_adapter_lr = target_lr if context_adapter_lr < 0.0 else context_adapter_lr
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'vlm':
                continue
            param_group['lr'] = (
                context_adapter_lr * schedule_factor
                if param_group.get('name') == 'context_adapter'
                else current_lr
            )
        return current_lr

    # Kept for callers and old tests that refer to the warmup-only helper.
    def _apply_vlm_schedule(self, step: int, actor_lr: float) -> tuple[float, bool]:
        context_encoder = getattr(self.raw_model, 'actor_context_encoder', None)
        if not bool(self.config.get('vlm_fuse', False)) or context_encoder is None:
            return 0.0, False
        freeze_steps = max(0, int(self.config.get('vlm_freeze_steps', 1000)))
        frozen = int(step) <= freeze_steps
        context_encoder.set_backbone_frozen(frozen)
        vlm_lr = 0.0 if frozen else float(actor_lr) * float(
            self.config.get('vlm_lr_multiplier', 0.1)
        )
        if self.actor_optimizer is not None:
            for param_group in self.actor_optimizer.param_groups:
                if param_group.get('name') == 'vlm':
                    param_group['lr'] = vlm_lr
        return vlm_lr, frozen

    @property
    def optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            name: optimizer
            for name, optimizer in (
                ('actor', self.actor_optimizer),
                ('critic', self.critic_optimizer),
            )
            if optimizer is not None
        }

    def _optimizer_named_parameters(self, optimizer_name: str):
        optimizer = self.optimizers.get(optimizer_name)
        if optimizer is None:
            return []
        parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        return [
            (name, parameter)
            for name, parameter in self.raw_model.named_parameters()
            if id(parameter) in parameter_ids
        ]

    def _optimizer_group_named_parameters(self, optimizer_name: str, group_name: str):
        optimizer = self.optimizers.get(optimizer_name)
        if optimizer is None:
            return []
        parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            if group.get('name') == group_name
            for parameter in group['params']
        }
        return [
            (name, parameter)
            for name, parameter in self.raw_model.named_parameters()
            if id(parameter) in parameter_ids
        ]

    def _optimizer_max_grad_norm(self, optimizer_name: str) -> float:
        configured = float(self.config.get(f'{optimizer_name}_max_grad_norm', -1.0))
        return float(self.config.get('max_grad_norm', 0.0)) if configured < 0.0 else configured

    def _online_critic_value_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.raw_model.named_parameters()
            if name.startswith(ONLINE_CRITIC_VALUE_PARAMETER_PREFIXES)
        ]

    @staticmethod
    def _named_grad_norm(named_parameters) -> torch.Tensor:
        gradients = [
            parameter.grad.detach().float()
            for _, parameter in named_parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            return torch.tensor(0.0)
        device = gradients[0].device
        norm_sq = torch.zeros((), device=device, dtype=torch.float64)
        for gradient in gradients:
            norm_sq = norm_sq + gradient.double().pow(2).sum()
        return norm_sq.sqrt().float()

    def _prepare_observations(self, observations):
        if isinstance(observations, Mapping):
            raise TypeError(
                'Dictionary observations must be unpacked with '
                '_prepare_policy_inputs().'
            )
        observations = to_tensor(observations, device=self.device)
        return observations

    def _prepare_policy_inputs(self, observations, proprioceptions=None):
        if isinstance(observations, Mapping):
            if proprioceptions is not None:
                raise ValueError(
                    'Do not pass proprioceptions separately when observations is a mapping.'
                )
            if 'images' not in observations:
                raise KeyError('Policy observation mapping is missing `images`.')
            proprioceptions = observations.get('proprioceptions')
            observations = observations['images']

        observations = self._prepare_observations(observations)
        use_proprioception = uses_proprioception(self.config)
        if proprioceptions is None:
            if use_proprioception:
                raise ValueError(
                    'This policy requires proprioceptions, but none were provided.'
                )
            return observations, None
        if not use_proprioception:
            raise ValueError(
                'Proprioceptions were provided while state conditioning is disabled.'
            )
        proprioceptions = to_tensor(proprioceptions, device=self.device).float()
        if proprioceptions.ndim != 3 or proprioceptions.shape[0] != observations.shape[0]:
            raise ValueError(
                'Policy proprioceptions must have shape [B, T, D] and share the '
                f'image batch size, got {tuple(proprioceptions.shape)}.'
            )
        return observations, proprioceptions

    def _build_actor_conditioning(
        self,
        observations: torch.Tensor,
        language_tokens: tuple[torch.Tensor | None, torch.Tensor | None] | None,
        proprioceptions: torch.Tensor | None = None,
        *,
        raw_model: nn.Module | None = None,
    ) -> ActorConditioning:
        raw_model = self.raw_model if raw_model is None else raw_model
        input_ids, attention_mask = language_tokens or (None, None)
        return raw_model.build_actor_conditioning(
            observations,
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprioceptions=proprioceptions,
        )

    @staticmethod
    def _null_actor_conditioning(
        conditioning: ActorConditioning,
        actor: nn.Module,
    ) -> ActorConditioning:
        if conditioning.context_tokens is None:
            return ActorConditioning(y=None)
        null_token = getattr(actor, 'null_context_token', None)
        if null_token is None:
            raise RuntimeError('Context-conditioned actor is missing null_context_token.')
        return ActorConditioning(
            y=None,
            context_tokens=null_token.to(
                device=conditioning.context_tokens.device,
                dtype=conditioning.context_tokens.dtype,
            ).expand_as(conditioning.context_tokens),
            context_attention_mask=conditioning.context_attention_mask,
        )

    def _make_action_views(self, actions: torch.Tensor, *, add_data_noise: bool):
        flow_actions = actions
        noise_std = float(self.config.get('data_noise', 0.0))
        if add_data_noise and noise_std > 0.0:
            flow_actions = actions + torch.randn_like(actions) * noise_std
        return actions, flow_actions

    def _aggregate_qs(self, qs: torch.Tensor) -> torch.Tensor:
        return qs.min(dim=0).values if self.config['q_agg'] == 'min' else qs.mean(dim=0)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(
            diff > 0,
            torch.full_like(diff, expectile),
            torch.full_like(diff, 1.0 - expectile),
        )
        return weight * diff.pow(2)

    def _use_critic_language_conditioning(self) -> bool:
        return bool(self.config.get('critic_use_language_conditioning', False))

    def _use_value_language_conditioning(self) -> bool:
        return bool(self.config.get('value_use_language_conditioning', False))

    def _use_state_language_conditioning(self) -> bool:
        return self._use_critic_language_conditioning() or self._use_value_language_conditioning()

    def _share_critic_value_state(self) -> bool:
        return bool(self.config.get('share_critic_value_state', False))

    def _requires_batch_language_tokens(self) -> bool:
        return self._use_language_conditioning() or self._use_state_language_conditioning()

    def _critic_language_arg(self, language_embed: torch.Tensor | None) -> torch.Tensor | None:
        return language_embed if self._use_critic_language_conditioning() else None

    def _value_language_arg(self, language_embed: torch.Tensor | None) -> torch.Tensor | None:
        return language_embed if self._use_value_language_conditioning() else None

    def _is_target_critic_module(self, critic) -> bool:
        return critic is getattr(self.raw_model, 'target_critic', None)

    def _prepare_critic_value_inputs(
        self,
        observations: torch.Tensor,
        language_embed: torch.Tensor | None,
        proprioceptions: torch.Tensor | None,
        *,
        target: bool,
        use_value_language: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._share_critic_value_state():
            observations = self.raw_model.encode_critic_value_state(
                observations,
                language_embed=language_embed,
                proprioceptions=proprioceptions,
                target=target,
            )
            return observations, {'language_embed': None}

        if proprioceptions is not None and bool(
            getattr(self.raw_model, 'use_context_proprioception', False)
        ):
            proprioceptions = self.raw_model.proprio_normalizer(proprioceptions.float())

        language_arg = (
            self._value_language_arg(language_embed)
            if use_value_language
            else self._critic_language_arg(language_embed)
        )
        kwargs = {'language_embed': language_arg}
        if proprioceptions is not None:
            kwargs['proprioceptions'] = proprioceptions
        return observations, kwargs

    def _critic_call(
        self,
        critic,
        observations: torch.Tensor,
        actions: torch.Tensor,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observations, kwargs = self._prepare_critic_value_inputs(
            observations,
            language_embed,
            proprioceptions,
            target=self._is_target_critic_module(critic),
            use_value_language=False,
        )
        return critic(observations, actions, **kwargs)

    def _critic_forward_prefix(
        self,
        critic,
        observations: torch.Tensor,
        actions: torch.Tensor,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observations, kwargs = self._prepare_critic_value_inputs(
            observations,
            language_embed,
            proprioceptions,
            target=self._is_target_critic_module(critic),
            use_value_language=False,
        )
        return critic.forward_prefix(observations, actions, **kwargs)

    def _value(
        self,
        observations: torch.Tensor,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observations, kwargs = self._prepare_critic_value_inputs(
            observations,
            language_embed,
            proprioceptions,
            target=False,
            use_value_language=True,
        )
        values = self.raw_model.value(observations, **kwargs)
        if values.ndim > 1 and values.shape[0] == 1:
            values = values.squeeze(0)
        return values

    def _encode_state_language(
        self,
        language_tokens: tuple[torch.Tensor | None, torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        if not self._use_state_language_conditioning():
            return None
        input_ids, attention_mask = language_tokens or (None, None)
        if input_ids is None or attention_mask is None:
            raise ValueError(
                'critic/value language conditioning is enabled, but language tokens were not provided.'
            )
        state_text_encoder = getattr(self.raw_model, 'state_text_encoder', None)
        if state_text_encoder is None:
            raise ValueError('critic/value language conditioning is enabled, but `state_text_encoder` is missing.')
        language_embed = state_text_encoder(input_ids, attention_mask).to(device=self.device)
        return language_embed.float()

    @staticmethod
    def _repeat_state_language_embed(
        language_embed: torch.Tensor | None,
        repeats: int,
    ) -> torch.Tensor | None:
        if language_embed is None:
            return None
        if repeats <= 0:
            raise ValueError(f'repeats must be positive, got {repeats}.')
        return language_embed[:, None, :].expand(language_embed.shape[0], repeats, -1).reshape(
            language_embed.shape[0] * repeats,
            language_embed.shape[-1],
        )

    def _get_train_mode(self) -> str:
        train_mode = str(self.config.get('train_mode', 'rl')).lower()
        if train_mode not in ('rl', 'il', 'iql', 'biflow_align'):
            raise ValueError(
                f"Unsupported train_mode `{train_mode}`. "
                "Expected one of ['rl', 'il', 'iql', 'biflow_align']."
            )
        return train_mode

    def _is_imitation_mode(self) -> bool:
        return self._get_train_mode() == 'il'

    def _is_iql_mode(self) -> bool:
        return self._get_train_mode() == 'iql'

    def _is_biflow_align_mode(self) -> bool:
        return self._get_train_mode() == 'biflow_align'

    def _is_iql_critic_mode(self) -> bool:
        train_mode = self._get_train_mode()
        return (
            train_mode == 'iql'
            or self._train_critic_during_biflow_align(train_mode)
            or self._train_critic_value_during_il(train_mode)
        )

    def _compose_total_actor_loss(
        self,
        actor_loss: torch.Tensor,
        bc_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        biflow_loss: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = (
            actor_loss
            + entropy_loss * self.config['alpha_actor_entropy']
            + biflow_loss
        )
        return total_loss + bc_loss

    def _iql_advantage_weights(self, adv: torch.Tensor) -> torch.Tensor:
        temperature = float(self.config.get('iql_temperature', 0.1))
        if temperature <= 0.0:
            raise ValueError(f'iql_temperature must be positive, got {temperature}.')
        mode = str(self.config.get('iql_temperature_mode', 'multiply')).strip().lower()
        if mode != 'multiply':
            raise ValueError(
                'RoMAN-Flow requires iql_temperature_mode=`multiply`.'
            )
        scaled_adv = temperature * adv
        return torch.exp(scaled_adv).clamp(max=float(self.config.get('iql_adv_clip', 100.0)))

    def _use_biflow(self) -> bool:
        return bool(self.config.get('use_biflow', False))

    def _use_language_conditioning(self) -> bool:
        return bool(self.config.get('use_language_conditioning', False))

    def _format_language_tokens(
        self,
        input_ids,
        attention_mask,
        batch_size: int,
        source: str = 'language',
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._requires_batch_language_tokens():
            return None, None
        if input_ids is None or attention_mask is None:
            raise ValueError(
                'Language conditioning is enabled, but '
                f'`{source}_input_ids` / `{source}_attention_mask` were not provided.'
            )
        input_ids = to_tensor(input_ids, device=self.device).long()
        attention_mask = to_tensor(attention_mask, device=self.device).long()
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        if input_ids.shape[0] == 1 and batch_size != 1:
            input_ids = input_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
        elif input_ids.shape[0] != batch_size:
            raise ValueError(
                f'`{source}` language token batch size mismatch: got {tuple(input_ids.shape)}, '
                f'expected leading size {batch_size}.'
            )
        return input_ids, attention_mask

    def _get_named_batch_language_tokens(
        self,
        batch: Dict[str, torch.Tensor],
        input_ids_key: str,
        attention_mask_key: str,
        source: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._requires_batch_language_tokens():
            return None, None
        input_ids = batch[input_ids_key]
        attention_mask = batch[attention_mask_key]

        batch_size = int(batch['observations'].shape[0])
        input_ids = to_tensor(input_ids, device=self.device).long()
        attention_mask = to_tensor(attention_mask, device=self.device).long()
        if input_ids.ndim >= 3:
            obs_horizon, action_horizon, sequence_len = self._get_sequence_spec(self.config)
            input_ids = input_ids.reshape(input_ids.shape[0], input_ids.shape[1], -1)
            attention_mask = attention_mask.reshape(attention_mask.shape[0], attention_mask.shape[1], -1)
            token_sequence_len = input_ids.shape[1]
            if token_sequence_len == sequence_len:
                token_index = obs_horizon - 1
            elif token_sequence_len == action_horizon:
                token_index = 0
            elif token_sequence_len == 1:
                token_index = 0
            else:
                raise ValueError(
                    f'`{source}` token shape {tuple(input_ids.shape)} is inconsistent with '
                    f'sequence_len={sequence_len} and action_horizon={action_horizon}.'
                )
            input_ids = input_ids[:, token_index]
            attention_mask = attention_mask[:, token_index]
        return self._format_language_tokens(
            input_ids,
            attention_mask,
            batch_size=batch_size,
            source=source,
        )

    def _get_batch_language_tokens(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._requires_batch_language_tokens():
            return None, None
        if 'language_input_ids' in batch and 'language_attention_mask' in batch:
            keys = ('language_input_ids', 'language_attention_mask')
        elif 'input_ids' in batch and 'attention_mask' in batch:
            keys = ('input_ids', 'attention_mask')
        else:
            raise ValueError(
                'Language conditioning is enabled, but the batch does not contain '
                '`language_input_ids` and `language_attention_mask`.'
            )
        return self._get_named_batch_language_tokens(batch, *keys, source='language')

    def _get_batch_state_language_tokens(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._use_state_language_conditioning():
            return None, None

        input_key = 'critic_language_input_ids'
        mask_key = 'critic_language_attention_mask'
        has_input_ids = input_key in batch
        has_attention_mask = mask_key in batch
        if has_input_ids != has_attention_mask:
            raise ValueError(
                'The batch must contain both `critic_language_input_ids` and '
                '`critic_language_attention_mask`.'
            )
        if has_input_ids:
            return self._get_named_batch_language_tokens(
                batch,
                input_key,
                mask_key,
                source='critic_language',
            )
        if bool(self.config.get('vlm_fuse', False)):
            raise ValueError(
                'VLM-fused critic/value language conditioning requires a dual-token buffer '
                'with `critic_language_input_ids` and `critic_language_attention_mask`. '
                'The standard language fields contain SmolVLM ids and cannot be passed to CLIP.'
            )
        return self._get_batch_language_tokens(batch)

    def _validate_train_mode_config(self):
        if self._share_critic_value_state():
            if not (
                bool(self.config.get('critic_use_language_conditioning', False))
                and bool(self.config.get('value_use_language_conditioning', False))
            ):
                raise ValueError(
                    'share_critic_value_state=True requires both '
                    'critic_use_language_conditioning=True and '
                    'value_use_language_conditioning=True.'
                )
            if int(self.config.get('critic_language_embed_dim', 512)) != int(
                self.config.get('value_language_embed_dim', 512)
            ):
                raise ValueError(
                    'share_critic_value_state=True requires critic_language_embed_dim '
                    'to equal value_language_embed_dim.'
                )
        if self._is_imitation_mode() and float(self.config.get('alpha_actor', 0.0)) <= 0.0:
            raise ValueError(
                'train_mode=`il` requires a positive `alpha_actor`, '
                f"but got alpha_actor={self.config.get('alpha_actor')}."
            )
        if self._is_biflow_align_mode() and not self._use_biflow():
            raise ValueError('train_mode=`biflow_align` requires `use_biflow=True`.')
        if self._use_biflow() and not self._is_biflow_align_mode():
            raise ValueError(
                'use_biflow=True is only supported with train_mode=`biflow_align`.'
            )
        idql_expectile = float(self.config.get('iql_expectile', 0.8))
        if not (0.0 < idql_expectile < 1.0):
            raise ValueError(f'iql_expectile must be in (0, 1), got {idql_expectile}.')
        if self._use_biflow() and float(self.config.get('alpha_actor_entropy', 0.0)) != 0.0:
            raise ValueError('`use_biflow=True` does not support alpha_actor_entropy because the one-step reverse model has no exact logdet.')
        norm_p = float(self.config.get('biflow_norm_p', 0.0))
        if norm_p < 0.0:
            raise ValueError(f'biflow_norm_p must be non-negative, got {norm_p}.')
        norm_eps = float(self.config.get('biflow_norm_eps', 1e-3))
        if norm_eps <= 0.0:
            raise ValueError(f'biflow_norm_eps must be positive, got {norm_eps}.')
        for final_key in (
            'biflow_eval_guidance_final',
            'biflow_td_guidance_final',
        ):
            final_guidance = self.config.get(final_key, -1.0)
            if final_guidance is not None and float(final_guidance) < -1.0:
                raise ValueError(f'{final_key} must be -1, None, or a non-negative value, got {final_guidance}.')
        if (
            self._is_biflow_align_mode()
            and float(self.config.get('biflow_prior_weight', 1.0)) <= 0.0
            and not bool(self.config.get('biflow_allow_data_only_align', False))
        ):
            raise ValueError(
                'BiFlow alignment with prior_weight=0 requires an explicit opt-in. '
                'Set `--agent.biflow_allow_data_only_align=True` for strict original-BiFlow '
                'data-forward distillation, or set `--agent.biflow_prior_weight>0` to keep '
                'the extra prior-space teacher distillation used by earlier ORL experiments.'
            )

    def _biflow_per_sample_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred.float() - target.float()).pow(2)
        if loss.ndim <= 1:
            return loss
        return loss.flatten(start_dim=1).mean(dim=1)

    def _biflow_reduce_state_losses(self, per_sample_losses: torch.Tensor) -> torch.Tensor:
        norm_p = float(self.config.get('biflow_norm_p', 0.0))
        if norm_p <= 0.0:
            return per_sample_losses.mean()
        norm_eps = float(self.config.get('biflow_norm_eps', 1e-3))
        adaptive_weight = (per_sample_losses + norm_eps).pow(norm_p).detach()
        return (per_sample_losses / adaptive_weight.clamp_min(norm_eps)).mean()

    def _biflow_state_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_sample = self._biflow_per_sample_mse(pred, target)
        return self._biflow_reduce_state_losses(per_sample), per_sample.mean()

    def _compute_biflow_alignment_loss(
        self,
        action_seq: torch.Tensor,
        actor_conditioning: ActorConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self._use_biflow() or not hasattr(self.raw_model, 'reverse_model'):
            raise ValueError('BiFlow alignment requires `use_biflow=True` and an initialized reverse_model.')

        clean_actions = action_seq.detach()
        noise_level_config = self.config.get('biflow_noise_level', None)
        noise_level = (
            float(self.config.get('data_noise', 0.0))
            if noise_level_config is None or float(noise_level_config) < 0.0
            else float(noise_level_config)
        )
        if self.model.training and noise_level > 0.0:
            teacher_actions = clean_actions + torch.randn_like(clean_actions) * noise_level
        else:
            teacher_actions = clean_actions
        teacher_conditioning = actor_conditioning.detach()
        actor_was_training = self.raw_model.actor.training
        self.raw_model.actor.eval()
        with torch.no_grad():
            teacher_z, _, teacher_states = self.raw_model.actor.forward(
                x=teacher_actions,
                **teacher_conditioning.kwargs(),
                return_sequence=True,
            )
        if actor_was_training:
            self.raw_model.actor.train()

        teacher_targets = [state.detach() for state in reversed(teacher_states[:-1])]
        teacher_targets.append(clean_actions)
        guidance_min = float(self.config.get('biflow_guidance_min', 0.0))
        guidance_max = float(self.config.get('biflow_guidance_max', 0.5))
        if guidance_max < guidance_min:
            raise ValueError(
                f'biflow_guidance_max must be >= biflow_guidance_min, got '
                f'{guidance_max} < {guidance_min}.'
            )
        if self.model.training and guidance_max > guidance_min:
            biflow_guidance = torch.empty(
                clean_actions.shape[0],
                device=clean_actions.device,
                dtype=clean_actions.dtype,
            ).uniform_(guidance_min, guidance_max)
        else:
            biflow_guidance = torch.full(
                (clean_actions.shape[0],),
                guidance_min,
                device=clean_actions.device,
                dtype=clean_actions.dtype,
            )
        biflow_guidance_schedule = self._make_biflow_guidance_schedule(
            biflow_guidance,
            clean_actions.shape[0],
            final_guidance=biflow_guidance,
            device=clean_actions.device,
            dtype=clean_actions.dtype,
        )
        pred_states = self.raw_model.reverse_model(
            z=teacher_z.detach(),
            **teacher_conditioning.kwargs(),
            guidance=biflow_guidance_schedule,
            return_sequence=True,
            guidance_per_block=True,
        )
        if not isinstance(pred_states, list) or len(pred_states) == 0:
            raise RuntimeError('reverse_model(return_sequence=True) must return a non-empty list of states.')

        num_states = len(teacher_targets)
        if len(pred_states) != num_states:
            raise ValueError(
                f'BiFlow alignment state count mismatch: reverse_model returned {len(pred_states)} states, '
                f'but SimFlow teacher produced {num_states} targets. Check bc_num_blocks and reverse_model num_blocks.'
            )
        data_intermediate_pairs = []
        if num_states > 1:
            data_intermediate_pairs = [
                self._biflow_state_mse_loss(pred, target)
                for pred, target in zip(pred_states[:-1], teacher_targets[:-1])
            ]
            data_intermediate_loss = torch.stack([loss for loss, _ in data_intermediate_pairs]).mean()
            data_intermediate_mse_raw = torch.stack([raw for _, raw in data_intermediate_pairs]).mean()
        else:
            data_intermediate_loss = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
            data_intermediate_mse_raw = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        data_action_pair = self._biflow_state_mse_loss(
            pred_states[-1],
            teacher_targets[-1],
        )
        data_action_loss, data_action_mse_raw = data_action_pair
        align_weight = float(self.config.get('biflow_align_weight', 1.0))
        action_weight = float(self.config.get('biflow_action_weight', 1.0))
        data_loss = (
            align_weight * data_intermediate_loss
            + action_weight * data_action_loss
        )

        prior_weight = float(self.config.get('biflow_prior_weight', 0.0))
        prior_intermediate_loss = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        prior_action_loss = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        prior_intermediate_mse_raw = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        prior_action_mse_raw = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        prior_loss = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        prior_z_norm = torch.zeros((), device=action_seq.device, dtype=action_seq.dtype)
        if prior_weight > 0.0:
            prior_fraction = float(self.config.get('biflow_prior_batch_fraction', 1.0))
            if prior_fraction <= 0.0:
                raise ValueError(f'biflow_prior_batch_fraction must be positive, got {prior_fraction}.')
            prior_batch_size = max(1, min(clean_actions.shape[0], int(round(clean_actions.shape[0] * prior_fraction))))
            prior_conditioning = teacher_conditioning.slice(slice(0, prior_batch_size))
            prior_z = self.prior.sample(prior_batch_size).to(device=clean_actions.device, dtype=clean_actions.dtype)
            prior_z = prior_z * float(self.config.get('biflow_prior_temperature', 1.0))
            teacher_guidance = float(self.config.get('biflow_prior_teacher_guidance', -1.0))
            if teacher_guidance < 0.0:
                teacher_guidance = float(self.config.get('cfg', 0.0))

            actor_was_training = self.raw_model.actor.training
            self.raw_model.actor.eval()
            with torch.no_grad():
                teacher_prior_seq, _ = self.raw_model.actor.reverse(
                    prior_z,
                    **prior_conditioning.kwargs(),
                    guidance=teacher_guidance,
                    return_sequence=True,
                )
            if actor_was_training:
                self.raw_model.actor.train()

            prior_targets = [state.detach() for state in teacher_prior_seq[1:]]
            prior_targets.append(teacher_prior_seq[-1].detach())
            prior_pred_states = self.raw_model.reverse_model(
                z=prior_z.detach(),
                **prior_conditioning.kwargs(),
                guidance=biflow_guidance_schedule[:, :prior_batch_size],
                return_sequence=True,
                guidance_per_block=True,
            )
            if len(prior_pred_states) != num_states or len(prior_targets) != num_states:
                raise ValueError(
                    f'BiFlow prior alignment state count mismatch: reverse_model={len(prior_pred_states)}, '
                    f'teacher_targets={len(prior_targets)}, expected={num_states}.'
                )
            prior_intermediate_pairs = []
            if num_states > 1:
                prior_intermediate_pairs = [
                    self._biflow_state_mse_loss(pred, target)
                    for pred, target in zip(prior_pred_states[:-1], prior_targets[:-1])
                ]
                prior_intermediate_loss = torch.stack([loss for loss, _ in prior_intermediate_pairs]).mean()
                prior_intermediate_mse_raw = torch.stack([raw for _, raw in prior_intermediate_pairs]).mean()
            prior_action_pair = self._biflow_state_mse_loss(
                prior_pred_states[-1],
                prior_targets[-1],
            )
            prior_action_loss, prior_action_mse_raw = prior_action_pair
            prior_loss = align_weight * prior_intermediate_loss + action_weight * prior_action_loss
            prior_z_norm = prior_z.detach().float().pow(2).mean()

        total_loss = data_loss + prior_weight * prior_loss
        info = {
            'biflow/intermediate_loss': data_intermediate_loss.detach(),
            'biflow/action_loss': data_action_loss.detach(),
            'biflow/total_loss': total_loss.detach(),
            'biflow/data_loss': data_loss.detach(),
            'biflow/data_intermediate_loss': data_intermediate_loss.detach(),
            'biflow/data_action_loss': data_action_loss.detach(),
            'biflow/data_intermediate_mse_raw': data_intermediate_mse_raw.detach(),
            'biflow/data_action_mse_raw': data_action_mse_raw.detach(),
            'biflow/prior_loss': prior_loss.detach(),
            'biflow/prior_intermediate_loss': prior_intermediate_loss.detach(),
            'biflow/prior_action_loss': prior_action_loss.detach(),
            'biflow/prior_intermediate_mse_raw': prior_intermediate_mse_raw.detach(),
            'biflow/prior_action_mse_raw': prior_action_mse_raw.detach(),
            'biflow/prior_weight': prior_weight,
            'biflow/prior_z_norm': prior_z_norm.detach(),
            'biflow/z_norm': teacher_z.detach().float().pow(2).mean(),
            'biflow/noise_level': noise_level,
            'biflow/norm_p': float(self.config.get('biflow_norm_p', 0.0)),
            'biflow/norm_eps': float(self.config.get('biflow_norm_eps', 1e-3)),
            'biflow/guidance_mean': biflow_guidance.detach().mean(),
            'biflow/guidance_final_mean': biflow_guidance_schedule[-1].detach().mean(),
            'biflow/num_states': float(num_states),
        }

        data_intermediate_term = (align_weight * data_intermediate_loss).detach()
        data_action_term = (action_weight * data_action_loss).detach()
        data_term_sum = (data_intermediate_term + data_action_term).clamp_min(1e-12)
        info['biflow/data_intermediate_objective_fraction'] = data_intermediate_term / data_term_sum
        info['biflow/data_action_objective_fraction'] = data_action_term / data_term_sum
        if prior_weight > 0.0:
            prior_intermediate_term = (align_weight * prior_intermediate_loss).detach()
            prior_action_term = (action_weight * prior_action_loss).detach()
            prior_term_sum = (prior_intermediate_term + prior_action_term).clamp_min(1e-12)
            info['biflow/prior_intermediate_objective_fraction'] = prior_intermediate_term / prior_term_sum
            info['biflow/prior_action_objective_fraction'] = prior_action_term / prior_term_sum

        return total_loss, pred_states[-1], info

    def _get_hubl_relabel_tensors(
        self,
        batch: Dict[str, torch.Tensor],
        reference_rewards: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if str(self.config.get('hubl_lambda_type', 'rank')) != 'rank':
            raise ValueError(
                f"Unsupported HUBL lambda type `{self.config.get('hubl_lambda_type')}`. "
                "This implementation currently supports only `rank`."
            )

        required_keys = ('hubl_lambda', 'hubl_rewards', 'hubl_discounts')
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise ValueError(
                f'HUBL is enabled but batch is missing required relabeled fields: {missing_keys}.'
            )

        hubl_lambda = batch['hubl_lambda'].float().reshape(-1)
        hubl_rewards = batch['hubl_rewards'].float()
        hubl_discounts = batch['hubl_discounts'].float()
        if hubl_rewards.ndim == 1:
            hubl_rewards = hubl_rewards.unsqueeze(1)
        if hubl_discounts.ndim == 1:
            hubl_discounts = hubl_discounts.unsqueeze(1)

        if reference_rewards is not None and (
            hubl_rewards.shape[0] != reference_rewards.shape[0]
            or hubl_discounts.shape[0] != reference_rewards.shape[0]
            or hubl_rewards.shape[1] < reference_rewards.shape[1]
            or hubl_discounts.shape[1] < reference_rewards.shape[1]
        ):
            raise ValueError(
                f'HUBL relabeled fields must cover rewards: '
                f'rewards={tuple(reference_rewards.shape)}, '
                f'hubl_rewards={tuple(hubl_rewards.shape)}, '
                f'hubl_discounts={tuple(hubl_discounts.shape)}.'
            )
        return hubl_lambda, hubl_rewards, hubl_discounts

    def _compute_iql_prefix_targets(
        self,
        batch,
        use_hubl: bool = False,
        state_language_embed: torch.Tensor | None = None,
    ):
        rewards = batch['rewards'].float()
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)

        masks = batch['masks'].float()
        if masks.ndim == 1:
            masks = masks.unsqueeze(1)

        prefix_next_obs = batch.get('prefix_next_observations')
        if prefix_next_obs is None:
            prefix_next_obs = batch['next_observations'].unsqueeze(1)

        prefix_len = min(rewards.shape[1], prefix_next_obs.shape[1])
        rewards = rewards[:, :prefix_len]
        masks = masks[:, :prefix_len]
        prefix_next_obs = prefix_next_obs[:, :prefix_len]
        batch_size = rewards.shape[0]

        flat_prefix_next_obs = prefix_next_obs.reshape(batch_size * prefix_len, *prefix_next_obs.shape[2:])
        prefix_next_proprioceptions = batch.get('prefix_next_proprioceptions')
        flat_prefix_next_proprioceptions = None
        if prefix_next_proprioceptions is not None:
            prefix_next_proprioceptions = prefix_next_proprioceptions[:, :prefix_len]
            flat_prefix_next_proprioceptions = prefix_next_proprioceptions.reshape(
                batch_size * prefix_len,
                *prefix_next_proprioceptions.shape[2:],
            )
        flat_state_language_embed = self._repeat_state_language_embed(state_language_embed, prefix_len)
        prefix_next_values = self._value(
            flat_prefix_next_obs,
            language_embed=flat_state_language_embed,
            proprioceptions=flat_prefix_next_proprioceptions,
        ).reshape(batch_size, prefix_len)

        hubl_info = {
            'target_mix_lambda': 0.0,
            'hubl_lambda_mean': None,
            'hubl_lambda_min': None,
            'hubl_lambda_max': None,
            'hubl_reward_mean': None,
            'hubl_discount_mean': None,
        }
        if use_hubl:
            hubl_lambda, hubl_rewards, hubl_discounts = self._get_hubl_relabel_tensors(
                batch, reference_rewards=rewards
            )
            hubl_rewards = hubl_rewards[:, :prefix_len]
            hubl_discounts = hubl_discounts[:, :prefix_len]
            if hubl_rewards.shape != rewards.shape or hubl_discounts.shape != rewards.shape:
                raise ValueError(
                    f'HUBL relabeled fields must align with IQL prefix rewards: '
                    f'rewards={tuple(rewards.shape)}, '
                    f'hubl_rewards={tuple(hubl_rewards.shape)}, '
                    f'hubl_discounts={tuple(hubl_discounts.shape)}.'
                )

            prefix_discounted_rewards = torch.zeros_like(hubl_rewards)
            prefix_targets = torch.zeros_like(hubl_rewards)
            cumulative_discount = torch.ones(batch_size, device=self.device, dtype=hubl_rewards.dtype)
            cumulative_return = torch.zeros(batch_size, device=self.device, dtype=hubl_rewards.dtype)
            for step in range(prefix_len):
                cumulative_return = cumulative_return + cumulative_discount * hubl_rewards[:, step]
                prefix_discounted_rewards[:, step] = cumulative_return
                cumulative_discount = cumulative_discount * hubl_discounts[:, step]
                prefix_targets[:, step] = cumulative_return + cumulative_discount * prefix_next_values[:, step]

            hubl_info = {
                'target_mix_lambda': hubl_lambda.mean().detach(),
                'hubl_lambda_mean': hubl_lambda.mean().detach(),
                'hubl_lambda_min': hubl_lambda.min().detach(),
                'hubl_lambda_max': hubl_lambda.max().detach(),
                'hubl_reward_mean': hubl_rewards.mean().detach(),
                'hubl_discount_mean': hubl_discounts.mean().detach(),
            }
        else:
            reward_discounts = (
                self.config['discount'] ** torch.arange(prefix_len, device=self.device, dtype=rewards.dtype)
            ).unsqueeze(0)
            prefix_discounted_rewards = torch.cumsum(rewards * reward_discounts, dim=1)
            bootstrap_discounts = (
                self.config['discount']
                ** torch.arange(1, prefix_len + 1, device=self.device, dtype=rewards.dtype)
            ).unsqueeze(0)
            prefix_targets = prefix_discounted_rewards + bootstrap_discounts * masks * prefix_next_values

            mc_lambda = float(self.config.get('mc_target_lambda', 0.0))
            if mc_lambda > 0.0 and 'mc_returns' in batch:
                mc_returns = batch['mc_returns'].float()
                if mc_returns.ndim >= 2:
                    mc_returns = mc_returns[:, 0]
                mc_returns = mc_returns.reshape(batch_size, 1)
                mc_lambda = min(1.0, max(0.0, mc_lambda))
                prefix_targets = mc_lambda * mc_returns + (1.0 - mc_lambda) * prefix_targets
                hubl_info['target_mix_lambda'] = mc_lambda

        return prefix_targets, prefix_discounted_rewards, prefix_next_values, masks, hubl_info

    def _compute_prefix_td_targets(
        self,
        batch,
        use_hubl: bool = False,
        language_tokens: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        state_language_embed: torch.Tensor | None = None,
    ):
        rewards = batch['rewards'].float()
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)

        masks = batch['masks'].float()
        if masks.ndim == 1:
            masks = masks.unsqueeze(1)

        prefix_next_obs = batch.get('prefix_next_observations')
        if prefix_next_obs is None:
            prefix_next_obs = batch['next_observations'].unsqueeze(1)

        prefix_len = min(rewards.shape[1], prefix_next_obs.shape[1])
        rewards = rewards[:, :prefix_len]
        masks = masks[:, :prefix_len]
        prefix_next_obs = prefix_next_obs[:, :prefix_len]
        batch_size = rewards.shape[0]

        reward_discounts = (self.config['discount'] ** torch.arange(prefix_len, device=self.device, dtype=rewards.dtype)).unsqueeze(0)
        prefix_discounted_rewards = torch.cumsum(rewards * reward_discounts, dim=1)

        flat_prefix_next_obs = prefix_next_obs.reshape(batch_size * prefix_len, *prefix_next_obs.shape[2:])
        prefix_next_proprioceptions = batch.get('prefix_next_proprioceptions')
        flat_prefix_next_proprioceptions = None
        if prefix_next_proprioceptions is not None:
            prefix_next_proprioceptions = prefix_next_proprioceptions[:, :prefix_len]
            flat_prefix_next_proprioceptions = prefix_next_proprioceptions.reshape(
                batch_size * prefix_len,
                *prefix_next_proprioceptions.shape[2:],
            )
        flat_language_tokens = (None, None)
        if language_tokens:
            language_input_ids, language_attention_mask = language_tokens
            if language_input_ids is not None and language_attention_mask is not None:
                flat_language_tokens = (
                    language_input_ids[:, None, :].expand(batch_size, prefix_len, -1).reshape(batch_size * prefix_len, -1),
                    language_attention_mask[:, None, :].expand(batch_size, prefix_len, -1).reshape(batch_size * prefix_len, -1),
                )
        next_actions = self.sample_actions(
            flat_prefix_next_obs,
            proprioceptions=flat_prefix_next_proprioceptions,
            input_ids=flat_language_tokens[0],
            attention_mask=flat_language_tokens[1],
            temperature=float(self.config.get('biflow_td_temperature', 1.0)) if self._use_biflow() else None,
            cfg=float(self.config.get('biflow_td_guidance', 0.0)) if self._use_biflow() else None,
            cfg_final=(
                self._get_biflow_guidance_final(
                    'biflow_td_guidance_final',
                    float(self.config.get('biflow_td_guidance', 0.0)),
                )
                if self._use_biflow()
                else None
            ),
        )
        flat_state_language_embed = self._repeat_state_language_embed(state_language_embed, prefix_len)
        next_qs = self._critic_call(
            self.raw_model.target_critic,
            flat_prefix_next_obs,
            next_actions,
            language_embed=flat_state_language_embed,
            proprioceptions=flat_prefix_next_proprioceptions,
        )
        next_chunk_values = self._aggregate_qs(next_qs).reshape(batch_size, prefix_len)

        hubl_info = {
            'target_mix_lambda': 0.0,
            'hubl_lambda_mean': None,
            'hubl_lambda_min': None,
            'hubl_lambda_max': None,
            'hubl_reward_mean': None,
            'hubl_discount_mean': None,
        }
        if use_hubl:
            hubl_lambda, hubl_rewards, hubl_discounts = self._get_hubl_relabel_tensors(
                batch, reference_rewards=rewards
            )
            hubl_rewards = hubl_rewards[:, :prefix_len]
            hubl_discounts = hubl_discounts[:, :prefix_len]
            if hubl_rewards.shape != rewards.shape or hubl_discounts.shape != rewards.shape:
                raise ValueError(
                    f'HUBL relabeled fields must align with chunk prefix rewards: '
                    f'rewards={tuple(rewards.shape)}, '
                    f'hubl_rewards={tuple(hubl_rewards.shape)}, '
                    f'hubl_discounts={tuple(hubl_discounts.shape)}.'
                )

            prefix_discounted_rewards = torch.zeros_like(hubl_rewards)
            prefix_targets = torch.zeros_like(hubl_rewards)
            cumulative_discount = torch.ones(batch_size, device=self.device, dtype=hubl_rewards.dtype)
            cumulative_return = torch.zeros(batch_size, device=self.device, dtype=hubl_rewards.dtype)
            for step in range(prefix_len):
                cumulative_return = cumulative_return + cumulative_discount * hubl_rewards[:, step]
                prefix_discounted_rewards[:, step] = cumulative_return
                cumulative_discount = cumulative_discount * hubl_discounts[:, step]
                prefix_targets[:, step] = cumulative_return + cumulative_discount * next_chunk_values[:, step]

            hubl_info = {
                'target_mix_lambda': hubl_lambda.mean().detach(),
                'hubl_lambda_mean': hubl_lambda.mean().detach(),
                'hubl_lambda_min': hubl_lambda.min().detach(),
                'hubl_lambda_max': hubl_lambda.max().detach(),
                'hubl_reward_mean': hubl_rewards.mean().detach(),
                'hubl_discount_mean': hubl_discounts.mean().detach(),
            }
        else:
            bootstrap_discounts = (
                self.config['discount']
                ** torch.arange(1, prefix_len + 1, device=self.device, dtype=rewards.dtype)
            ).unsqueeze(0)
            prefix_targets = prefix_discounted_rewards + bootstrap_discounts * masks * next_chunk_values

        return prefix_targets, prefix_discounted_rewards, next_chunk_values, masks, hubl_info

    def _resolve_sample_backend(self, sample_backend: str | None = None) -> str:
        backend = 'biflow' if self._use_biflow() else 'simflow'
        if sample_backend is not None:
            backend = str(sample_backend).strip().lower()
            if backend == 'auto':
                backend = 'biflow' if self._use_biflow() else 'simflow'
        if backend not in ('simflow', 'biflow'):
            raise ValueError(f"Unsupported sample_backend `{sample_backend}`. Expected auto, simflow, or biflow.")
        if backend == 'biflow' and (not self._use_biflow() or not hasattr(self.raw_model, 'reverse_model')):
            raise ValueError('sample_backend=`biflow` requires `use_biflow=True` and an initialized reverse_model.')
        return backend

    def sample_actions(
        self,
        observations,
        cfg=None,
        temperature=None,
        model=None,
        apply_denoising=False,
        input_ids=None,
        attention_mask=None,
        cfg_final=None,
        sample_backend=None,
        proprioceptions=None,
    ):
        observations, proprioceptions = self._prepare_policy_inputs(
            observations,
            proprioceptions=proprioceptions,
        )
        if self._use_language_conditioning():
            language_input_ids, language_attention_mask = self._format_language_tokens(
                input_ids,
                attention_mask,
                batch_size=int(observations.shape[0]),
                source='language',
            )
        else:
            language_input_ids, language_attention_mask = None, None
        active_model = model if model is not None else self.model
        active_raw_model = active_model.module if hasattr(active_model, 'module') else active_model
        was_training = active_model.training
        guidance = self.config['cfg'] if cfg is None else cfg
        backend = self._resolve_sample_backend(sample_backend)
        use_biflow = backend == 'biflow'
        if temperature is None:
            temperature = (
                float(self.config.get('biflow_eval_temperature', 0.0))
                if use_biflow
                else 1.0
            )
        if use_biflow and cfg is None:
            guidance = self.config.get('biflow_eval_guidance', guidance)
        biflow_eval_guidance_base = float(guidance)
        active_model.eval()
        with torch.no_grad():
            actor_conditioning = self._build_actor_conditioning(
                observations,
                (language_input_ids, language_attention_mask),
                proprioceptions=proprioceptions,
                raw_model=active_raw_model,
            )
            b_noise = self.prior.sample(observations.shape[0]) * temperature
            if use_biflow:
                if not hasattr(active_raw_model, 'reverse_model'):
                    raise ValueError('`use_biflow=True` but the active model has no reverse_model.')
                guidance = self._make_biflow_guidance_schedule(
                    guidance,
                    int(observations.shape[0]),
                    final_guidance=(
                        self._get_biflow_guidance_final(
                            'biflow_eval_guidance_final',
                            biflow_eval_guidance_base,
                        )
                        if cfg_final is None or float(cfg_final) < 0.0
                        else cfg_final
                    ),
                    device=b_noise.device,
                    dtype=b_noise.dtype,
                )
                actions = self._biflow_sampling_reverse_model(active_raw_model)(
                    z=b_noise,
                    **actor_conditioning.kwargs(),
                    guidance=guidance,
                    guidance_per_block=True,
                )
            else:
                actions, _ = active_raw_model.actor.reverse(
                    x=b_noise,
                    **actor_conditioning.kwargs(),
                    guidance=guidance,
                )

        denoising_lr = float(self.config.get('denoising_lr', 0.0))
        if (not use_biflow) and apply_denoising and denoising_lr > 0.0:
            denoising_batch_size = 16
            prev_requires_grad = [p.requires_grad for p in active_model.parameters()]
            for p in active_model.parameters():
                p.requires_grad = False

            denoised_actions = []
            with torch.enable_grad():
                for start in range(0, actions.shape[0], denoising_batch_size):
                    end = start + denoising_batch_size
                    x_chunk = actions[start:end].detach().clone()
                    x_chunk.requires_grad_(True)
                    conditioning_chunk = actor_conditioning.slice(slice(start, end)).detach()
                    null_conditioning_chunk = self._null_actor_conditioning(
                        conditioning_chunk,
                        active_raw_model.actor,
                    )
                    z_cond, logdets_cond = active_raw_model.actor.forward(
                        x=x_chunk,
                        **conditioning_chunk.kwargs(),
                    )
                    loss_cond = active_raw_model.actor.get_loss(z_cond, logdets_cond)
                    z_unc, logdets_unc = active_raw_model.actor.forward(
                        x=x_chunk,
                        **null_conditioning_chunk.kwargs(),
                    )
                    loss_unc = active_raw_model.actor.get_loss(z_unc, logdets_unc)
                    loss = loss_cond - loss_unc
                    grad = torch.autograd.grad(loss, [x_chunk])[0]
                    grad_scale = grad.abs().mean(dim=tuple(range(1, grad.ndim)), keepdim=True).clamp_min(1e-6)
                    x_chunk = x_chunk - denoising_lr * (grad / grad_scale)
                    x_chunk = x_chunk.detach()
                    denoised_actions.append(x_chunk)

            actions = torch.cat(denoised_actions, dim=0)

            for p, requires_grad in zip(active_model.parameters(), prev_requires_grad):
                p.requires_grad = requires_grad

        if was_training:
            active_model.train()
        return actions

    def total_loss(self, batch, full_update=True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute losses without entering DDP; used by validation and diagnostics."""
        return self._total_loss_impl(
            batch,
            full_update=full_update,
        )

    def _training_total_loss(self, batch, full_update=True):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model(
                self._total_loss_impl,
                batch,
                full_update=full_update,
            )
        return self._total_loss_impl(
            batch,
            full_update=full_update,
        )

    def _total_loss_impl(self, batch, full_update=True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = self._prepare_batch(batch)
        language_tokens = self._get_batch_language_tokens(batch)
        state_language_tokens = self._get_batch_state_language_tokens(batch)
        state_language_embed = self._encode_state_language(state_language_tokens)
        proprioceptions = batch.get('proprioceptions')
        action_valid_mask = self._get_action_valid_mask(batch, batch['actions'])
        valid_action_samples = action_valid_mask.all(dim=1).float()
        is_biflow_align_mode = self._is_biflow_align_mode()
        add_data_noise = (
            self.config.get('data_noise', 0.0) > 0.0
            and self.model.training
            and full_update
            and not is_biflow_align_mode
        )
        clean_action_seq, flow_action_seq = self._make_action_views(
            batch['actions'],
            add_data_noise=add_data_noise,
        )
        batch['actions'] = clean_action_seq
        info = {}
        use_amp = bool(self.config.get('use_amp', False)) and self.device.type == 'cuda'
        is_imitation_mode = self._is_imitation_mode()
        is_iql_mode = self._is_iql_mode()
        is_iql_critic_mode = self._is_iql_critic_mode()
        train_critic_value_during_il = self._train_critic_value_during_il()
        train_critic_during_align = self._train_critic_during_biflow_align()
        use_hubl = bool(self.config.get('use_hubl', False))

        info.update({
            'data/action_noise_std': (flow_action_seq - clean_action_seq).float().std(unbiased=False).detach(),
            'data/critic_action_noise_std': torch.tensor(0.0, device=self.device),
        })

        if is_biflow_align_mode:
            zero_scalar = torch.tensor(0.0, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    actor_conditioning = self._build_actor_conditioning(
                        batch['observations'],
                        language_tokens,
                        proprioceptions=proprioceptions,
                    )
                biflow_loss, biflow_actions, biflow_info = self._compute_biflow_alignment_loss(
                    clean_action_seq,
                    actor_conditioning,
                )
            biflow_last = biflow_actions[:, -1] if biflow_actions.ndim == 3 else biflow_actions
            gt_last = clean_action_seq[:, -1] if clean_action_seq.ndim == 3 else clean_action_seq
            sampled_action_mse = (biflow_last - gt_last).pow(2).sum(dim=-1).mean()
            sampled_action_mse_loss = (biflow_actions - clean_action_seq).pow(2).mean()

            critic_loss = zero_scalar
            q = zero_scalar
            q_prefix_agg = None
            target_q = zero_scalar
            next_q = zero_scalar
            chunk_reward = zero_scalar
            bellman_target = zero_scalar
            mc_target = zero_scalar
            target_mix_lambda = 0.0
            value_loss = zero_scalar
            value_v = zero_scalar
            value_q = zero_scalar
            prefix_targets = None
            prefix_bootstrap_values = None
            prefix_discounted_rewards = None
            hubl_lambda_mean = None
            hubl_lambda_min = None
            hubl_lambda_max = None
            hubl_reward_mean = None
            hubl_discount_mean = None
            adv = torch.zeros(batch['observations'].shape[0], device=self.device)
            weights = torch.ones_like(adv)

            if train_critic_during_align:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    with torch.no_grad():
                        (
                            prefix_targets,
                            prefix_discounted_rewards,
                            prefix_bootstrap_values,
                            _,
                            prefix_hubl_info,
                        ) = self._compute_iql_prefix_targets(
                            batch,
                            use_hubl=use_hubl,
                            state_language_embed=state_language_embed,
                        )
                        target_q = prefix_targets.mean(dim=-1)
                        next_q = prefix_bootstrap_values.mean(dim=-1)
                        chunk_reward = prefix_discounted_rewards[:, -1]
                        bellman_target = prefix_targets[:, -1]
                        mc_target = torch.zeros_like(target_q)
                        target_mix_lambda = prefix_hubl_info['target_mix_lambda']
                        hubl_lambda_mean = prefix_hubl_info['hubl_lambda_mean']
                        hubl_lambda_min = prefix_hubl_info['hubl_lambda_min']
                        hubl_lambda_max = prefix_hubl_info['hubl_lambda_max']
                        hubl_reward_mean = prefix_hubl_info['hubl_reward_mean']
                        hubl_discount_mean = prefix_hubl_info['hubl_discount_mean']
                    q = self._critic_call(
                        self.raw_model.critic,
                        batch['observations'],
                        batch['actions'],
                        language_embed=state_language_embed,
                        proprioceptions=proprioceptions,
                    )
                    q_prefix = self._critic_forward_prefix(
                        self.raw_model.critic,
                        batch['observations'],
                        batch['actions'],
                        language_embed=state_language_embed,
                        proprioceptions=proprioceptions,
                    )
                    prefix_len = min(q_prefix.shape[-1], prefix_targets.shape[1])
                    q_prefix = q_prefix[..., :prefix_len]
                    prefix_targets = prefix_targets[:, :prefix_len]
                    prefix_bootstrap_values = prefix_bootstrap_values[:, :prefix_len]
                    prefix_discounted_rewards = prefix_discounted_rewards[:, :prefix_len]
                    q_prefix_agg = self._aggregate_qs(q_prefix)
                    critic_loss = self._masked_mean(
                        (q_prefix - prefix_targets.unsqueeze(0)).pow(2),
                        action_valid_mask[:, :prefix_len],
                        'critic prefix loss',
                    )
                    target_q = prefix_targets.mean(dim=-1)
                    next_q = prefix_bootstrap_values.mean(dim=-1)
                    chunk_reward = prefix_discounted_rewards[:, -1]
                    bellman_target = prefix_targets[:, -1]
                    with torch.no_grad():
                        value_q = self._aggregate_qs(
                            self._critic_call(
                                self.raw_model.target_critic,
                                batch['observations'],
                                batch['actions'],
                                language_embed=state_language_embed,
                                proprioceptions=proprioceptions,
                            )
                        )
                    value_v = self._value(
                        batch['observations'],
                        language_embed=state_language_embed,
                        proprioceptions=proprioceptions,
                    )
                    value_loss = self._masked_mean(
                        self._expectile_loss(
                            value_q - value_v,
                            float(self.config.get('iql_expectile', 0.8)),
                        ),
                        action_valid_mask.any(dim=1),
                        'value loss',
                    )
                    with torch.no_grad():
                        adv = value_q.detach() - value_v.detach()
                        weights = self._iql_advantage_weights(adv)

            critic_update_loss = (
                critic_loss + value_loss
                if train_critic_during_align
                else zero_scalar
            )
            info.update({
                'critic/critic_loss': critic_loss.detach(),
                'critic/q_mean': q.mean().detach(),
                'critic/q_max': q.max().detach(),
                'critic/q_min': q.min().detach(),
                'critic/reward_mean': chunk_reward.mean().detach(),
                'critic/bellman_target_mean': bellman_target.mean().detach(),
                'critic/mc_target_mean': mc_target.mean().detach(),
                'critic/target_mix_lambda': target_mix_lambda,
                'critic/target_q_mean': target_q.mean().detach(),
                'critic/target_q_max': target_q.max().detach(),
                'critic/target_q_min': target_q.min().detach(),
                'critic/next_q_mean': next_q.mean().detach(),
                'critic/next_q_max': next_q.max().detach(),
                'critic/next_q_min': next_q.min().detach(),
                'value/value_loss': value_loss.detach(),
                'value/v_mean': value_v.mean().detach(),
                'value/q_mean': value_q.mean().detach(),
                'actor/total_loss': biflow_loss.detach(),
                'actor/actor_loss': zero_scalar.detach(),
                'actor/bc_loss': zero_scalar.detach(),
                'actor/iql_actor_loss': zero_scalar.detach(),
                'actor/adv_mean': adv.mean().detach(),
                'actor/adv_max': adv.max().detach(),
                'actor/weight_mean': weights.mean().detach(),
                'actor/weight_max': weights.max().detach(),
                'actor/mse': sampled_action_mse.detach(),
                'actor/mse_loss': sampled_action_mse_loss.detach(),
                'actor/lam': zero_scalar.detach(),
                'actor/p_norm': biflow_info['biflow/z_norm'].detach(),
                'actor/p_logprob_mean': zero_scalar.detach(),
                'actor/p_logdet_mean': zero_scalar.detach(),
                'actor/b_logdet_mean': zero_scalar.detach(),
                'actor/actor_entropy': zero_scalar.detach(),
                'actor/align_train_critic': torch.tensor(float(train_critic_during_align), device=self.device),
            })
            if q_prefix_agg is not None:
                info.update({
                    'critic/prefix_q_mean': q_prefix_agg.mean().detach(),
                    'critic/prefix_target_mean': prefix_targets.mean().detach(),
                    'critic/prefix_bootstrap_mean': prefix_bootstrap_values.mean().detach(),
                })
            if hubl_lambda_mean is not None:
                info.update({
                    'critic/hubl_lambda_mean': hubl_lambda_mean,
                    'critic/hubl_lambda_min': hubl_lambda_min,
                    'critic/hubl_lambda_max': hubl_lambda_max,
                })
            if hubl_reward_mean is not None and hubl_discount_mean is not None:
                info.update({
                    'critic/hubl_reward_mean': hubl_reward_mean,
                    'critic/hubl_discount_mean': hubl_discount_mean,
                })
            info.update(biflow_info)
            return critic_update_loss, biflow_loss, info

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            prefix_targets = None
            prefix_discounted_rewards = None
            prefix_bootstrap_values = None
            q_prefix_agg = None
            hubl_lambda_mean = None
            hubl_lambda_min = None
            hubl_lambda_max = None
            hubl_reward_mean = None
            hubl_discount_mean = None
            with torch.no_grad():
                zero_scalar = torch.tensor(0.0, device=self.device)
                value_loss = zero_scalar
                value_v = zero_scalar
                value_q = zero_scalar
                if is_imitation_mode and not train_critic_value_during_il:
                    target_q = zero_scalar
                    next_q = zero_scalar
                    chunk_reward = zero_scalar
                    bellman_target = zero_scalar
                    mc_target = zero_scalar
                    target_mix_lambda = 0.0
                elif is_iql_critic_mode:
                    (
                        prefix_targets,
                        prefix_discounted_rewards,
                        prefix_bootstrap_values,
                        _,
                        prefix_hubl_info,
                    ) = self._compute_iql_prefix_targets(
                        batch,
                        use_hubl=use_hubl,
                        state_language_embed=state_language_embed,
                    )
                    target_q = prefix_targets.mean(dim=-1)
                    next_q = prefix_bootstrap_values.mean(dim=-1)
                    chunk_reward = prefix_discounted_rewards[:, -1]
                    bellman_target = prefix_targets[:, -1]
                    mc_target = torch.zeros_like(target_q)
                    target_mix_lambda = prefix_hubl_info['target_mix_lambda']
                    hubl_lambda_mean = prefix_hubl_info['hubl_lambda_mean']
                    hubl_lambda_min = prefix_hubl_info['hubl_lambda_min']
                    hubl_lambda_max = prefix_hubl_info['hubl_lambda_max']
                    hubl_reward_mean = prefix_hubl_info['hubl_reward_mean']
                    hubl_discount_mean = prefix_hubl_info['hubl_discount_mean']
                else:
                    (
                        prefix_targets,
                        prefix_discounted_rewards,
                        prefix_bootstrap_values,
                        _,
                        prefix_hubl_info,
                    ) = self._compute_prefix_td_targets(
                        batch,
                        use_hubl=use_hubl,
                        language_tokens=language_tokens,
                        state_language_embed=state_language_embed,
                    )
                    target_q = prefix_targets.mean(dim=-1)
                    next_q = prefix_bootstrap_values.mean(dim=-1)
                    chunk_reward = prefix_discounted_rewards[:, -1]
                    bellman_target = prefix_targets[:, -1]
                    mc_target = torch.zeros_like(target_q)
                    target_mix_lambda = prefix_hubl_info['target_mix_lambda']
                    hubl_lambda_mean = prefix_hubl_info['hubl_lambda_mean']
                    hubl_lambda_min = prefix_hubl_info['hubl_lambda_min']
                    hubl_lambda_max = prefix_hubl_info['hubl_lambda_max']
                    hubl_reward_mean = prefix_hubl_info['hubl_reward_mean']
                    hubl_discount_mean = prefix_hubl_info['hubl_discount_mean']
            if is_imitation_mode and not train_critic_value_during_il:
                q = zero_scalar
                critic_loss = zero_scalar
            else:
                q = self._critic_call(
                    self.raw_model.critic,
                    batch['observations'],
                    batch['actions'],
                    language_embed=state_language_embed,
                    proprioceptions=proprioceptions,
                )
                q_prefix = self._critic_forward_prefix(
                    self.raw_model.critic,
                    batch['observations'],
                    batch['actions'],
                    language_embed=state_language_embed,
                    proprioceptions=proprioceptions,
                )
                prefix_len = min(q_prefix.shape[-1], prefix_targets.shape[1])
                q_prefix = q_prefix[..., :prefix_len]
                prefix_targets = prefix_targets[:, :prefix_len]
                prefix_discounted_rewards = prefix_discounted_rewards[:, :prefix_len]
                prefix_bootstrap_values = prefix_bootstrap_values[:, :prefix_len]
                q_prefix_agg = self._aggregate_qs(q_prefix)
                critic_loss = self._masked_mean(
                    (q_prefix - prefix_targets.unsqueeze(0)).pow(2),
                    action_valid_mask[:, :prefix_len],
                    'critic prefix loss',
                )
                target_q = prefix_targets.mean(dim=-1)
                next_q = prefix_bootstrap_values.mean(dim=-1)
                chunk_reward = prefix_discounted_rewards[:, -1]
                bellman_target = prefix_targets[:, -1]
                if is_iql_critic_mode:
                    with torch.no_grad():
                        value_q = self._aggregate_qs(
                            self._critic_call(
                                self.raw_model.target_critic,
                                batch['observations'],
                                batch['actions'],
                                language_embed=state_language_embed,
                                proprioceptions=proprioceptions,
                            )
                        )
                    value_v = self._value(
                        batch['observations'],
                        language_embed=state_language_embed,
                        proprioceptions=proprioceptions,
                    )
                    value_loss = self._masked_mean(
                        self._expectile_loss(
                            value_q - value_v,
                            float(self.config.get('iql_expectile', 0.8)),
                        ),
                        action_valid_mask.any(dim=1),
                        'value loss',
                    )
            critic_update_loss = (
                critic_loss + value_loss
                if is_iql_critic_mode
                else critic_loss
            )
        info.update({
            'critic/critic_loss': critic_loss.detach(),
            'critic/q_mean': q.mean().detach(),
            'critic/q_max': q.max().detach(),
            'critic/q_min': q.min().detach(),
            'critic/reward_mean': chunk_reward.mean().detach(),
            'critic/bellman_target_mean': bellman_target.mean().detach(),
            'critic/mc_target_mean': mc_target.mean().detach(),
            'critic/target_mix_lambda': target_mix_lambda,
            'critic/target_q_mean': target_q.mean().detach(),
            'critic/target_q_max': target_q.max().detach(),
            'critic/target_q_min': target_q.min().detach(),
            'critic/next_q_mean': next_q.mean().detach(),
            'critic/next_q_max': next_q.max().detach(),
            'critic/next_q_min': next_q.min().detach(),
            'value/value_loss': value_loss.detach(),
            'value/v_mean': value_v.mean().detach(),
            'value/q_mean': value_q.mean().detach(),
            'critic/il_train_critic_value': torch.tensor(
                float(train_critic_value_during_il),
                device=self.device,
            ),
        })
        if q_prefix_agg is not None:
            info.update({
                'critic/prefix_q_mean': q_prefix_agg.mean().detach(),
                'critic/prefix_target_mean': prefix_targets.mean().detach(),
                'critic/prefix_bootstrap_mean': prefix_bootstrap_values.mean().detach(),
            })
        if hubl_lambda_mean is not None:
            info.update({
                'critic/hubl_lambda_mean': hubl_lambda_mean,
                'critic/hubl_lambda_min': hubl_lambda_min,
                'critic/hubl_lambda_max': hubl_lambda_max,
            })
        if hubl_reward_mean is not None and hubl_discount_mean is not None:
            info.update({
                'critic/hubl_reward_mean': hubl_reward_mean,
                'critic/hubl_discount_mean': hubl_discount_mean,
            })
        actor_critic_requires_grad = []
        if full_update:
            # Actor Q objectives need gradients through Q with respect to sampled
            # actions, but must not update critic/value parameters.
            actor_critic_requires_grad = [
                (parameter, parameter.requires_grad)
                for _, parameter in self._online_critic_value_named_parameters()
            ]
            for parameter, _ in actor_critic_requires_grad:
                parameter.requires_grad_(False)
            action_seq = clean_action_seq
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                actor_conditioning = self._build_actor_conditioning(
                    batch['observations'],
                    language_tokens,
                    proprioceptions=proprioceptions,
                )
                biflow_loss = torch.tensor(0.0, device=self.device)
                biflow_info = {}
                sampled_actor_available = not (
                    (is_iql_mode or is_imitation_mode)
                    and float(self.config.get('alpha_actor_entropy', 0.0)) == 0.0
                )
                reverse_q_loss = torch.tensor(0.0, device=self.device)
                if sampled_actor_available:
                    b_noise = self.prior.sample(batch['observations'].shape[0])
                    b_actions, b_logdets = self.raw_model.actor.reverse(
                        x=b_noise,
                        **actor_conditioning.kwargs(),
                        guidance=0.0,
                    )
                    q = self._aggregate_qs(
                        self._critic_call(
                            self.raw_model.critic,
                            batch['observations'],
                            b_actions,
                            language_embed=state_language_embed,
                            proprioceptions=proprioceptions,
                        )
                    )
                    entropy_loss = (self.prior.log_prob(b_noise) - b_logdets).mean()
                else:
                    b_actions = None
                    b_logdets = None
                    q = torch.tensor(0.0, device=self.device)
                    entropy_loss = torch.tensor(0.0, device=self.device)

                if is_imitation_mode:
                    q_for_norm = None
                    actor_loss = torch.tensor(0.0, device=self.device)
                    lam = torch.tensor(0.0, device=self.device)
                elif is_iql_mode:
                    q_for_norm = None
                    actor_loss = torch.tensor(0.0, device=self.device)
                    lam = torch.tensor(0.0, device=self.device)
                else:
                    assert b_actions is not None
                    q_for_norm = q
                    reverse_q_loss = -q.mean()
                    actor_loss = reverse_q_loss
                    if self.config['normalize_q_loss']:
                        lam = (1.0 / q_for_norm.abs().mean()).detach()
                        actor_loss = lam * actor_loss
                    else:
                        lam = torch.tensor(0.0, device=self.device)

                p_noise, p_logdets = self.raw_model.actor.forward(
                    x=flow_action_seq,
                    **actor_conditioning.kwargs(),
                )
                p_log_prob = self.prior.log_prob(p_noise)
                log_pi = p_log_prob + p_logdets
                nll_loss = -self._masked_mean(log_pi, valid_action_samples, 'actor NLL')
                iql_actor_loss = torch.tensor(0.0, device=self.device)
                adv = torch.zeros(batch['observations'].shape[0], device=self.device)
                weights = torch.ones_like(adv)
                if is_iql_mode:
                    with torch.no_grad():
                        data_q = self._aggregate_qs(
                            self._critic_call(
                                self.raw_model.target_critic,
                                batch['observations'],
                                action_seq,
                                language_embed=state_language_embed,
                                proprioceptions=proprioceptions,
                            )
                        )
                        data_v = self._value(
                            batch['observations'],
                            language_embed=state_language_embed,
                            proprioceptions=proprioceptions,
                        )
                        adv = data_q - data_v
                        weights = self._iql_advantage_weights(adv)
                    iql_actor_loss = -self._masked_mean(
                        weights.detach() * log_pi,
                        valid_action_samples,
                        'IQL actor loss',
                    )
                    actor_loss = iql_actor_loss * float(self.config.get('iql_actor_loss_scale', 1.0))
                    bc_loss = nll_loss * float(self.config.get('iql_bc_alpha', 0.0))
                else:
                    bc_loss = nll_loss * self.config['alpha_actor']

            total_actor_loss = self._compose_total_actor_loss(
                actor_loss,
                bc_loss,
                entropy_loss,
                biflow_loss,
            )

            if sampled_actor_available:
                metric_b_actions = b_actions.float()
                b_acts_last = metric_b_actions[:, -1] if metric_b_actions.ndim == 3 else metric_b_actions
                gt_acts_last = clean_action_seq[:, -1] if clean_action_seq.ndim == 3 else clean_action_seq
                sampled_action_mse = (b_acts_last - gt_acts_last).pow(2).sum(dim=-1).mean()
                sampled_action_mse_loss = (metric_b_actions - clean_action_seq).pow(2).mean()
                sampled_b_logdet_mean = b_logdets.mean()
            else:
                sampled_action_mse = torch.tensor(float('nan'), device=self.device)
                sampled_action_mse_loss = torch.tensor(float('nan'), device=self.device)
                sampled_b_logdet_mean = torch.tensor(float('nan'), device=self.device)

            info.update({
                'data/action_valid_fraction': action_valid_mask.mean().detach(),
                'data/fully_valid_action_chunk_fraction': valid_action_samples.mean().detach(),
                'actor/total_loss': total_actor_loss.detach(),
                'actor/actor_loss': actor_loss.detach(),
                'actor/bc_loss': bc_loss.detach(),
                'actor/iql_actor_loss': iql_actor_loss.detach(),
                'actor/adv_mean': adv.mean().detach(),
                'actor/adv_max': adv.max().detach(),
                'actor/weight_mean': weights.mean().detach(),
                'actor/weight_max': weights.max().detach(),
                'actor/mse': sampled_action_mse.detach(),
                'actor/mse_loss': sampled_action_mse_loss.detach(),
                'actor/lam': lam.detach(),
                'actor/p_norm': p_noise.pow(2).mean().detach(),
                'actor/p_logprob_mean': p_log_prob.mean().detach(),
                'actor/p_logdet_mean': p_logdets.mean().detach(),
                'actor/b_logdet_mean': sampled_b_logdet_mean.detach(),
                'actor/actor_entropy': (-entropy_loss).detach(),
                'actor/reverse_q_loss': reverse_q_loss.detach(),
            })
            info.update(biflow_info)
            if (not is_imitation_mode) and q_for_norm is not None:
                info['actor/prefix_q_mean'] = q_for_norm.mean().detach()

            for parameter, requires_grad in actor_critic_requires_grad:
                parameter.requires_grad_(requires_grad)
            return critic_update_loss, total_actor_loss, info

        return critic_update_loss, None, info

    def _assert_finite_gradients(self):
        nonfinite_names = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad.detach()).all().item())
        ]
        local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        local_status = torch.tensor(
            [int(local_rank), int(not nonfinite_names)],
            device=self.device,
            dtype=torch.int32,
        )
        failing_ranks = []
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.zeros_like(local_status) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_status)
            failing_ranks = [int(item[0].item()) for item in gathered if not bool(item[1].item())]
        elif nonfinite_names:
            failing_ranks = [0]
        if failing_ranks:
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            local_detail = nonfinite_names[:8] if nonfinite_names else ["nonfinite gradient on another rank"]
            raise RuntimeError(
                "Non-finite gradient detected before optimizer.step; aborting training. "
                f"Failing ranks: {failing_ranks}. Local parameters: {local_detail}."
            )

    def update(self, batch, full_update=True, step: int | None = None):
        self.model.train()
        self.update_step = self.update_step + 1 if step is None else int(step)
        if self.update_step <= 0:
            raise ValueError(f'update step must be positive, got {self.update_step}.')

        current_actor_lr = self._apply_optimizer_lr_schedule('actor', self.update_step)
        current_critic_lr = self._apply_optimizer_lr_schedule('critic', self.update_step)
        current_vlm_lr, vlm_frozen = self._apply_vlm_schedule(
            self.update_step,
            current_actor_lr,
        )

        warmup_steps = max(0, int(self.config.get('iql_critic_warmup_steps', 1000)))
        warmup_active = self._is_iql_mode() and self.update_step <= warmup_steps
        update_actor = bool(full_update) and not warmup_active
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        critic_loss, actor_loss, info = self._training_total_loss(
            batch,
            full_update=update_actor,
        )
        finite_loss = torch.isfinite(critic_loss.detach()).all()
        if actor_loss is not None:
            finite_loss = torch.logical_and(finite_loss, torch.isfinite(actor_loss.detach()).all())
        finite_loss = finite_loss.to(device=self.device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
        if not bool(finite_loss.item()):
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                'Non-finite loss detected during VINFTorchAgent.update; '
                'aborting instead of skipping optimizer.step.'
            )

        is_actor_only_mode = (
            (self._is_imitation_mode() and not self._train_critic_value_during_il())
            or (self._is_biflow_align_mode() and not self._train_critic_during_biflow_align())
        )
        critic_should_step = not is_actor_only_mode
        actor_should_step = update_actor and actor_loss is not None
        if critic_should_step and self.critic_optimizer is None:
            raise RuntimeError('Critic/value loss is active but critic_optimizer is unavailable.')
        if actor_should_step and self.actor_optimizer is None:
            raise RuntimeError('Actor loss is active but actor_optimizer is unavailable.')

        backward_losses = []
        if critic_should_step:
            backward_losses.append(critic_loss)
        if actor_should_step:
            backward_losses.append(actor_loss)
        if not backward_losses:
            raise RuntimeError('No active loss is available for this update.')
        backward_loss = sum(backward_losses)
        use_amp = bool(self.config.get('use_amp', False)) and self.device.type == 'cuda'
        if use_amp:
            self.scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        stepped_optimizers = []
        if actor_should_step:
            stepped_optimizers.append(('actor', self.actor_optimizer))
        if critic_should_step:
            stepped_optimizers.append(('critic', self.critic_optimizer))
        if use_amp:
            for _, optimizer in stepped_optimizers:
                self.scaler.unscale_(optimizer)
        self._assert_finite_gradients()

        clip_groups = []
        for optimizer_name, _ in stepped_optimizers:
            if optimizer_name != 'actor':
                clip_groups.append((
                    optimizer_name,
                    self._optimizer_named_parameters(optimizer_name),
                    self._optimizer_max_grad_norm(optimizer_name),
                ))
                continue

            actor_parameters = self._optimizer_group_named_parameters('actor', 'actor')
            context_parameters = self._optimizer_group_named_parameters(
                'actor', 'context_adapter'
            )
            vlm_parameters = self._optimizer_group_named_parameters('actor', 'vlm')
            context_max_grad_norm = float(
                self.config.get('context_adapter_max_grad_norm', -1.0)
            )
            vlm_max_grad_norm = float(self.config.get('vlm_max_grad_norm', -1.0))
            split_context = context_max_grad_norm >= 0.0 and bool(context_parameters)
            split_vlm = vlm_max_grad_norm >= 0.0 and bool(vlm_parameters)
            if not split_context and not split_vlm:
                clip_groups.append((
                    'actor',
                    self._optimizer_named_parameters('actor'),
                    self._optimizer_max_grad_norm('actor'),
                ))
                continue

            remaining_parameters = list(actor_parameters)
            if not split_context:
                remaining_parameters.extend(context_parameters)
            if not split_vlm:
                remaining_parameters.extend(vlm_parameters)
            if remaining_parameters:
                clip_groups.append((
                    'actor_non_vlm' if split_vlm else 'actor_remainder',
                    remaining_parameters,
                    self._optimizer_max_grad_norm('actor'),
                ))
            if split_context:
                clip_groups.append((
                    'context_adapter',
                    context_parameters,
                    context_max_grad_norm,
                ))
            if split_vlm:
                clip_groups.append(('vlm', vlm_parameters, vlm_max_grad_norm))

        preclip_norms = {}
        postclip_norms = {}
        for clip_group_name, named_parameters, max_grad_norm in clip_groups:
            if max_grad_norm > 0.0:
                preclip_norm = _clip_grad_norm_finite_(named_parameters, max_grad_norm)
            else:
                preclip_norm = self._named_grad_norm(named_parameters)
            preclip_norms[clip_group_name] = preclip_norm.to(self.device)
            postclip_norms[clip_group_name] = self._named_grad_norm(named_parameters).to(
                self.device
            )

        grad_max = None
        grad_min = None
        for _, parameter in self.raw_model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            current_max = parameter.grad.detach().max()
            current_min = parameter.grad.detach().min()
            grad_max = current_max if grad_max is None else torch.maximum(grad_max, current_max)
            grad_min = current_min if grad_min is None else torch.minimum(grad_min, current_min)

        if use_amp:
            for _, optimizer in stepped_optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            for _, optimizer in stepped_optimizers:
                optimizer.step()

        if critic_should_step:
            self.raw_model.soft_update_target(self.config['tau'])
        if actor_should_step and bool(self.config.get('biflow_use_ema', False)):
            self.raw_model.soft_update_reverse_model_ema(
                float(self.config.get('biflow_ema_decay', 0.9999))
            )

        zero = torch.tensor(0.0, device=self.device)
        actor_flow_preclip = preclip_norms.get(
            'actor_non_vlm', preclip_norms.get('actor_remainder', zero)
        )
        context_adapter_preclip = preclip_norms.get('context_adapter', zero)
        actor_non_vlm_preclip = (
            actor_flow_preclip.float().square()
            + context_adapter_preclip.float().square()
        ).sqrt()
        vlm_preclip = preclip_norms.get('vlm', zero)
        actor_preclip = preclip_norms.get(
            'actor',
            (
                actor_non_vlm_preclip.float().square()
                + vlm_preclip.float().square()
            ).sqrt(),
        )
        critic_preclip = preclip_norms.get('critic', zero)
        actor_flow_postclip = postclip_norms.get(
            'actor_non_vlm', postclip_norms.get('actor_remainder', zero)
        )
        context_adapter_postclip = postclip_norms.get('context_adapter', zero)
        actor_non_vlm_postclip = (
            actor_flow_postclip.float().square()
            + context_adapter_postclip.float().square()
        ).sqrt()
        vlm_postclip = postclip_norms.get('vlm', zero)
        actor_postclip = postclip_norms.get(
            'actor',
            (
                actor_non_vlm_postclip.float().square()
                + vlm_postclip.float().square()
            ).sqrt(),
        )
        critic_postclip = postclip_norms.get('critic', zero)
        total_postclip = (actor_postclip.float().square() + critic_postclip.float().square()).sqrt()
        context_adapter_lr = 0.0
        if self.actor_optimizer is not None:
            for param_group in self.actor_optimizer.param_groups:
                if param_group.get('name') == 'context_adapter':
                    context_adapter_lr = float(param_group['lr'])
                    break
        info.update({
            'update/skipped_nonfinite': zero,
            'update/step': torch.tensor(float(self.update_step), device=self.device),
            'update/iql_warmup_active': torch.tensor(float(warmup_active), device=self.device),
            'update/iql_warmup_steps': torch.tensor(float(warmup_steps), device=self.device),
            'update/actor_stepped': torch.tensor(float(actor_should_step), device=self.device),
            'update/critic_stepped': torch.tensor(float(critic_should_step), device=self.device),
            'optimizer/actor_lr': torch.tensor(
                float(self.actor_optimizer.param_groups[0]['lr']) if self.actor_optimizer is not None else 0.0,
                device=self.device,
            ),
            'optimizer/critic_lr': torch.tensor(
                float(current_critic_lr),
                device=self.device,
            ),
            'optimizer/vlm_lr': torch.tensor(float(current_vlm_lr), device=self.device),
            'optimizer/context_adapter_lr': torch.tensor(
                context_adapter_lr,
                device=self.device,
            ),
            'optimizer/vlm_frozen': torch.tensor(float(vlm_frozen), device=self.device),
            'grad/max': grad_max.detach().cpu() if grad_max is not None else torch.tensor(0.0),
            'grad/min': grad_min.detach().cpu() if grad_min is not None else torch.tensor(0.0),
            'grad/norm': total_postclip.detach().cpu(),
            'grad/preclip_norm': (
                actor_preclip.float().square() + critic_preclip.float().square()
            ).sqrt().detach().cpu(),
            'grad/actor_preclip_norm': actor_preclip.detach().cpu(),
            'grad/actor_postclip_norm': actor_postclip.detach().cpu(),
            'grad/actor_non_vlm_preclip_norm': actor_non_vlm_preclip.detach().cpu(),
            'grad/actor_non_vlm_postclip_norm': actor_non_vlm_postclip.detach().cpu(),
            'grad/actor_flow_preclip_norm': actor_flow_preclip.detach().cpu(),
            'grad/actor_flow_postclip_norm': actor_flow_postclip.detach().cpu(),
            'grad/context_adapter_preclip_norm': context_adapter_preclip.detach().cpu(),
            'grad/context_adapter_postclip_norm': context_adapter_postclip.detach().cpu(),
            'grad/vlm_preclip_norm': vlm_preclip.detach().cpu(),
            'grad/vlm_postclip_norm': vlm_postclip.detach().cpu(),
            'grad/critic_preclip_norm': critic_preclip.detach().cpu(),
            'grad/critic_postclip_norm': critic_postclip.detach().cpu(),
        })
        return self, {
            key: (
                value.item()
                if torch.is_tensor(value) and value.numel() == 1
                else value.detach().cpu().numpy()
                if torch.is_tensor(value)
                else value
            )
            for key, value in info.items()
        }

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizers': {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            'update_step': int(self.update_step),
            'config': self.config,
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
        }

    def load_state_dict(self, state_dict):
        model_state = state_dict['model']
        self.model.load_state_dict(model_state)
        optimizer_states = state_dict.get('optimizers')
        if optimizer_states is not None:
            for name, optimizer in self.optimizers.items():
                saved_state = optimizer_states.get(name)
                if saved_state is None:
                    raise ValueError(f'Checkpoint is missing `{name}` optimizer state.')
                optimizer.load_state_dict(saved_state)
        elif 'optimizer' in state_dict:
            raise ValueError(
                'Legacy shared optimizer checkpoints cannot resume release training.'
            )
        self.update_step = int(state_dict.get('update_step', self.update_step))
        if self.scaler is not None and state_dict.get('scaler') is not None:
            self.scaler.load_state_dict(state_dict['scaler'])


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            lr=1e-4,
            actor_lr=-1.0,
            critic_lr=-1.0,
            optimizer_type='adam',
            optimizer_weight_decay=0.0,
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            actor_lr_warmup_steps=0,
            critic_lr_warmup_steps=0,
            actor_lr_schedule='constant',
            critic_lr_schedule='constant',
            actor_lr_schedule_steps=0,
            critic_lr_schedule_steps=0,
            actor_lr_schedule_delay_steps=0,
            critic_lr_schedule_delay_steps=0,
            lr_cosine_min_ratio=0.0,
            lr_warmup_start_factor=0.0,
            batch_size=256,
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            q_agg='min',
            critic_hidden_dim=512,
            critic_num_layers=2,
            critic_num_heads=8,
            critic_dropout=0.0,
            critic_prefix_reduce='last',
            critic_use_language_conditioning=False,
            value_use_language_conditioning=False,
            share_critic_value_state=False,
            reinitialize_critic_value_on_pretrain=False,
            critic_language_embed_dim=512,
            value_language_embed_dim=512,
            train_mode='iql',
            iql_expectile=0.8,
            iql_temperature=0.1,
            iql_temperature_mode='multiply',
            iql_adv_clip=100.0,
            iql_actor_loss_scale=1.0,
            iql_bc_alpha=0.0,
            iql_critic_warmup_steps=1000,
            il_train_critic_value=False,
            alpha_actor=0.0,
            alpha_actor_entropy=0.0,
            use_biflow=False,
            biflow_channels=512,
            biflow_num_layers=2,
            biflow_num_heads=8,
            biflow_num_condition_tokens=1,
            biflow_num_guidance_tokens=1,
            biflow_use_ema=False,
            biflow_sample_use_ema=True,
            biflow_ema_decay=0.9999,
            biflow_align_weight=1.0,
            biflow_action_weight=1.0,
            biflow_norm_p=0.0,
            biflow_norm_eps=0.001,
            biflow_noise_level=-1.0,
            biflow_prior_weight=1.0,
            biflow_prior_batch_fraction=1.0,
            biflow_prior_temperature=1.0,
            biflow_prior_teacher_guidance=1.1,
            biflow_allow_data_only_align=False,
            biflow_guidance_min=0.0,
            biflow_guidance_max=0.5,
            biflow_eval_temperature=0.0,
            biflow_eval_guidance=0.0,
            biflow_eval_guidance_final=-1.0,
            biflow_td_temperature=1.0,
            biflow_td_guidance=0.0,
            biflow_td_guidance_final=-1.0,
            biflow_label_drop_rate=0.0,
            biflow_align_freeze_teacher_requires_grad=True,
            biflow_freeze_teacher_in_align=True,
            biflow_align_train_critic=False,
            biflow_align_freeze_actor=True,
            bc_rep_size=256,
            bc_num_blocks=6,
            bc_channels=1152,
            layers_per_block=(2, 2, 2, 2, 2, 18),
            num_heads=16,
            simflow_log_scale_clip=5.0,
            use_language_conditioning=True,
            conditioning_mode='context',
            vlm_fuse=True,
            language_model_path='/path/to/clip-vit-base-patch32',
            language_max_length=77,
            smolvlm_model_path='/path/to/SmolVLM-500M-Instruct',
            vlm_image_size=384,
            vlm_language_max_length=50,
            vlm_freeze_steps=1000,
            vlm_lr_multiplier=0.1,
            vlm_gradient_checkpointing=True,
            vlm_freeze_text_model=False,
            vlm_image_normalization='processor',
            vlm_train_last_n_vision_layers=-1,
            label_drop_prob=0.1,
            cfg=1.5,
            action_len=8,
            obs_horizon=2,
            actor_context_obs_horizon=0,
            normalize_q_loss=False,
            use_amp=False,
            max_grad_norm=1.0,
            actor_max_grad_norm=-1.0,
            critic_max_grad_norm=-1.0,
            vlm_max_grad_norm=-1.0,
            context_adapter_lr=-1.0,
            context_adapter_max_grad_norm=-1.0,
            data_noise=0.05,
            mc_target_lambda=0.0,
            use_hubl=False,
            hubl_lambda_type='rank',
            hubl_alpha=0.1,
            denoising_lr=0.0,
            eval_temperature=1.0,
            image_num_views=2,
            impala_adaptive_pool_hw='',
            robomimic_use_proprioception=False,
            robomimic_proprio_dim=9,
            use_proprioception=False,
            proprio_dim=8,
            proprio_injection='vlm_context_token',
            proprio_q01=(-1.0,) * 8,
            proprio_q99=(1.0,) * 8,
            robomimic_use_crop_augmentation=False,
            robomimic_crop_size=76,
            robomimic_resnet_input_size=224,
            robomimic_max_obs_horizon=8,
            robomimic_resnet_pretrained_weights='IMAGENET1K_V1',
            actor_encoder=ml_collections.config_dict.placeholder(str),
            critic_encoder='impala',
        )
    )
    return config
