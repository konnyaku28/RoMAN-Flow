from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_tensor(x, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def batch_flatten(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 2:
        return x
    return x.reshape(x.shape[0], -1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class LanguageStateFusion(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        language_dim: int = 512,
        hidden_dim: int | None = None,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.language_dim = int(language_dim)
        self.hidden_dim = int(hidden_dim or max(self.obs_dim, self.language_dim))
        self.obs_norm = nn.LayerNorm(self.obs_dim) if layer_norm else Identity()
        self.language_norm = nn.LayerNorm(self.language_dim) if layer_norm else Identity()
        self.fusion = nn.Sequential(
            nn.Linear(self.obs_dim + self.language_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.obs_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, obs_feat: torch.Tensor, language_embed: torch.Tensor | None) -> torch.Tensor:
        if language_embed is None:
            raise ValueError('LanguageStateFusion requires `language_embed` when language state conditioning is enabled.')
        if obs_feat.ndim != 2:
            raise ValueError(f'LanguageStateFusion expects obs_feat shape [B, C], got {tuple(obs_feat.shape)}.')
        if language_embed.ndim != 2:
            raise ValueError(
                f'LanguageStateFusion expects language_embed shape [B, C], got {tuple(language_embed.shape)}.'
            )
        if obs_feat.shape[0] != language_embed.shape[0]:
            raise ValueError(
                f'LanguageStateFusion batch mismatch: obs_feat={tuple(obs_feat.shape)}, '
                f'language_embed={tuple(language_embed.shape)}.'
            )
        if obs_feat.shape[-1] != self.obs_dim:
            raise ValueError(f'LanguageStateFusion expected obs_dim={self.obs_dim}, got {obs_feat.shape[-1]}.')
        if language_embed.shape[-1] != self.language_dim:
            raise ValueError(
                f'LanguageStateFusion expected language_dim={self.language_dim}, '
                f'got {language_embed.shape[-1]}.'
            )
        language_embed = language_embed.to(device=obs_feat.device, dtype=obs_feat.dtype)
        fused_input = torch.cat([self.obs_norm(obs_feat), self.language_norm(language_embed)], dim=-1)
        return obs_feat + self.fusion(fused_input)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activate_final: bool = False, layer_norm: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.feature = None

        dims = [self.input_dim, *self.hidden_dims]
        for i in range(len(self.hidden_dims)):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if self.layer_norm and (i + 1 < len(self.hidden_dims) or self.activate_final):
                self.norms.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = batch_flatten(x)
        norm_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = F.gelu(x)
                if self.layer_norm:
                    x = self.norms[norm_idx](x)
                    norm_idx += 1
            if i == len(self.hidden_dims) - 2:
                self.feature = x.detach()
        return x


class VectorObservationEncoder(nn.Module):
    def __init__(self, input_size: Optional[int], rep_size: int, encoder: Optional[nn.Module] = None):
        super().__init__()
        if input_size is None:
            raise ValueError('VectorObservationEncoder requires an explicit input_size.')
        self.rep_size = rep_size
        self.encoder = encoder
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.SiLU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.SiLU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.SiLU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.SiLU(),
            
            nn.Linear(512, rep_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            x = self.encoder(x)
        else:
            x = batch_flatten(x.float())
        return self.net(x)


class Value(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        num_ensembles: int = 2,
        encoder: Optional[nn.Module] = None,
        language_embed_dim: int | None = None,
        language_obs_dim: int | None = None,
        proprio_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.layer_norm = layer_norm
        self.num_ensembles = num_ensembles
        self.encoder = encoder
        self.language_embed_dim = None if language_embed_dim is None else int(language_embed_dim)
        self.language_obs_dim = int(language_obs_dim or input_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            from utils.multimodal_context import GatedProprioceptionFusion

            self.proprio_fusion = GatedProprioceptionFusion(
                proprio_dim=self.proprio_dim,
                feature_dim=self.language_obs_dim,
            )
        else:
            self.proprio_fusion = None
        self.state_fusion = (
            LanguageStateFusion(
                obs_dim=self.language_obs_dim,
                language_dim=self.language_embed_dim,
                layer_norm=self.layer_norm,
            )
            if self.language_embed_dim is not None and self.language_embed_dim > 0
            else None
        )
        self.value_nets = nn.ModuleList(
            [MLP(input_dim=self.input_dim, hidden_dims=(*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm) for _ in range(self.num_ensembles)]
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is not None:
            if self.proprio_fusion is not None or proprioceptions is None:
                obs_feat = self.encoder(observations)
            else:
                obs_feat = self.encoder(
                    observations,
                    proprioceptions=proprioceptions,
                )
        else:
            if observations.ndim == 3 and observations.shape[-1] == self.input_dim:
                obs_feat = observations.float()
            else:
                obs_feat = batch_flatten(observations.float())
        if obs_feat.ndim == 3:
            if observation_mask is None:
                obs_feat = obs_feat.mean(dim=1)
            else:
                mask = observation_mask.to(device=obs_feat.device, dtype=obs_feat.dtype)
                if tuple(mask.shape) != tuple(obs_feat.shape[:2]):
                    raise ValueError(
                        'observation_mask must match token axes '
                        f'{tuple(obs_feat.shape[:2])}, got {tuple(mask.shape)}.'
                    )
                denominator = mask.sum(dim=1, keepdim=True).clamp_min_(1.0)
                obs_feat = (obs_feat * mask.unsqueeze(-1)).sum(dim=1) / denominator
        if self.proprio_fusion is not None:
            if proprioceptions is None:
                raise ValueError('State-conditioned value requires proprioceptions.')
            obs_feat = self.proprio_fusion(obs_feat, proprioceptions)
        if self.state_fusion is not None:
            obs_feat = self.state_fusion(obs_feat, language_embed)
        inputs = [obs_feat]
        if actions is not None:
            inputs.append(batch_flatten(actions.float()))
        x = torch.cat(inputs, dim=-1)
        qs = [net(x).squeeze(-1) for net in self.value_nets]
        return torch.stack(qs, dim=0)


class ChunkTransformerCriticHead(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.action_len = action_len
        self.hidden_dim = hidden_dim
        self.state_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # Keep the legacy parameter shape so vector-input checkpoints remain loadable.
        self.pos_embed = nn.Parameter(torch.zeros(1, action_len + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim) if layer_norm else Identity()
        self.q_head = nn.Linear(hidden_dim, 1)

    def _prepare_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, int]:
        actions = actions.float()
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if actions.ndim != 3:
            raise ValueError(
                f'ChunkTransformerCriticHead expects actions with shape [B, D] or [B, T, D], got {tuple(actions.shape)}'
            )

        valid_len = actions.shape[1]
        if valid_len > self.action_len:
            actions = actions[:, : self.action_len]
            valid_len = self.action_len
        elif valid_len < self.action_len:
            pad_source = actions[:, -1:] if valid_len > 0 else torch.zeros(
                actions.shape[0], 1, actions.shape[-1], device=actions.device, dtype=actions.dtype
            )
            pad = pad_source.repeat(1, self.action_len - valid_len, 1)
            actions = torch.cat([actions, pad], dim=1)

        return actions, valid_len

    def _build_causal_mask(
        self,
        num_observation_tokens: int,
        num_action_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_len = num_observation_tokens + num_action_tokens
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask[:num_observation_tokens, :num_observation_tokens] = False
        mask[num_observation_tokens:, :num_observation_tokens] = False
        mask[num_observation_tokens:, num_observation_tokens:] = torch.triu(
            torch.ones(
                num_action_tokens,
                num_action_tokens,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        obs_feat: torch.Tensor,
        actions: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions, valid_len = self._prepare_actions(actions)
        if obs_feat.ndim == 2:
            observation_tokens = self.state_proj(obs_feat).unsqueeze(1)
        elif obs_feat.ndim == 3:
            observation_tokens = self.state_proj(obs_feat)
        else:
            raise ValueError(
                'ChunkTransformerCriticHead expects observation features [B, D] '
                f'or [B, P, D], got {tuple(obs_feat.shape)}.'
            )
        if observation_mask is None:
            observation_mask = torch.ones(
                observation_tokens.shape[:2],
                device=observation_tokens.device,
                dtype=torch.bool,
            )
        else:
            observation_mask = observation_mask.to(
                device=observation_tokens.device,
                dtype=torch.bool,
            )
            if tuple(observation_mask.shape) != tuple(observation_tokens.shape[:2]):
                raise ValueError(
                    'observation_mask must match critic observation tokens '
                    f'{tuple(observation_tokens.shape[:2])}, got '
                    f'{tuple(observation_mask.shape)}.'
                )
        action_tokens = self.action_proj(actions) + self.pos_embed[:, 1 : self.action_len + 1]
        observation_tokens = observation_tokens + self.pos_embed[:, :1]
        tokens = torch.cat([observation_tokens, action_tokens], dim=1)
        causal_mask = self._build_causal_mask(
            observation_tokens.shape[1],
            action_tokens.shape[1],
            tokens.device,
        )
        key_padding_mask = torch.cat(
            [
                ~observation_mask,
                torch.zeros(
                    action_tokens.shape[:2],
                    device=tokens.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        prefix_hidden = self.output_norm(hidden[:, observation_tokens.shape[1] :])
        prefix_q = self.q_head(prefix_hidden).squeeze(-1)
        return prefix_q[:, :valid_len]


class ChunkTransformerValue(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_len: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        num_ensembles: int = 2,
        encoder: Optional[nn.Module] = None,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        prefix_reduce: str = 'mean',
        language_embed_dim: int | None = None,
        proprio_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_len = action_len
        self.hidden_dims = tuple(hidden_dims)
        self.layer_norm = layer_norm
        self.num_ensembles = num_ensembles
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prefix_reduce = prefix_reduce
        self.language_embed_dim = None if language_embed_dim is None else int(language_embed_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            from utils.multimodal_context import GatedProprioceptionFusion

            self.proprio_fusion = GatedProprioceptionFusion(
                proprio_dim=self.proprio_dim,
                feature_dim=self.obs_dim,
            )
        else:
            self.proprio_fusion = None
        self.state_fusion = (
            LanguageStateFusion(
                obs_dim=self.obs_dim,
                language_dim=self.language_embed_dim,
                layer_norm=self.layer_norm,
            )
            if self.language_embed_dim is not None and self.language_embed_dim > 0
            else None
        )
        self.value_nets = nn.ModuleList(
            [
                ChunkTransformerCriticHead(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    action_len=self.action_len,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    layer_norm=self.layer_norm,
                )
                for _ in range(self.num_ensembles)
            ]
        )

    def _encode_observations(
        self,
        observations: torch.Tensor,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is not None:
            if self.proprio_fusion is not None or proprioceptions is None:
                return self.encoder(observations)
            return self.encoder(
                observations,
                proprioceptions=proprioceptions,
            )
        if observations.ndim == 3 and observations.shape[-1] == self.obs_dim:
            return observations.float()
        return batch_flatten(observations.float())

    def _prepare_state(
        self,
        observations: torch.Tensor,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_feat = self._encode_observations(
            observations,
            proprioceptions=proprioceptions,
        )
        if self.proprio_fusion is not None:
            if proprioceptions is None:
                raise ValueError('State-conditioned chunk critic requires proprioceptions.')
            if obs_feat.ndim != 2:
                raise ValueError('Proprioception fusion requires vector observation features.')
            obs_feat = self.proprio_fusion(obs_feat, proprioceptions)
        if self.state_fusion is not None and obs_feat.ndim != 2:
            raise ValueError('Language state fusion requires vector observation features.')
        if self.state_fusion is not None:
            obs_feat = self.state_fusion(obs_feat, language_embed)
        return obs_feat

    def reduce_prefix(self, prefix_qs: torch.Tensor) -> torch.Tensor:
        if prefix_qs.ndim != 3:
            raise ValueError(f'reduce_prefix expects shape [E, B, H], got {tuple(prefix_qs.shape)}')
        if self.prefix_reduce == 'mean':
            return prefix_qs.mean(dim=-1)
        if self.prefix_reduce == 'last':
            return prefix_qs[..., -1]
        raise ValueError(f'Unsupported critic prefix reduction: {self.prefix_reduce}')

    def forward_prefix(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions is None:
            raise ValueError('ChunkTransformerValue requires actions to compute prefix Q-values.')
        obs_feat = self._prepare_state(
            observations,
            language_embed=language_embed,
            proprioceptions=proprioceptions,
        )
        qs = [
            net(obs_feat, actions, observation_mask=observation_mask)
            for net in self.value_nets
        ]
        return torch.stack(qs, dim=0)

    def forward(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        language_embed: torch.Tensor | None = None,
        proprioceptions: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix_qs = self.forward_prefix(
            observations,
            actions,
            language_embed=language_embed,
            proprioceptions=proprioceptions,
            observation_mask=observation_mask,
        )
        return self.reduce_prefix(prefix_qs)


@dataclass
class DiagGaussianPrior:
    dim: int
    device: torch.device
    action_len: int | None = None

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.action_len is None:
            return torch.randn(batch_size, self.dim, device=self.device)
        return torch.randn(batch_size, self.action_len, self.dim, device=self.device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            log_z = -0.5 * self.dim * np.log(2 * np.pi)
            return log_z - 0.5 * x.pow(2).sum(dim=-1)
        if x.ndim == 3:
            log_z = -0.5 * self.dim * np.log(2 * np.pi)
            token_log_prob = log_z - 0.5 * x.pow(2).sum(dim=-1)
            return token_log_prob.sum(dim=-1)
        raise ValueError(
            f'DiagGaussianPrior.log_prob expects shape [B, D] or [B, T, D], got {tuple(x.shape)}'
        )
