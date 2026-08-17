from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def unsqueeze(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.unsqueeze(t, dim=dim)


class TimestepEmbedder(nn.Module):
    """Embed scalar guidance values into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=self.mlp[0].weight.device)
        if t.ndim == 0:
            t = t[None]
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class InContextLayer(nn.Module):
    """A small DiT-style Transformer layer."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.attn(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.mlp(self.norm2(x))
        return x


class InContextBlock(nn.Module):
    """One non-autoregressive reverse block with observation and guidance tokens."""

    def __init__(
        self,
        action_len: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        cond_channels: int,
        num_condition_tokens: int = 1,
        num_guidance_tokens: int = 1,
        context_dim: int | None = None,
    ):
        super().__init__()
        if context_dim is None:
            raise ValueError('context_dim is required by InContextBlock.')
        self.action_len = action_len
        self.hidden_size = hidden_size
        self.num_condition_tokens = num_condition_tokens
        self.num_guidance_tokens = num_guidance_tokens

        self.blocks = nn.Sequential(
            *[InContextLayer(hidden_size, num_heads=num_heads) for _ in range(depth)]
        )
        self.cond_embedder = nn.Linear(cond_channels, hidden_size)
        self.null_cond = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.context_projection = nn.Linear(int(context_dim), hidden_size)
        self.null_context = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.condition_tokens = nn.Parameter(
            torch.randn(num_condition_tokens, hidden_size) * 0.02
        )
        self.guidance_tokens = nn.Parameter(torch.randn(num_guidance_tokens, hidden_size) * 0.02)
        self.guidance_embedder = TimestepEmbedder(hidden_size=hidden_size)

    def _cond_embedding(
        self,
        y: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            cond = self.null_cond.to(device=device, dtype=dtype).repeat(batch_size, 1)
        else:
            cond = self.cond_embedder(y.to(device=device, dtype=dtype))
            if cond_drop_mask is not None:
                null_cond = self.null_cond.to(device=device, dtype=dtype).repeat(batch_size, 1)
                cond = torch.where(cond_drop_mask[:, None], null_cond, cond)

        return cond

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        guidance: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        cond = self._cond_embedding(
            y,
            batch_size,
            x.dtype,
            x.device,
            cond_drop_mask=cond_drop_mask,
        )
        c_tokens = self.condition_tokens.to(dtype=x.dtype).unsqueeze(0) + unsqueeze(cond, 1)
        guidance_emb = self.guidance_embedder(guidance).to(dtype=x.dtype)
        g_tokens = self.guidance_tokens.to(dtype=x.dtype).unsqueeze(0) + unsqueeze(guidance_emb, 1)

        token_parts = [x]
        valid_parts = [torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)]
        if context_tokens is None:
            projected_context = self.null_context.to(device=x.device, dtype=x.dtype).expand(
                batch_size, -1, -1
            )
            context_attention_mask = torch.ones(
                batch_size, 1, device=x.device, dtype=torch.bool
            )
        else:
            projected_context = self.context_projection(
                context_tokens.to(device=x.device, dtype=x.dtype)
            )
            if context_attention_mask is None:
                context_attention_mask = torch.ones(
                    projected_context.shape[:2], device=x.device, dtype=torch.bool
                )
            else:
                context_attention_mask = context_attention_mask.to(
                    device=x.device, dtype=torch.bool
                )
            if cond_drop_mask is not None:
                null_context = self.null_context.to(
                    device=x.device, dtype=x.dtype
                ).expand_as(projected_context)
                projected_context = torch.where(
                    cond_drop_mask[:, None, None],
                    null_context,
                    projected_context,
                )
        token_parts.append(projected_context)
        valid_parts.append(context_attention_mask)
        token_parts.extend([c_tokens, g_tokens])
        valid_parts.extend([
            torch.ones(c_tokens.shape[:2], device=x.device, dtype=torch.bool),
            torch.ones(g_tokens.shape[:2], device=x.device, dtype=torch.bool),
        ])
        x = torch.cat(token_parts, dim=1)
        key_padding_mask = ~torch.cat(valid_parts, dim=1)
        for layer in self.blocks:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x[:, : self.action_len]


class InContextReverseModel(nn.Module):
    """BiFlow-style one-step inverse model for action chunks."""

    def __init__(
        self,
        action_len: int,
        in_channels: int,
        cond_channels: int,
        channels: int = 512,
        num_layers: int = 2,
        num_blocks: int = 7,
        num_heads: int = 8,
        label_drop_rate: float = 0.1,
        num_condition_tokens: int = 1,
        num_guidance_tokens: int = 1,
        context_dim: int | None = None,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        if context_dim is None:
            raise ValueError('context_dim is required by InContextReverseModel.')
        self.action_len = action_len
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.channels = channels
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.label_drop_rate = label_drop_rate

        self.z_embedder = nn.Linear(in_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(1, action_len, channels) * 0.02)
        self.blocks = nn.ModuleList(
            [
                InContextBlock(
                    action_len=action_len,
                    hidden_size=channels,
                    depth=num_layers,
                    num_heads=num_heads,
                    cond_channels=cond_channels,
                    num_condition_tokens=num_condition_tokens,
                    num_guidance_tokens=num_guidance_tokens,
                    context_dim=context_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.heads = nn.ModuleList([nn.Linear(channels, in_channels) for _ in range(num_blocks)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _format_guidance(
        self,
        guidance: torch.Tensor | float,
        batch_size: int,
        device,
        dtype,
        guidance_per_block: bool = False,
    ) -> torch.Tensor:
        """Return guidance as [num_blocks, batch_size].

        Original BiFlow sampling accepts a guidance vector with one value per
        reverse block. Training usually passes one value per sample, which is
        shared by all reverse blocks.
        """
        if not torch.is_tensor(guidance):
            if isinstance(guidance, (list, tuple)):
                guidance = torch.tensor(guidance, device=device, dtype=dtype)
            else:
                return torch.full(
                    (self.num_blocks, batch_size),
                    float(guidance),
                    device=device,
                    dtype=dtype,
                )
        else:
            guidance = guidance.to(device=device, dtype=dtype)
        if guidance.ndim == 0:
            guidance = guidance.reshape(1)
        if guidance.ndim == 1:
            if guidance_per_block:
                if guidance.shape[0] == 1:
                    guidance = guidance.repeat(self.num_blocks)
                if guidance.shape[0] != self.num_blocks:
                    raise ValueError(
                        f'per-block guidance must have shape [{self.num_blocks}], '
                        f'got {tuple(guidance.shape)}.'
                    )
                return guidance[:, None].expand(self.num_blocks, batch_size)
            if guidance.shape[0] == batch_size:
                return guidance[None, :].expand(self.num_blocks, batch_size)
            if guidance.shape[0] == 1:
                return guidance.reshape(1, 1).expand(self.num_blocks, batch_size)
            if guidance.shape[0] == self.num_blocks and guidance.shape[0] != batch_size:
                return guidance[:, None].expand(self.num_blocks, batch_size)
            raise ValueError(
                f'guidance must be scalar, shape [{batch_size}], shape [{self.num_blocks}], '
                f'or shape [{self.num_blocks}, {batch_size}], got {tuple(guidance.shape)}.'
            )
        if guidance.ndim == 2:
            if guidance.shape == (self.num_blocks, batch_size):
                return guidance
            if guidance.shape == (batch_size, self.num_blocks):
                return guidance.transpose(0, 1)
            if guidance.shape == (self.num_blocks, 1):
                return guidance.expand(self.num_blocks, batch_size)
            if guidance.shape == (1, batch_size):
                return guidance.expand(self.num_blocks, batch_size)
            if guidance.shape == (1, 1):
                return guidance.expand(self.num_blocks, batch_size)
            raise ValueError(
                f'2D guidance must have shape [{self.num_blocks}, {batch_size}] '
                f'or [{batch_size}, {self.num_blocks}], got {tuple(guidance.shape)}.'
            )
        raise ValueError(f'guidance must be scalar, 1D, or 2D, got {tuple(guidance.shape)}.')

    @staticmethod
    def _validate_nonnegative_guidance(name: str, guidance: torch.Tensor) -> None:
        if torch.any(guidance < 0):
            min_value = float(guidance.min().detach().cpu())
            raise ValueError(f'{name} must be non-negative for normalized BiFlow mixing, got min={min_value}.')

    def _maybe_drop_conditions(
        self,
        guidance: torch.Tensor,
        batch_size: int,
        device: torch.device,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not enabled or not self.training or self.label_drop_rate <= 0:
            return guidance, None
        drop_mask = torch.rand(batch_size, device=device) < self.label_drop_rate
        guidance = torch.where(drop_mask[None, :], torch.zeros_like(guidance), guidance)
        return guidance, drop_mask

    def _forward_sequence(
        self,
        z: torch.Tensor,
        y: torch.Tensor | None,
        guidance: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
        blend_uncond: bool = False,
        context_tokens: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        hi = self.z_embedder(z) + self.pos_embed.to(dtype=z.dtype)
        zs = []
        for block_idx, (block, head) in enumerate(zip(self.blocks, self.heads)):
            guidance_i = guidance[block_idx]
            guidance_null = torch.zeros_like(guidance_i)
            hi_cond = block(
                hi,
                y,
                guidance=guidance_i,
                cond_drop_mask=cond_drop_mask,
                context_tokens=context_tokens,
                context_attention_mask=context_attention_mask,
            )
            if blend_uncond:
                numerator = hi_cond
                denominator = torch.ones_like(guidance_i[:, None, None])
            if blend_uncond:
                hi_uncond = block(
                    hi,
                    None,
                    guidance=guidance_null,
                    context_tokens=None,
                    context_attention_mask=None,
                ).detach()
                numerator = numerator + guidance_i[:, None, None] * hi_uncond
                denominator = denominator + guidance_i[:, None, None]
            if blend_uncond:
                hi = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
            else:
                hi = hi_cond
            zs.append(head(hi))
        return zs

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: torch.Tensor | float = 0.0,
        return_sequence: bool = False,
        guidance_per_block: bool = False,
        context_tokens: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if z.ndim == 2:
            z = z.unsqueeze(1).repeat(1, self.action_len, 1)
        elif z.ndim == 3 and z.shape[1] != self.action_len:
            if z.shape[1] > self.action_len:
                z = z[:, : self.action_len]
            else:
                z = torch.cat([z, z[:, -1:].repeat(1, self.action_len - z.shape[1], 1)], dim=1)
        elif z.ndim != 3:
            raise ValueError(f'z must have shape [B, D] or [B, T, D], got {tuple(z.shape)}.')

        batch_size = z.shape[0]
        guidance = self._format_guidance(
            guidance,
            batch_size,
            z.device,
            z.dtype,
            guidance_per_block=guidance_per_block,
        )
        self._validate_nonnegative_guidance('guidance', guidance)
        guidance, cond_drop_mask = self._maybe_drop_conditions(
            guidance,
            batch_size,
            z.device,
            enabled=return_sequence,
        )
        if context_tokens is None:
            raise ValueError('BiFlow context conditioning requires context_tokens.')
        if context_tokens.ndim != 3 or context_tokens.shape[0] != batch_size:
            raise ValueError(
                f'Invalid BiFlow context shape: {tuple(context_tokens.shape)}.'
            )
        if context_attention_mask is not None and tuple(context_attention_mask.shape) != tuple(context_tokens.shape[:2]):
            raise ValueError(
                'BiFlow context_attention_mask must match context token batch/sequence dimensions.'
            )

        cond_seq = self._forward_sequence(
            z,
            y,
            guidance,
            cond_drop_mask=cond_drop_mask,
            blend_uncond=True,
            context_tokens=context_tokens,
            context_attention_mask=context_attention_mask,
        )

        if return_sequence:
            return cond_seq
        return cond_seq[-1]
