import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from typing import Optional, Sequence, Tuple

from utils.networks_torch import MLP as TorchMLP


class TorchResnetStack(torch_nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_features: int,
        num_blocks: int,
        max_pooling: bool = True,
    ):
        super().__init__()
        self.max_pooling = max_pooling

        self.first_conv = torch_nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        torch_nn.init.xavier_uniform_(self.first_conv.weight)
        torch_nn.init.zeros_(self.first_conv.bias)

        self.res_blocks = torch_nn.ModuleList()
        for _ in range(num_blocks):
            block = torch_nn.ModuleList(
                [
                    torch_nn.Conv2d(
                        num_features, num_features, kernel_size=3, stride=1, padding=1
                    ),
                    torch_nn.Conv2d(
                        num_features, num_features, kernel_size=3, stride=1, padding=1
                    ),
                ]
            )
            torch_nn.init.xavier_uniform_(block[0].weight)
            torch_nn.init.zeros_(block[0].bias)
            torch_nn.init.xavier_uniform_(block[1].weight)
            torch_nn.init.zeros_(block[1].bias)
            self.res_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)

        if self.max_pooling:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        for conv1, conv2 in self.res_blocks:
            block_input = out
            out = F.relu(out)
            out = conv1(out)
            out = F.relu(out)
            out = conv2(out)
            out = out + block_input

        return out


class TorchImpalaEncoder(torch_nn.Module):

    def __init__(
        self,
        in_channels: int = 6,
        width: int = 1,
        stack_sizes: Tuple[int, ...] = (16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: Optional[float] = None,
        mlp_hidden_dims: Sequence[int] = (512,),
        layer_norm: bool = False,
        adaptive_pool_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.adaptive_pool_hw = adaptive_pool_hw

        self.stack_blocks = torch_nn.ModuleList()
        current_channels = in_channels
        for stack_ch in stack_sizes:
            out_ch = stack_ch * width
            self.stack_blocks.append(
                TorchResnetStack(
                    in_channels=current_channels,
                    num_features=out_ch,
                    num_blocks=num_blocks,
                )
            )
            current_channels = out_ch

        self.dropout = (
            torch_nn.Dropout(p=dropout_rate) if dropout_rate is not None else None
        )

        self.ln = None
        self._layer_norm_enabled = layer_norm

        self.mlp_hidden_dims = list(mlp_hidden_dims)
        self.mlp = None

    def _build_head(self, channels: int, flat_dim: int, device: torch.device):
        if self.mlp is not None:
            return

        if self._layer_norm_enabled:
            self.ln = torch_nn.LayerNorm(channels, eps=1e-6).to(device)

        self.mlp = TorchMLP(
            hidden_dims=self.mlp_hidden_dims,
            input_dim=flat_dim,
            activate_final=True,
            layer_norm=self._layer_norm_enabled,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(
                f"IMPALA encoder expects [B, V, C, T, H, W], got {tuple(x.shape)}."
            )
        batch_size, num_views, channels, horizon, _, _ = x.shape
        x = x.reshape(batch_size * num_views * horizon, channels, *x.shape[-2:])

        x = x.float() / 255.0

        out = x
        for block in self.stack_blocks:
            out = block(out)
            if self.dropout is not None:
                out = self.dropout(out)

        out = F.relu(out)
        if self.adaptive_pool_hw is not None:
            out = F.adaptive_avg_pool2d(out, tuple(self.adaptive_pool_hw))

        _, output_channels, output_height, output_width = out.shape
        flat_dim = (
            output_channels
            * output_height
            * output_width
            * num_views
            * horizon
        )
        self._build_head(output_channels, flat_dim, out.device)

        out = out.permute(0, 2, 3, 1).contiguous()

        if self.ln is not None:
            out = self.ln(out)

        out = out.reshape(batch_size, -1)
        out = self.mlp(out)

        return out


class TorchRoboMimicSpatialResNet18(torch_nn.Module):
    """Two-view RoboMimic encoder that preserves ResNet spatial features."""

    def __init__(
        self,
        num_views=2,
        proprio_dim=9,
        use_proprioception=True,
        use_augmentation=True,
        crop_size=76,
        output_size=224,
        max_obs_horizon=8,
        pretrained_weights="IMAGENET1K_V1",
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.proprio_dim = int(proprio_dim)
        self.use_proprioception = bool(use_proprioception)
        self.use_augmentation = bool(use_augmentation)
        self.crop_size = int(crop_size)
        self.output_size = int(output_size)
        self.max_obs_horizon = int(max_obs_horizon)
        self.token_dim = 512
        if self.num_views != 2:
            raise ValueError(
                f"RoboMimic spatial ResNet expects two camera views, got {self.num_views}."
            )
        if self.crop_size <= 0 or self.output_size <= 0:
            raise ValueError("crop_size and output_size must be positive.")

        weights = pretrained_weights
        if isinstance(weights, str) and weights.strip().lower() in ("", "none", "null"):
            weights = None
        self.camera_backbones = torch_nn.ModuleList()
        for _ in range(self.num_views):
            model = torchvision.models.resnet18(weights=weights)
            self.camera_backbones.append(
                torch_nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                )
            )

        self.spatial_embedding = torch_nn.Parameter(
            torch.zeros(1, 1, 1, 49, self.token_dim)
        )
        self.camera_embedding = torch_nn.Parameter(
            torch.zeros(1, 1, self.num_views, 1, self.token_dim)
        )
        self.temporal_embedding = torch_nn.Parameter(
            torch.zeros(1, self.max_obs_horizon, 1, 1, self.token_dim)
        )
        self.token_type_embedding = torch_nn.Parameter(torch.zeros(2, self.token_dim))
        self.proprio_projection = (
            torch_nn.Sequential(
                torch_nn.Linear(self.proprio_dim, self.token_dim),
                torch_nn.LayerNorm(self.token_dim),
            )
            if self.use_proprioception
            else None
        )
        torch_nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        torch_nn.init.trunc_normal_(self.camera_embedding, std=0.02)
        torch_nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        torch_nn.init.trunc_normal_(self.token_type_embedding, std=0.02)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _crop(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, time, height, width = images.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"RoboMimic images must be at least {self.crop_size}x{self.crop_size}, "
                f"got {height}x{width}."
            )
        max_top = height - self.crop_size
        max_left = width - self.crop_size
        if self.training and self.use_augmentation:
            tops = torch.randint(
                max_top + 1,
                (batch_size, num_views),
                device=images.device,
            )
            lefts = torch.randint(
                max_left + 1,
                (batch_size, num_views),
                device=images.device,
            )
        else:
            tops = torch.full(
                (batch_size, num_views),
                max_top // 2,
                device=images.device,
                dtype=torch.long,
            )
            lefts = torch.full(
                (batch_size, num_views),
                max_left // 2,
                device=images.device,
                dtype=torch.long,
            )

        flat_images = images.reshape(
            batch_size * num_views,
            channels,
            time,
            height,
            width,
        )
        patches = flat_images.unfold(3, self.crop_size, 1).unfold(
            4,
            self.crop_size,
            1,
        )
        flat_indices = torch.arange(
            batch_size * num_views,
            device=images.device,
        )
        flat_tops = tops.reshape(-1)
        flat_lefts = lefts.reshape(-1)
        # Tensor indexing selects one camera-specific crop while retaining all
        # frames, so temporal windows share geometry without CUDA synchronizations.
        cropped = patches[
            flat_indices,
            :,
            :,
            flat_tops,
            flat_lefts,
            :,
            :,
        ]
        return cropped.reshape(
            batch_size,
            num_views,
            channels,
            time,
            self.crop_size,
            self.crop_size,
        )

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = self._crop(images).float().div(255.0)
        batch_size, num_views, channels, time = images.shape[:4]
        images = rearrange(images, "b v c t h w -> (b v t) c h w")
        images = torch.nn.functional.interpolate(
            images,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.imagenet_mean) / self.imagenet_std
        return rearrange(
            images,
            "(b v t) c h w -> b v t c h w",
            b=batch_size,
            v=num_views,
            t=time,
        )

    def forward_tokens(
        self,
        imgs: torch.Tensor,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if imgs.ndim != 6:
            raise ValueError(
                "RoboMimic spatial ResNet expects images [B, V, C, T, H, W], "
                f"got {tuple(imgs.shape)}."
            )
        batch_size, num_views, channels, time = imgs.shape[:4]
        if num_views != self.num_views or channels != 3:
            raise ValueError(
                f"Expected [B, {self.num_views}, 3, T, H, W], got {tuple(imgs.shape)}."
            )
        if time > self.max_obs_horizon:
            raise ValueError(
                f"Observation horizon {time} exceeds max_obs_horizon={self.max_obs_horizon}."
            )

        images = self._preprocess(imgs)
        per_view_features = []
        for view_idx, backbone in enumerate(self.camera_backbones):
            view_images = rearrange(images[:, view_idx], "b t c h w -> (b t) c h w")
            features = backbone(view_images)
            if tuple(features.shape[-2:]) != (7, 7):
                raise RuntimeError(
                    f"ResNet-18 spatial output must be 7x7, got {tuple(features.shape[-2:])}."
                )
            features = rearrange(
                features,
                "(b t) c h w -> b t (h w) c",
                b=batch_size,
                t=time,
            )
            per_view_features.append(features)
        visual_tokens = torch.stack(per_view_features, dim=2)
        visual_tokens = (
            visual_tokens
            + self.spatial_embedding
            + self.camera_embedding
            + self.temporal_embedding[:, :time]
            + self.token_type_embedding[0].view(1, 1, 1, 1, -1)
        )

        if self.use_proprioception:
            if proprioceptions is None:
                raise ValueError(
                    "Proprioception is required by this RoboMimic encoder."
                )
            if tuple(proprioceptions.shape) != (batch_size, time, self.proprio_dim):
                raise ValueError(
                    "Expected proprioceptions with shape "
                    f"[{batch_size}, {time}, {self.proprio_dim}], got "
                    f"{tuple(proprioceptions.shape)}."
                )
            state_tokens = self.proprio_projection(proprioceptions.float())
            state_tokens = (
                state_tokens
                + self.temporal_embedding[:, :time, 0, 0]
                + self.token_type_embedding[1].view(1, 1, -1)
            ).unsqueeze(2)
            visual_tokens = rearrange(
                visual_tokens,
                "b t v p c -> b t (v p) c",
            )
            tokens = torch.cat([visual_tokens, state_tokens], dim=2)
            return rearrange(tokens, "b t p c -> b (t p) c")

        if proprioceptions is not None:
            raise ValueError(
                "Proprioceptions were provided while use_proprioception=False."
            )
        return rearrange(visual_tokens, "b t v p c -> b (t v p) c")

    def forward(
        self,
        imgs: torch.Tensor,
        proprioceptions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_tokens(imgs, proprioceptions=proprioceptions)

