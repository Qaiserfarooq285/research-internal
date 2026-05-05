#!/usr/bin/env python3

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from timm.models.layers import DropPath


class SpatialMambaBlock(nn.Module):
    """Efficient Spatial Mamba Block - OPTIMIZED for FLOPs reduction.

    Achieves FLOPs reduction through:
    1. Reduced state dimension (4 vs 16)
    2. Reduced Mamba expansion ratio (1.0 vs 2)
    3. Projected hidden dim (7/8 C vs full C)
    4. Single Mamba pass only

    FFN uses standard 2x expansion to preserve representation capacity.
    Target FLOPs: below the baseline MambaVision-T reference.
    """

    def __init__(self, dim, d_state=4, d_conv=3, expand=1.0, drop_path=0.0):
        """
        Args:
            dim: Feature dimension
            d_state: Mamba state dimension (reduced from 16 to 4 for efficiency)
            d_conv: Mamba conv dimension
            expand: Expansion ratio for Mamba (reduced from 2 to 1.0)
            drop_path: Drop path rate
        """
        super().__init__()
        self.hidden_dim = max(int(dim * 7 / 8), 1)
        self.input_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.output_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # Single Mamba scanner with reduced complexity
        self.mamba = Mamba(d_model=self.hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # FFN: 2x expansion (standard — avoids contracting bottleneck)
        ffn_dim = max(int(self.hidden_dim * 2), 1)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, self.hidden_dim)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C)  [channel-last]

        Returns:
            Output tensor of shape (B, H, W, C)
        """
        shortcut = x                          # (B, H, W, C) — original dim
        h = self.input_proj(x)                # (B, H, W, hidden_dim)
        h = self.norm1(h)
        B, H, W, C = h.shape

        # Flatten spatial dims → sequence, apply Mamba, restore
        h = self.mamba(h.reshape(B, H * W, C)).reshape(B, H, W, C)

        # FFN with second norm (both in hidden_dim space)
        h = h + self.drop_path(self.ffn(self.norm2(h)))

        # Project back to original dim, then single residual
        h = self.output_proj(h)               # (B, H, W, C)
        return shortcut + self.drop_path(h)   # (B, H, W, C)


class SpatialMambaLayer(nn.Module):
    """Layer composed of multiple SpatialMambaBlock instances.

    Accepts input in MambaVision's native channel-first format (B, C, H, W),
    converts internally to channel-last for Mamba processing, and returns
    channel-first (B, C, H, W) — compatible with the BatchNorm2d head.

    NOTE: Stage 4 (i=3) has downsample=False in MambaVision, so no
    Downsample module is attached here.  The layer simply processes
    features and returns them at the same spatial resolution.
    """

    def __init__(self, dim, depth, drop_path=0.0, **kwargs):
        """
        Args:
            dim: Feature dimension (channel count in channel-first input)
            depth: Number of SpatialMambaBlock layers
            drop_path: Drop path rate (scalar or list; scalar is used as max)
            **kwargs: Absorbed for API compatibility with MambaVisionLayer
                      (num_heads, window_size, mlp_ratio, qkv_bias, …)
        """
        super().__init__()

        # Accept either a scalar max rate or a pre-built list
        if isinstance(drop_path, (list, tuple)):
            drop_path_rates = list(drop_path)
            # Pad / trim to match depth
            if len(drop_path_rates) < depth:
                drop_path_rates += [drop_path_rates[-1]] * (depth - len(drop_path_rates))
            drop_path_rates = drop_path_rates[:depth]
        else:
            drop_path_rates = torch.linspace(0, drop_path, depth).tolist()

        self.blocks = nn.ModuleList(
            SpatialMambaBlock(dim=dim, drop_path=drop_path_rates[i])
            for i in range(depth)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W)  — channel-first, as produced by MambaVision stages 1-3

        Returns:
            (B, C, H, W)  — same format, compatible with BatchNorm2d norm head
        """
        # Convert to channel-last for Mamba blocks
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)

        for blk in self.blocks:
            x = blk(x)

        # Convert back to channel-first
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)
        return x
