#!/usr/bin/env python
"""
mamba_model.py
Mamba-DLoc: Replaces the ResNet bottleneck in DLoc with Mamba selective
state-space (S6) blocks for efficient long-range sequence modelling.

Architecture overview
---------------------
CNN Stem  (4 -> 128 channels, 2x spatial downsample)
  -> Tokenize (flatten spatial -> sequence)
  -> 4x MambaResidualBlock  (LayerNorm -> Mamba -> residual)
  -> Reshape back to spatial
  -> CNN Decoder (upsample back to original resolution)

Input:  [B, 4, 161, 361]  AoA heatmaps
Output: [B, 1, 161, 361]  location heatmap
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try to import the CUDA-optimised Mamba implementation (Linux + CUDA only).
# Fall back to a pure-PyTorch implementation on Windows / CPU.
# ---------------------------------------------------------------------------
HAS_MAMBA_SSM = False
try:
    from mamba_ssm import Mamba as CudaMamba  # type: ignore
    HAS_MAMBA_SSM = True
except ImportError:
    CudaMamba = None


# ===================================================================
# Pure-PyTorch Mamba (S6 selective scan) - works on any backend
# ===================================================================
class PurePytorchMamba(nn.Module):
    """
    Minimal re-implementation of the Mamba selective-scan block (S6).

    Interface: [B, L, D] -> [B, L, D]

    Parameters
    ----------
    d_model : int   - input / output dimension (D)
    d_state : int   - SSM hidden state dimension (N)
    d_conv  : int   - local convolution kernel width
    expand  : int   - expansion factor for inner dimension (E = expand * D)
    """

    def __init__(self, d_model: int = 128, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand           # E
        self.dt_rank = math.ceil(d_model / 16)     # rank for delta-t projection

        # --- input projection: D -> 2*E  (main path + gate) ---
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # --- causal depthwise conv on main path ---
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,  # causal: trim later
            groups=self.d_inner, bias=True,
        )

        # --- SSM parameter projections ---
        #   from the convolved main path (E) -> dt_rank + 2*N
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        #   dt_rank -> E  (broadcast delta across inner dim)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # --- learnable SSM parameters ---
        # A_log: [E, N]  initialised as log(1..N) broadcast to each inner dim
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))           # learned in log-space

        # D (skip connection scalar per inner dim)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --- output projection: E -> D ---
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, L, D]
        returns : [B, L, D]
        """
        B, L, _ = x.shape

        # --- input projection & split ---
        xz = self.in_proj(x)                              # [B, L, 2E]
        x_main, z = xz.chunk(2, dim=-1)                   # each [B, L, E]

        # --- causal conv1d on main path ---
        # conv1d expects [B, E, L]
        x_main = x_main.transpose(1, 2)                   # [B, E, L]
        x_main = self.conv1d(x_main)[:, :, :L]            # causal trim
        x_main = x_main.transpose(1, 2)                   # [B, L, E]
        x_main = F.silu(x_main)

        # --- SSM parameters ---
        x_ssm = self.x_proj(x_main)                       # [B, L, dt_rank + 2N]
        dt, B_param, C_param = x_ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # dt: [B, L, dt_rank] -> [B, L, E]
        dt = F.softplus(self.dt_proj(dt))                  # [B, L, E]

        # A: [E, N]  (negative for stability)
        A = -torch.exp(self.A_log)                         # [E, N]

        # --- discretize ---
        # dA: [B, L, E, N]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: [B, L, E, N]
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)       # broadcast E

        # --- selective scan (sequential, kept simple for correctness) ---
        # x_main: [B, L, E]  ->  weight input
        # h: [B, E, N]       SSM hidden state
        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t, :, :] * h + dB[:, t, :, :] * x_main[:, t, :].unsqueeze(-1)
            y_t = (h * C_param[:, t, :].unsqueeze(1)).sum(dim=-1)   # [B, E]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                         # [B, L, E]

        # --- skip connection ---
        y = y + x_main * self.D.unsqueeze(0).unsqueeze(0)

        # --- gate ---
        y = y * F.silu(z)

        # --- output projection ---
        return self.out_proj(y)                            # [B, L, D]


# ===================================================================
# Mamba residual block: LayerNorm -> Mamba -> residual add
# ===================================================================
class MambaResidualBlock(nn.Module):
    """
    Pre-norm residual block wrapping a Mamba layer.

    Interface: [B, L, D] -> [B, L, D]
    Uses the CUDA Mamba kernel when available, otherwise falls back to
    PurePytorchMamba.
    """

    def __init__(self, d_model: int = 128, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA_SSM and CudaMamba is not None:
            self.mamba = CudaMamba(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
        else:
            self.mamba = PurePytorchMamba(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


# ===================================================================
# MambaDLocNet - full end-to-end model
# ===================================================================
class MambaDLocNet(nn.Module):
    """
    Mamba-DLoc: CNN stem  ->  Mamba sequence blocks  ->  CNN decoder.

    Input:  [B, 4, 161, 361]
    Output: [B, 1, 161, 361]
    """

    # Spatial sizes at each resolution level (full, mid, low)
    SPATIAL_SIZES = [(161, 361), (81, 181), (41, 91)]

    def __init__(self, input_nc: int = 4, d_model: int = 128,
                 n_mamba_blocks: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model

        # ------ CNN Stem (encoder) ------
        self.stem = nn.Sequential(
            # [B,input_nc,161,361] -> [B,32,161,361]
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # [B,32,161,361] -> [B,64,81,181]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # [B,64,81,181] -> [B,128,41,91]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ------ Mamba sequence blocks ------
        self.mamba_blocks = nn.Sequential(*[
            MambaResidualBlock(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
            for _ in range(n_mamba_blocks)
        ])

        # ------ CNN Decoder ------
        # Stage 1: [B,128,41,91] -> interpolate to (81,181) -> conv 128->64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 2: [B,64,81,181] -> interpolate to (161,361) -> conv 64->32
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Stage 3: 1x1 head -> sigmoid
        self.dec_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    # ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 4, 161, 361]
        returns : [B, 1, 161, 361]
        """
        B = x.size(0)

        # --- CNN Stem ---
        feat = self.stem(x)                                # [B, 128, 41, 91]

        # --- Tokenize: flatten spatial dims into sequence ---
        _, C, H, W = feat.shape
        tokens = feat.reshape(B, C, H * W).permute(0, 2, 1)  # [B, 3731, 128]

        # --- Mamba blocks ---
        tokens = self.mamba_blocks(tokens)                 # [B, 3731, 128]

        # --- Reshape back to spatial ---
        feat = tokens.permute(0, 2, 1).reshape(B, C, H, W)   # [B, 128, 41, 91]

        # --- CNN Decoder ---
        feat = F.interpolate(feat, size=self.SPATIAL_SIZES[1],
                             mode='bilinear', align_corners=False)   # [B,128,81,181]
        feat = self.dec_conv1(feat)                                   # [B,64,81,181]

        feat = F.interpolate(feat, size=self.SPATIAL_SIZES[0],
                             mode='bilinear', align_corners=False)   # [B,64,161,361]
        feat = self.dec_conv2(feat)                                   # [B,32,161,361]

        out = self.dec_head(feat)                                     # [B,1,161,361]
        return out
