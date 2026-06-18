"""
student_tinycnn.py
Tiny CNN student model for DLoc knowledge distillation.

Architecture (from professor's spec):
  Block 1: Conv(4->16) + DepthwiseSep(16->16) + MaxPool
  Block 2: DepthwiseSep(16->32) + DepthwiseSep(32->32) + MaxPool
  Block 3: DepthwiseSep(32->64)
  Decoder 1: ConvTranspose(64->32, stride=2)
  Decoder 2: ConvTranspose(32->16, stride=2)
  Output: Conv(16->1, 1x1) + Sigmoid

Adapted for 161x361 input (professor's spec used 64x64).
~30K params — extremely lightweight.
"""
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class TinyCNNStudent(nn.Module):
    OUTPUT_SIZE = (161, 361)

    def __init__(self, input_nc=4):
        super().__init__()

        # Block 1: 4 -> 16, downsample 2x
        self.block1 = nn.Sequential(
            nn.Conv2d(input_nc, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 16),
            nn.MaxPool2d(2),
        )

        # Block 2: 16 -> 32, downsample 2x
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            DepthwiseSeparableConv(32, 32),
            nn.MaxPool2d(2),
        )

        # Block 3: 32 -> 64 (no downsample)
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
        )

        # Decoder 1: 64 -> 32, upsample 2x
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Decoder 2: 32 -> 16, upsample 2x
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)   # [B,16, H/2, W/2]
        x = self.block2(x)   # [B,32, H/4, W/4]
        x = self.block3(x)   # [B,64, H/4, W/4]
        x = self.dec1(x)     # [B,32, H/2, W/2]
        x = self.dec2(x)     # [B,16, H,   W]
        x = self.head(x)     # [B,1,  H,   W]
        # Ensure exact output size
        if x.shape[2:] != self.OUTPUT_SIZE:
            x = nn.functional.interpolate(x, size=self.OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    model = TinyCNNStudent(input_nc=4)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,} ({n/1e6:.2f}M)")
    x = torch.randn(2, 4, 161, 361)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 1, 161, 361), f"Wrong shape: {y.shape}"
    print("OK")
