"""
student_mobilenet.py
MobileNetV2-based lightweight student model for DLoc knowledge distillation.

Architecture (from professor's Stage 1 spec):
  Encoder: MobileNetV2 backbone (first conv modified for 4-channel input)
  Decoder: Upsample + Conv (320->128->64->32->1) + Sigmoid
  Output: [B, 1, 161, 361] location heatmap

~2.3M params vs DLoc's ~15M params.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetStudent(nn.Module):
    # Target output spatial size
    OUTPUT_SIZE = (161, 361)

    def __init__(self, input_nc=4):
        super().__init__()

        # Load MobileNetV2 backbone (features[0:18] = 320-channel output)
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.encoder = mobilenet.features[:-1]  # drop final 320->1280 conv

        # Replace first conv: 3-channel RGB -> 4-channel AoA
        old_conv = self.encoder[0][0]
        self.encoder[0][0] = nn.Conv2d(
            input_nc, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Decoder: 320 -> 128 -> 64 -> 32 -> 1
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(320, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        # Ensure exact output size (MobileNetV2 downsamples aggressively)
        if out.shape[2:] != self.OUTPUT_SIZE:
            out = nn.functional.interpolate(out, size=self.OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return out


if __name__ == "__main__":
    model = MobileNetStudent(input_nc=4)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,} ({n/1e6:.2f}M)")
    x = torch.randn(2, 4, 161, 361)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 1, 161, 361), f"Wrong shape: {y.shape}"
    print("OK")
