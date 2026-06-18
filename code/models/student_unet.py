"""
student_unet.py
MobileNetV2 + U-Net skip connections student model for DLoc.

Architecture (Professor Phase 2, Step 5 Option 1):
  Encoder: MobileNetV2 backbone split into 4 stages for skip connections
    - inc:  ConvBNReLU(4->32, stride=2)
    - enc1: features[1:4]   -> 24 channels  (Skip 1)
    - enc2: features[4:7]   -> 32 channels  (Skip 2)
    - enc3: features[7:14]  -> 96 channels  (Skip 3)
    - enc4: features[14:18] -> 320 channels (Bottleneck)
  Decoder: Upsample + Concat skip + Conv
    - dec1: 320+96=416 -> 128
    - dec2: 128+32=160 -> 64
    - dec3: 64+24=88   -> 32
    - final: 32 -> 1 + Sigmoid
  Output: [B, 1, 161, 361]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetUNetStudent(nn.Module):
    OUTPUT_SIZE = (161, 361)

    def __init__(self, input_nc=4):
        super().__init__()

        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Modify first conv for input_nc channels
        orig_conv = mobilenet.features[0][0]
        self.inc = nn.Sequential(
            nn.Conv2d(input_nc, orig_conv.out_channels,
                      kernel_size=orig_conv.kernel_size,
                      stride=orig_conv.stride,
                      padding=orig_conv.padding, bias=False),
            mobilenet.features[0][1],  # BatchNorm
            mobilenet.features[0][2],  # ReLU6
        )

        # Encoder stages (split to capture skip connections)
        self.enc1 = mobilenet.features[1:4]    # -> 24 ch
        self.enc2 = mobilenet.features[4:7]    # -> 32 ch
        self.enc3 = mobilenet.features[7:14]   # -> 96 ch
        self.enc4 = mobilenet.features[14:-1]  # -> 320 ch (drop final 1x1 conv)

        # Decoder with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(320 + 96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 24, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)
        s1 = self.enc1(x0)       # skip 1: 24 ch
        s2 = self.enc2(s1)       # skip 2: 32 ch
        s3 = self.enc3(s2)       # skip 3: 96 ch
        bottle = self.enc4(s3)   # bottleneck: 320 ch

        # Decoder with skip concatenation
        d1 = self.up1(bottle)
        s3r = F.interpolate(s3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, s3r], dim=1))

        d2 = self.up2(d1)
        s2r = F.interpolate(s2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2r], dim=1))

        d3 = self.up3(d2)
        s1r = F.interpolate(s1, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, s1r], dim=1))

        out = self.final_conv(d3)
        if out.shape[2:] != self.OUTPUT_SIZE:
            out = F.interpolate(out, size=self.OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return out


if __name__ == "__main__":
    model = MobileNetUNetStudent(input_nc=4)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,} ({n/1e6:.2f}M)")
    x = torch.randn(2, 4, 161, 361)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 1, 161, 361), f"Wrong shape: {y.shape}"
    print("OK")
