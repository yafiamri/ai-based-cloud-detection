import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# GroupNorm helper
def gn(num_channels):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels)

# HRFS-ASPP module
class HRFSASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRFSASPP, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.branch4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            gn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        out5 = self.global_pool(x)
        out5 = F.interpolate(out5, size=(h, w), mode='bilinear', align_corners=False)

        residual = self.residual(x)
        concat = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.fusion(concat) + residual
        return out

# A-FAM module
class AFAM(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super(AFAM, self).__init__()

        self.conv_low = nn.Conv2d(low_channels, out_channels, kernel_size=1)
        self.conv_high = nn.Conv2d(high_channels, out_channels, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            gn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=True)

        low = self.conv_low(low_feat)
        high = self.conv_high(high_feat)

        concat = torch.cat([low, high], dim=1)
        attn = self.attention(concat)

        fused = attn * low + (1 - attn) * high
        out = self.fusion(fused)
        return out

# CloudDeepLabV3+ (Full Version)
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self):
        super(CloudDeepLabV3Plus, self).__init__()

        self.backbone = timm.create_model("tf_efficientnetv2_s", features_only=True, pretrained=True)

        self.hrfs_aspp = HRFSASPP(in_channels=256, out_channels=256)
        self.afam = AFAM(low_channels=24, high_channels=256, out_channels=256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            gn(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.final_upsample = lambda x: F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.backbone(x)
        low_feat = features[0]     # 24 channels
        high_feat = features[4]    # 256 channels

        x = self.hrfs_aspp(high_feat)   # [B, 256, H/4, W/4]
        x = self.afam(low_feat, x)      # [B, 256, H, W]
        x = self.decoder(x)             # Output [B, 1, H, W]
        x = self.final_upsample(x)
        return {"out": x}