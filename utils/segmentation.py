# utils/segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import os

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
    
def load_segmentation_model(weight_path="models/clouddeeplabv3.pth"):
    drive_id = "14uQx6dGlV8iCJdQqhWZ6KczfQa7XuaEA"
    if not os.path.exists(weight_path):
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, weight_path, quiet=False)
    model = CloudDeepLabV3Plus()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def prepare_input_tensor(image_np):
    img_resized = cv2.resize((image_np * 255).astype(np.uint8), (512, 512)) / 255.0
    input_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float()
    return input_tensor

def predict_segmentation(model, input_tensor):
    with torch.no_grad():
        pred = model(input_tensor)["out"].squeeze().numpy()
    return (pred > 0.5).astype(np.uint8)

def detect_circle_roi(image_np):
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    return mask

def canvas_to_mask(canvas_result, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if canvas_result and canvas_result.json_data:
        canvas_h, canvas_w = canvas_result.image_data.shape[:2]
        scale_x = width / canvas_w
        scale_y = height / canvas_h
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                l = int(obj["left"] * scale_x)
                t = int(obj["top"] * scale_y)
                w = int(obj["width"] * scale_x)
                h = int(obj["height"] * scale_y)
                mask[t:t+h, l:l+w] = 1
            elif obj["type"] == "path" and obj.get("path"):
                coords = [[int(item[1] * scale_x), int(item[2] * scale_y)]
                          for item in obj["path"] if isinstance(item, list) and len(item) >= 3]
                if len(coords) >= 3:
                    poly = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [poly], 1)
            elif obj["type"] == "line":
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                x1 = int(round((left + float(obj["x1"])) * scale_x))
                y1 = int(round((top + float(obj["y1"])) * scale_y))
                x2 = int(round((left + float(obj["x2"])) * scale_x))
                y2 = int(round((top + float(obj["y2"])) * scale_y))
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
                radius = int(round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2))
                if 0 <= cx < width and 0 <= cy < height and radius > 0:
                    cv2.circle(mask, (cx, cy), radius, 1, -1)
    return mask