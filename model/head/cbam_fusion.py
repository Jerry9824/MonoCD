# cbam_fusion.py
import torch
from torch import nn
import torch.nn.functional as F

class CBAMFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力——根据 mask 生成通道权重
        self.ca_fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.ca_fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
        # 空间注意力——根据 feat 做通道池化后加 mask
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, feat, mask):
        B, C, H, W = feat.shape
        
        # —— 通道注意力 —— #
        # 先融合 mask 权重：池化得到每个通道的 mask 特征
        mask_pool = F.adaptive_avg_pool2d(feat * mask, 1).view(B, C)
        w = F.relu(self.ca_fc1(mask_pool))
        w = torch.sigmoid(self.ca_fc2(w)).view(B, C, 1, 1)
        feat = feat * w  # 加权通道

        # —— 空间注意力 —— #
        avg = torch.mean(feat, dim=1, keepdim=True)
        mx  = torch.max (feat, dim=1, keepdim=True)[0]
        sa  = torch.cat([avg, mx], dim=1)              # [B,2,H,W]
        sa  = torch.sigmoid(self.sa_conv(sa))          # [B,1,H,W]
        feat = feat * sa * (mask + 1.0)                # 同时加上分割 mask

        return feat
