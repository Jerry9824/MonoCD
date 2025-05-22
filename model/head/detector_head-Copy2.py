import torch
from torch import nn
import pdb
import torch.nn.functional as F   # 新增用于 L1 损失计算
from .detector_predictor import make_predictor
from .detector_loss import make_loss_evaluator
from .detector_infer import make_post_processor
from losses.dice_loss import DiceLoss

class Detect_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Head, self).__init__()

        self.predictor = make_predictor(cfg, in_channels)
        self.cfg = cfg
        # ---------- 前景 mask 预测分支 -------------- ### NEW BEGIN
        mid = in_channels // 2          # 例：C=256 → mid=128

        # ✦ 残差双卷积块（Res-DoubleConv） ---------------------------------
        class ResBlock(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out)
                )
                # 若升/降通道，用 1×1 投影；否则恒等
                self.proj = (nn.Identity() if c_in == c_out else
                             nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False),
                                           nn.BatchNorm2d(c_out)))

                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.conv(x) + self.proj(x))
        # ------------------------------------------------------------------

        self.mask_decoder = nn.Sequential(
            # stage-0 : 1/4 → 1/4（残差提特征）
            ResBlock(in_channels, mid),

            # stage-1 : 1/4 → 1/2
            nn.ConvTranspose2d(mid, mid // 2, 2, stride=2),          # 上采样×2
            ResBlock(mid // 2, mid // 2),

            # stage-2 : 1/2 → 1
            nn.ConvTranspose2d(mid // 2, mid // 4, 2, stride=2),     # 再上采样×2
            ResBlock(mid // 4, mid // 4),

            # 输出 1-channel logits
            nn.Conv2d(mid // 4, 1, 1)
        )

        self.mask_loss = DiceLoss()
        self.dice_w    = getattr(cfg.MODEL.HEAD, "DICE_WEIGHT", 1.0)  # 可在 yaml 中配置
        # ---------- 前景 mask 预测分支 -------------- ### NEW END
        
        self.loss_evaluator = make_loss_evaluator(cfg)
        self.post_processor = make_post_processor(cfg)

    def forward(self, features, targets=None, test=False):
        # 判断是否处于“mask 预训练”阶段
        # 当 LOSS_NAMES 只包含 "loss_mask" 时，就只跑 mask 分支
        is_mask_pretrain = (
            self.training
            and getattr(self.cfg.MODEL.HEAD, "LOSS_NAMES", []) == ["loss_mask"]
        )
        if is_mask_pretrain:
            # —— 1) 只计算 mask 相关损失，不执行检测分支 —— 
            features_map = features[-1] if isinstance(features, (list, tuple)) else features
            mask_logit     = self.mask_decoder(features_map)                                 # [B,1,4H,4W]
            mask_pred_high = torch.sigmoid(mask_logit)                                       # [B,1,4H,4W]
            mask_gt        = torch.stack([t.get_field("mask") for t in targets]).to(mask_pred_high.device)  # [B,1,H,W]
            mask_pred      = F.interpolate(
                mask_pred_high,
                size=(mask_gt.size(2), mask_gt.size(3)),
                mode="bilinear",
                align_corners=False,
            )                                                                               # [B,1,H,W]
            loss_mask      = self.mask_loss(mask_pred, mask_gt)
            # 背景区域 L1 Loss
            mask_inv     = 1 - mask_gt
            loss_mask_bg = F.l1_loss(mask_pred * mask_inv, torch.zeros_like(mask_pred))
            mask_bg_w    = getattr(self.cfg.MODEL.HEAD, "MASK_BG_WEIGHT", 1.0)
            # 返回两个损失
            return {
                "loss_mask":     self.dice_w * loss_mask,
                "loss_mask_bg":  mask_bg_w * loss_mask_bg,
            }, {
                "loss_mask":     loss_mask.item(),
                "loss_mask_bg":  loss_mask_bg.item(),
            }

        # —— 否则正常执行“检测 + mask”联合训练逻辑 —— 
        if self.training:
            # 1) 先算 mask 分支损失
            features_map = features[-1] if isinstance(features, (list, tuple)) else features
            # 1) 用原始 decoder 得到高分辨率 mask logits
            mask_logit     = self.mask_decoder(features_map)                                 # [B,1,4H,4W]
            mask_pred_high = torch.sigmoid(mask_logit)                                       # [B,1,4H,4W]

            # 2) down-sample 回和 mask_gt 一样的 H×W 大小，才能做 loss
            mask_gt = torch.stack([t.get_field("mask") for t in targets]).to(mask_pred_high.device)  # [B,1,H,W]
            mask_pred = F.interpolate(
                mask_pred_high,
                size=(mask_gt.size(2), mask_gt.size(3)),
                mode="bilinear",
                align_corners=False,
            )                                                                               # [B,1,H,W]

            # 3) 计算 DiceLoss + 背景 L1
            loss_mask    = self.mask_loss(mask_pred, mask_gt)
            mask_inv     = 1 - mask_gt
            loss_mask_bg = F.l1_loss(mask_pred * mask_inv, torch.zeros_like(mask_pred))
            mask_bg_w    = getattr(self.cfg.MODEL.HEAD, "MASK_BG_WEIGHT", 1.0)

            # 后面照旧：用 mask_pred 融合特征，调用 predictor，再把 loss_mask、loss_mask_bg 放入 loss_dict

            # 2) 用 mask 融合特征再做检测
            fused_map = features_map * (mask_pred + 1.0)
            if isinstance(features, (list, tuple)):
                features = list(features)
                features[-1] = fused_map
            else:
                features = fused_map
            x = self.predictor(features, targets)
            loss_dict, log_dict = self.loss_evaluator(x, targets)

            # 3) 合并 mask 损失到 loss_dict
            loss_dict["loss_mask"]    = self.dice_w * loss_mask
            loss_dict["loss_mask_bg"] = mask_bg_w * loss_mask_bg
            log_dict["loss_mask"]     = loss_mask.item()
            log_dict["loss_mask_bg"]  = loss_mask_bg.item()
            return loss_dict, log_dict

        else:
            # —— 推理阶段同之前：用 mask 融合特征，然后再预测 —— 
            features_map = features[-1] if isinstance(features, (list, tuple)) else features
            mask_logit     = self.mask_decoder(features_map)                                 # [B,1,4H,4W]
            mask_pred_high = torch.sigmoid(mask_logit)                                       # [B,1,4H,4W]
            mask_pred      = F.interpolate(
                mask_pred_high,
                size=(features_map.size(2), features_map.size(3)),
                mode="bilinear",
                align_corners=False,
            )                                                                               # [B,1,H,W]
            fused_map    = features_map * (mask_pred + 1.0)
            if isinstance(features, (list, tuple)):
                features = list(features)
                features[-1] = fused_map
            else:
                features = fused_map
            x = self.predictor(features, targets)
            return self.post_processor(x, targets, test=test, features=features)

            
def bulid_head(cfg, in_channels):
    return Detect_Head(cfg, in_channels)