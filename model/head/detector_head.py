import torch
from torch import nn
import pdb

from .detector_predictor import make_predictor
from .detector_loss import make_loss_evaluator
from .detector_infer import make_post_processor
from losses.dice_loss import DiceLoss

class Detect_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Head, self).__init__()

        self.predictor = make_predictor(cfg, in_channels)
        # ---------- 前景 mask 预测分支 -------------- ### NEW BEGIN
        mid = in_channels // 2
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1)    # 直接预测下采样分辨率的单通道 mask logits
        )
        self.mask_loss = DiceLoss()
        self.dice_w    = getattr(cfg.MODEL.HEAD, "DICE_WEIGHT", 1.0)  # 可在 yaml 中配置
        # ---------- 前景 mask 预测分支 -------------- ### NEW END
        
        self.loss_evaluator = make_loss_evaluator(cfg)
        self.post_processor = make_post_processor(cfg)

    def forward(self, features, targets=None, test=False):
        x = self.predictor(features, targets)

        if self.training:
            loss_dict, log_dict = self.loss_evaluator(x, targets)

            # ------- DiceLoss 计算 ------- ### NEW BEGIN
            features_map = features[-1] if isinstance(features, (list,tuple)) else features
            mask_pred = torch.sigmoid(self.mask_decoder(features_map))  # [B,1,H,W]
            mask_gt   = torch.stack([t.get_field("mask") for t in targets]).to(mask_pred.device)
            loss_mask = self.mask_loss(mask_pred, mask_gt)
            loss_dict["loss_mask"] = self.dice_w * loss_mask
            log_dict["loss_mask"]  = loss_mask.item()
            # ------- DiceLoss 计算 ------- ### NEW END

            return loss_dict, log_dict
        else:
            result, eval_utils, visualize_preds = self.post_processor(x, targets, test=test, features=features)
            
            return result, eval_utils, visualize_preds

def bulid_head(cfg, in_channels):
    return Detect_Head(cfg, in_channels)