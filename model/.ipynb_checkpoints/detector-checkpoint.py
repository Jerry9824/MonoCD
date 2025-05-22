import torch
from torch import nn

from structures.image_list import to_image_list

from .backbone import build_backbone
from .head.detector_head import bulid_head

from model.layers.uncert_wrapper import make_multitask_wrapper

class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        # 构建 backbone（主干网络）和 neck（如果有的话）
        self.backbone = build_backbone(cfg)
        # 保存 cfg，后面 forward 里用于判断是否 mask-only 预热阶段
        self.cfg = cfg

        # 构建 heads，即我们之前改过的 Detect_Head
        self.heads = bulid_head(cfg, self.backbone.out_channels)

        # 判断是否 test split，用于推理时传给 heads
        self.test = cfg.DATASETS.TEST_SPLIT == 'test'

    def forward(self, images, targets=None):
        # 训练时必须传入 targets
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 将 list[tensor] 转为 ImageList，并拿出其中的 tensor
        images = to_image_list(images)

        # 判断是否处于“mask-only 预热”阶段
        # 以 cfg.MODEL.HEAD.LOSS_NAMES 只包含 ["loss_mask"] 为标志
        is_mask_pretrain = (
            self.training
            and getattr(self.cfg.MODEL.HEAD, "LOSS_NAMES", []) == ["loss_mask"]
        )

        if is_mask_pretrain:
            # ===== mask-only 预热阶段 =====
            # 用 no_grad 跳过 backbone/FPN 的激活保存和梯度计算
            with torch.no_grad():
                raw_feats = self.backbone(images.tensors)
            # detach 掉计算图，确保后续反向不会回传到 backbone
            if isinstance(raw_feats, (list, tuple)):
                features = [f.detach() for f in raw_feats]
            else:
                features = raw_feats.detach()
        else:
            # ===== 正常联合训练或推理阶段 =====
            # backbone 和 neck 正常执行，保留梯度和激活
            features = self.backbone(images.tensors)

        if self.training:
            # 训练模式：走 Detect_Head 里的训练逻辑
            #   mask-only 阶段会只计算 mask_loss，不会调用 predictor/loss_evaluator
            #   联合训练阶段会同时计算检测 loss + mask loss
            loss_dict, log_loss_dict = self.heads(features, targets)
            return loss_dict, log_loss_dict
        else:
            # 推理模式：调用 Detect_Head 里的推理逻辑，输出结果 + 可视化数据
            result, eval_utils, visualize_preds = self.heads(
                features, targets, test=self.test
            )
            return result, eval_utils, visualize_preds