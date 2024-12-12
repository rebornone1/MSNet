import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU
# from .attentions.SCSA import SCSA
# from .utils import DouLSK
# from .utils import MCA
# from .utils import EVit

# from .neck.feature_attention import ScaleFeatureSelection
# from .utils import SRM_1280
# from .utils import SRM
# from .utils import ULSAM
# from .utils import STB
# from efficientvit.models.nn.ops import EfficientViTBlock
# from .utils import WTConv
# from .utils import EMA
# from .utils import MLKA
# from .utils import SPASPP

class PSENet(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)

        self.det_head = build_head(detection_head)
        # self.SCSA = SCSA(1024, 4)
        # self.DLKN = DouLSK.DouLSK(1024)
        # self.MCA = MCA.MCALayer(1024)
        # self.Evit = EVit.CascadedGroupAttention(1024, 64, 4, 4, 184)
        # self.Evit = EVit.LocalWindowAttention(1024, 64, 4, 4)
        # self.SPASPP = SPASPP.SPASPP(1024, 1024, 1024)
        # self.MAB = MLKA.MAB(1024)
        # self.EMA = EMA.EMA(1024)
        # self.WTC = WTConv.WTConv2d(1024,1024)
        # self.effvit = EfficientViTBlock(1024)
        # self.STB = STB.SwinTransformerBlock(1024, 1024, 8, 2) # 原本为(1024, 1024, 8, 8)
        # self.concat_attention = ScaleFeatureSelection(1024, 256, attention_type='scale_spatial')
        # self.SRM_1280 = SRM_1280.SRM()
        # self.SRM = SRM.SRM()
        # self.ULSAM = ULSAM.ULSAM(1024, 1024, 184, 184, 4)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # FPN
        f1, f2, f3, f4 = self.fpn(f[0], f[1], f[2], f[3])

        # f1, f2, f3, f4 = self.SRM(f1, f2, f3, f4)
        # f = self.SRM_1280(f1, f2, f3, f4)
        # print(f.shape)
        # f = self.ULSAM(f)
        # print(f.shape)
        f = torch.cat((f1, f2, f3, f4), 1)
        # f = self.SCSA(f)
        # f = self.DLKN(f)
        # f = self.MCA(f)
        # f = self.Evit(f)
        # f = self.SPASPP(f)
        # f = self.MAB(f)

        # f = self.EMA(f)
        # f = self.WTC(f)

        # f = self.STB(f)
        # f = self.concat_attention(f, [f1, f2, f3, f4])
        # f = self.effvit(f)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection

        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 1)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs
