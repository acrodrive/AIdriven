# modeling/roi_heads/roi_heads_3d.py
from typing import List, Dict
import torch
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures import Instances
from .fast_rcnn_3d import FastRCNN3DOutputLayers

@ROI_HEADS_REGISTRY.register()
class ROIHeads3D(StandardROIHeads):
    """
    box_predictor만 3D 지원 버전으로 교체.
    """
    def _init_box_head(self, cfg):
        super()._init_box_head(cfg)
        # self.box_head는 이미 구성됨. predictor만 교체
        self.box_predictor = FastRCNN3DOutputLayers(cfg, self.box_head.output_shape)

    def _forward_box(self, features, proposals):
        """
        StandardROIHeads._forward_box 복사 후 3D 분기 대응
        """
        box_features = self.box_pooler([features[f] for f in self.box_in_features], [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)     # (sum_proposals, rep_dim)
        pred_class_logits, pred_proposal_deltas, pred_3d = self.box_predictor(box_features)

        # 손실 or 추론
        if self.training:
            losses = self.box_predictor.losses((pred_class_logits, pred_proposal_deltas, pred_3d), proposals)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference((pred_class_logits, pred_proposal_deltas), proposals)
            # 3D 결과 부착
            self.box_predictor.inference_3d(box_features, pred_instances)
            return pred_instances, {}
