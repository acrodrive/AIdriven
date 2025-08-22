# Cruise/model/head/roi_heads_3d.py
from typing import List
import torch
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures import Instances
from .fast_rcnn_3d import FastRCNN3DOutputLayers

@ROI_HEADS_REGISTRY.register()
class ROIHeads3D(StandardROIHeads):
    """
    Detectron2 v0.6 규약에 맞춘 ROIHeads:
      - __init__는 키워드 전용 인자(**kwargs)로 부모와 동일하게 받고
      - from_config에서 부모 설정을 그대로 가져온 뒤 cfg를 추가 전달
      - 초기화 후 box_predictor만 3D용으로 교체
      - _forward_box에서 3D 분기 손실/추론 처리
    """

    @classmethod
    def from_config(cls, cfg, input_shape):
        # 부모가 필요한 키워드 인자들을 모두 구성
        ret = super().from_config(cfg, input_shape)
        # 우리 predictor 생성에 cfg가 필요하므로 함께 넘겨준다
        ret["cfg"] = cfg
        return ret

    @configurable
    def __init__(self, *, cfg=None, **kwargs):
        # 부모 초기화(여기서 box_pooler/box_head/기본 box_predictor까지 만들어짐)
        super().__init__(**kwargs)
        assert self.box_head is not None, "box_head must be built by StandardROIHeads"
        # box predictor만 3D 버전으로 교체
        self.box_predictor = FastRCNN3DOutputLayers(cfg, self.box_head.output_shape)

    def _forward_box(self, features, proposals):
        """
        StandardROIHeads._forward_box를 기반으로,
        FastRCNN3DOutputLayers의 추가 출력/손실을 처리.
        """
        box_features = self.box_pooler(
            [features[f] for f in self.box_in_features],
            [x.proposal_boxes for x in proposals],
        )
        box_features = self.box_head(box_features)  # (sum_props, rep_dim)

        pred_class_logits, pred_proposal_deltas, pred_3d = self.box_predictor(box_features)

        if self.training:
            losses = self.box_predictor.losses(
                (pred_class_logits, pred_proposal_deltas, pred_3d), proposals
            )
            return [], losses
        else:
            # 2D 박스/클래스는 부모 predictor의 inference를 그대로 활용
            pred_instances, _ = self.box_predictor.inference(
                (pred_class_logits, pred_proposal_deltas), proposals
            )
            # 3D 결과 부착
            self.box_predictor.inference_3d(box_features, pred_instances)
            return pred_instances, {}
