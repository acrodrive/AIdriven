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
        ret = super().from_config(cfg, input_shape)
        ret["cfg"] = cfg
        return ret

    @configurable
    def __init__(self, *, cfg=None, **kwargs):
        # StandardROIHeads를 상속받음
        # self.box_head는 기존 head임
        super().__init__(**kwargs)
        assert self.box_head is not None, "box_head must be built by StandardROIHeads"
        
        # 기존 head의 가중치를 얼림
        for p in self.box_head.parameters():
            p.requires_grad = False
        
        # custom 3d head 추가
        self.box_predictor = FastRCNN3DOutputLayers(cfg, self.box_head.output_shape)

    def _forward_box(self, features, proposals):
        """
        StandardROIHeads._forward_box 규약:
        - self.training == True -> '손실 dict'만 반환
        - self.training == False -> 'pred_instances(Instances)'만 반환
        
        features: backbone에서 추출한 다중 스케일 특징맵 (P2, P3, P4, P5)
        proposals: RPN에서 생성된 객체 후보 영역들
        """
        # ⭐ head 입력부, box_pooler가 RoIAlign 수행 (proposal 영역에 대해 feature map에서 고정된 크기의 특징 추출)
        box_features = self.box_pooler( # 여러 해상도의 FPN 특징맵에서 RPN이 제안한 박스마다 고정 크기의 RoI 특징을 잘라오는 연산
            [features[f] for f in self.box_in_features], # 근데 이제 이미지 특징맵을 넣고
            [x.proposal_boxes for x in proposals], #RPN에서 생성된 proposals을 넣어서
        )

        # box_pooler의 출력은 결과 텐서로 배치 내 모든 proposal을 세로로 이어 붙인 형태임
        # 차원은 (모든 proposal 개수의 합, 채널 C, pooler_output, pooler_output)
        # 이미지 픽셀 좌표계를 쓰며, 학습 시에는 proposal에 GT가 매칭된 전경 샘플과 배경 샘플이 섞여 들어옴(❓)
        # 역전파 시 RoIAlign을 거친 그 레벨의 FPN 피처로 그래디언트가 흐름 (❓)

        # 특징 처리
        # ⭐ box_head는 이렇게 뽑힌 C×H×W RoI 특징을 상위 표현(❓)으로 변환하는 모듈
        box_features = self.box_head(box_features)  # (sum_props, rep_dim)
        # ⭐ box predictor 나야 커스텀 헤드. 근데 class는 왜 뱉니❓; 그리고 추론 된 거 아니야 여기서❓
        pred_class_logits, pred_proposal_deltas, pred_3d = self.box_predictor(box_features)

        if self.training:
            # 반드시 dict만 반환해야 함 (❓)
            # ⭐ 손실 계산
            losses = self.box_predictor.losses(
                (pred_class_logits, pred_proposal_deltas, pred_3d), proposals
            )
            # 혹시 내부가 실수로 튜플을 주면 방어적으로 dict만 취함
            if isinstance(losses, tuple):
                losses = losses[-1]
            return losses

        else:
            # 또 추론하는거여❓ 근데 instances가 정확히 뭐야❓
            pred_instances, _ = self.box_predictor.inference_cls_only(
                (pred_class_logits, pred_proposal_deltas), proposals
            )
            
            # custom head(self.box_predictor)로 
            self.box_predictor.inference_3d(box_features, pred_instances)
            return pred_instances

