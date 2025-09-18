# modeling/roi_heads/fast_rcnn_3d.py
import torch, torch.nn as nn, torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats
from detectron2.structures import Boxes, Instances
from typing import List, Dict, Tuple

class FastRCNN3DOutputLayers(FastRCNNOutputLayers):
    """
    기존 FastRCNNOutputLayers에 3D 회귀 분기를 추가:
      - bbox_3d_pred: (x,y,z,w,l,h,sinψ,cosψ) -> 8 차원 (class-agnostic)
    """
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__(cfg, input_shape)
        rep_dim = self.cls_score.in_features
        self.bbox_3d_pred = nn.Linear(rep_dim, 8)
        nn.init.normal_(self.bbox_3d_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_3d_pred.bias, 0)

    def forward(self, x):
        # x: (N, rep_dim) from box head
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        bbox3d = self.bbox_3d_pred(x)  # (N, 8)
        return scores, proposal_deltas, bbox3d

    def _prepare_3d_targets(self, proposals):
        """
        proposals: list[Instances] (학습용, gt_classes가 라벨링된 상태)
        반환: (cat over images) gt_3d: (M,6), gt_yaw_sc: (M,2)
        -> M은 배치 내 포지티브(전경) proposal 총 개수
        """
        gt_3d_list = []
        gt_yaw_list = []

        # self.num_classes 가 ROIHeads의 num_classes와 일치한다고 가정
        num_classes = getattr(self, "num_classes", None)

        for p in proposals:
            # 전경 인덱스 추출 (배경 = num_classes)
            # (Detectron2 기본 규약: 0..C-1 foreground, C background, -1 ignore)
            gtcls = p.gt_classes
            if num_classes is not None:
                fg_mask = (gtcls >= 0) & (gtcls < num_classes)
            else:
                # 혹시 num_classes 접근이 어려우면 배경이 아닌 것만 전경으로 취급
                fg_mask = (gtcls >= 0)
            fg_inds = fg_mask.nonzero().squeeze(1)

            # 전경이 0개면 빈 텐서 푸시하고 다음 이미지로
            if fg_inds.numel() == 0:
                device = gtcls.device
                gt_3d_list.append(torch.zeros((0, 6), device=device, dtype=torch.float32))
                gt_yaw_list.append(torch.zeros((0, 2), device=device, dtype=torch.float32))
                continue

            # 3D 필드가 proposals에 없으면(혹은 길이 불일치) 안전 스킵
            if (not hasattr(p, "gt_boxes3d")) or (not hasattr(p, "gt_yaw_sincos")):
                device = gtcls.device
                gt_3d_list.append(torch.zeros((0, 6), device=device, dtype=torch.float32))
                gt_yaw_list.append(torch.zeros((0, 2), device=device, dtype=torch.float32))
                continue

            # 정상 경로: 전경 인덱스만 추출
            gt_3d_list.append(p.gt_boxes3d[fg_inds])
            gt_yaw_list.append(p.gt_yaw_sincos[fg_inds])

        if len(gt_3d_list) == 0:
            # 방어적 처리: 비정상 케이스
            device = proposals[0].gt_classes.device if len(proposals) else torch.device("cpu")
            return (torch.zeros((0, 6), device=device, dtype=torch.float32),
                    torch.zeros((0, 2), device=device, dtype=torch.float32))

        # 전 이미지 합치기 (전경이 없던 이미지는 (0,*) 이므로 문제 없음)
        return torch.cat(gt_3d_list, dim=0), torch.cat(gt_yaw_list, dim=0)


    def losses(self, predictions, proposals: List[Instances]) -> Dict[str, torch.Tensor]:
        """
        기존 2D 분류/회귀 손실 + 3D 회귀 손실 추가.
        빈 배치/3D 미존재 시에도 안전하게 0 스칼라 손실을 반환하도록 방어 코드 포함.
        """
        # --- predictions 안전 언패킹 ---
        if isinstance(predictions, (list, tuple)):
            if len(predictions) == 3:
                scores, proposal_deltas, bbox3d = predictions
            elif len(predictions) == 2:
                scores, proposal_deltas = predictions
                bbox3d = None
            else:
                raise ValueError(f"Unexpected predictions length: {len(predictions)}")
        elif isinstance(predictions, dict):
            # 가능한 키 이름 호환
            scores = predictions.get("scores", predictions.get("pred_class_logits"))
            proposal_deltas = predictions.get("proposal_deltas", predictions.get("pred_proposal_deltas"))
            bbox3d = predictions.get("pred_3d", predictions.get("bbox3d"))
        else:
            raise TypeError(f"Unexpected predictions type: {type(predictions)}")

        # --- 2D 손실 ---
        losses = super().losses((scores, proposal_deltas), proposals)

        # --- 3D 타깃 준비 ---
        gt_3d, gt_yaw_sc = self._prepare_3d_targets(proposals)  # (M,6), (M,2)
        zero = scores.new_zeros(())  # pred_3d가 없어도 안전하게 0 스칼라 생성

        M = int(gt_3d.shape[0]) if hasattr(gt_3d, "shape") else 0
        m_use = min(M, int(bbox3d.shape[0])) if (bbox3d is not None and hasattr(bbox3d, "shape")) else 0

        if m_use > 0:
            pred_3d = bbox3d[:m_use]              # (m_use, 8) = [x,y,z,w,l,h,sin(yaw),cos(yaw)]
            pred_xyz = pred_3d[:, 0:3]
            pred_wlh = pred_3d[:, 3:6]
            pred_yaw_sc = pred_3d[:, 6:8]

            loss_xyz = F.smooth_l1_loss(pred_xyz, gt_3d[:m_use, 0:3], reduction="mean")
            loss_wlh = F.smooth_l1_loss(pred_wlh, gt_3d[:m_use, 3:6], reduction="mean")
            loss_yaw = F.l1_loss(pred_yaw_sc, gt_yaw_sc[:m_use], reduction="mean")
        else:
            loss_xyz = zero
            loss_wlh = zero
            loss_yaw = zero

        losses.update({
            "loss_3d_xyz": loss_xyz,
            "loss_3d_wlh": loss_wlh,
            "loss_3d_yaw_sc": loss_yaw,
        })
        return losses

    @torch.no_grad()
    def inference_3d(self, box_features: torch.Tensor, pred_instances: List[Instances]):
        """
        테스트 시 3D 회귀 결과를 pred_instances에 부착
        """
        _, _, bbox3d = self.forward(box_features)
        # fast_rcnn inference와 같은 순서로 정렬되어 들어옴
        start = 0
        for inst in pred_instances:
            n = len(inst)
            cur = bbox3d[start:start+n]
            start += n
            xyz = cur[:, 0:3]
            wlh = cur[:, 3:6]
            yaw_sc = cur[:, 6:8]
            yaw = torch.atan2(yaw_sc[:,0], yaw_sc[:,1])  # atan2(sin, cos) => ψ
            inst.pred_box3d = torch.cat([xyz, wlh, yaw.unsqueeze(1)], dim=1)  # (n,7)
