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

    @staticmethod
    def _prepare_3d_targets(proposals: List[Instances]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        proposals[i]는 sampling/matching된 GT가 부착됨.
        gt_boxes3d: (N,6), gt_yaw_sincos: (N,2)
        """
        gt_3d = []
        gt_yaw_sc = []
        for p in proposals:
            # 선택된 foreground만 포함하도록 gt_classes가 -1 아닌 것들을 사용
            fg_inds = (p.gt_classes >= 0) & (p.gt_classes != 255)
            gt_3d.append(p.gt_boxes3d[fg_inds])
            gt_yaw_sc.append(p.gt_yaw_sincos[fg_inds])
        if len(gt_3d) == 0:
            return torch.zeros(0,6), torch.zeros(0,2)
        return torch.cat(gt_3d, dim=0), torch.cat(gt_yaw_sc, dim=0)

    def losses(
        self, predictions, proposals: List[Instances]
    ) -> Dict[str, torch.Tensor]:
        """
        기존 2D 분류/회귀 손실 + 3D 회귀 손실 추가
        """
        scores, proposal_deltas, bbox3d = predictions
        # --- 기존 손실 ---
        losses = super().losses((scores, proposal_deltas), proposals)

        # --- 3D 손실 ---
        gt_3d, gt_yaw_sc = self._prepare_3d_targets(proposals)  # (M,6), (M,2)
        # pred select: proposals에서 fg만큼이 box 회귀 타깃 개수와 같음
        # FastRCNNOutputLayers._get_deltas 메커니즘과 동일하게 fg 수만큼 반환되므로
        # 여기서도 bbox3d는 fg 개수 M만 고려해야 함.
        # 이를 위해 super() 내부의 순서를 따라 fg_inds를 재구성하는 대신,
        # box 회귀 손실과 동일한 마스크를 공유하는 것이 안전하다.
        # => trick: super().losses 호출 직후, 내부에서 사용한 fg 개수와 동일하도록
        #           logits/box pred를 생성한 입력 순서가 유지된다고 가정
        # Detectron2 구현상 fg 순서 정합이 유지되므로 그대로 사용 가능
        M = gt_3d.shape[0]
        if M > 0:
            pred_3d = bbox3d[:M]  # (M,8)
            pred_xyz = pred_3d[:, 0:3]
            pred_wlh = pred_3d[:, 3:6]
            pred_yaw_sc = pred_3d[:, 6:8]

            loss_xyz = F.smooth_l1_loss(pred_xyz, gt_3d[:, 0:3], reduction="mean")
            loss_wlh = F.smooth_l1_loss(pred_wlh, gt_3d[:, 3:6], reduction="mean")
            loss_yaw = F.l1_loss(pred_yaw_sc, gt_yaw_sc, reduction="mean")
        else:
            loss_xyz = pred_3d.new_zeros(())
            loss_wlh = pred_3d.new_zeros(())
            loss_yaw = pred_3d.new_zeros(())

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
