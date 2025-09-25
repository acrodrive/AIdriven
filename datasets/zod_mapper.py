"""
원본 데이터(이미지, 어노테이션)는 모델이 곧바로 사용할 수 있는 형태가 아님.
예를 들어 다음과 같은 원본이 데이터가 있음

{
  "file_name": "frame_00123.png",
  "annotations": [
    {"bbox": [100, 200, 300, 400], "category_id": 2},
    {"bbox": [50, 50, 80, 120], "category_id": 1}
  ]
}


이것을 그대로 모델에 입력할 수 없음.

이미지를 불러와서 Tensor로 바꿔야 하고 박스 좌표를 정규화(normalize)하고, 데이터 증강(augmentation: flip, crop, color jitter 등)을 적용해야 함.

이렇게 데이터 전처리를 담당하는 것이 mapper임 -> 근데 이거 왜 register가 하고 있음? -> 데이터로더에서 배치 단위로 불러올 때 적용되는 변환/가공 단계라고 하고 register에서 준비해둔 dict들을 받아서 모델에 넣을 수 있는 이미지 형태(tensor 등)로 바꿔준대 그럼 됐네

Detectron2의 기본 mapper는 DatasetMapper임
하지만 ZOD와 같은 외부 데이터 셋을 쓰는 경우 당연히 거기에서 정의하는 원본 데이터는 다른 형태일 것임.
그렇게 되면 기본 매퍼가 이해하지 못하니까, ZODMapper 같은 커스텀 매퍼를 만들어야 함.
"""

import copy
import torch
from detectron2.structures import Boxes, Instances
from detectron2.data import detection_utils as utils

class ZODMapper:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)

        # 1) Load image (BGR) -> RGB tensor; make it writable to avoid warnings
        image = utils.read_image(d["file_name"], format="BGR")
        utils.check_image_size(d, image)
        image_rgb = image[:, :, ::-1].copy()
        d["image"] = torch.as_tensor(image_rgb.transpose(2, 0, 1)).contiguous()

        # 2) Build Instances: keep ONLY objects that have 3D labels, so field lengths match
        H, W = image.shape[0], image.shape[1]
        instances = Instances((H, W))

        entries = []
        for ann in d.get("annotations", []):
            if "bbox" not in ann or "category_id" not in ann:
                continue
            b3d = ann.get("bbox3d", None)
            ysc = ann.get("yaw_sincos", ann.get("yaw_sc"))
            if b3d is None or ysc is None:
                continue
            entries.append((ann["bbox"], ann["category_id"], b3d, ysc))

        if entries:
            boxes      = [e[0] for e in entries]
            classes    = [e[1] for e in entries]
            boxes3d    = [e[2] for e in entries]
            yaw_sincos = [e[3] for e in entries]

            # All tensors have identical length by construction
            instances.gt_boxes      = Boxes(torch.tensor(boxes, dtype=torch.float32)) # 2dbox는 쫌 빼라
            instances.gt_classes    = torch.tensor(classes, dtype=torch.int64)
            instances.gt_boxes3d    = torch.tensor(boxes3d, dtype=torch.float32)   # (N,6) -> [x,y,z,w,l,h]
            instances.gt_yaw_sincos = torch.tensor(yaw_sincos, dtype=torch.float32) # (N,2) -> [sin(yaw), cos(yaw)]
        else:
            # Empty tensors across the board keep Instances length consistent (0)
            instances.gt_boxes      = Boxes(torch.zeros((0,4), dtype=torch.float32))
            instances.gt_classes    = torch.zeros((0,), dtype=torch.int64)
            instances.gt_boxes3d    = torch.zeros((0,6), dtype=torch.float32)
            instances.gt_yaw_sincos = torch.zeros((0,2), dtype=torch.float32)

        d.pop("annotations", None)
        d["instances"] = instances
        return d
