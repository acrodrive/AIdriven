import copy
import torch
from detectron2.structures import Boxes, Instances
from detectron2.data import detection_utils as utils

class ZOD3DMapper:
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
            instances.gt_boxes      = Boxes(torch.tensor(boxes, dtype=torch.float32))
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
