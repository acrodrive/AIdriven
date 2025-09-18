
import copy
import torch
import cv2
from detectron2.structures import Boxes, Instances
from detectron2.data import detection_utils as utils

class ZOD3DMapper:
    def __init__(self, is_train=True, augmentations=None):
        self.is_train = is_train
        self.augmentations = augmentations  # not used here; keep simple

    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)

        # 1) load image
        image = utils.read_image(d["file_name"], format="BGR")
        image = image[:, :, ::-1]  # to RGB
        utils.check_image_size(d, image)

        # 2) set tensor
        d["image"] = torch.as_tensor(image.transpose(2, 0, 1)).contiguous()

        if "annotations" in d:
            tw, th = d["image"].shape[2], d["image"].shape[1]
            boxes = []
            classes = []
            boxes3d = []
            yaw_sc = []

            for ann in d["annotations"]:
                # basic 2D
                if "bbox" not in ann or "category_id" not in ann:
                    continue
                boxes.append(ann["bbox"])
                classes.append(ann["category_id"])

                # extras for 3D
                if "bbox3d" in ann and "yaw_sc" in ann:
                    boxes3d.append(ann["bbox3d"])
                    yaw_sc.append(ann["yaw_sc"])
                else:
                    # if training 3D head, skip this instance from all lists to keep indices aligned
                    boxes.pop()
                    classes.pop()

            # build Instances
            instances = Instances((image.shape[0], image.shape[1]))

            if boxes:
                instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            else:
                instances.gt_boxes = Boxes(torch.zeros((0,4), dtype=torch.float32))
                instances.gt_classes = torch.zeros((0,), dtype=torch.int64)

            if boxes3d:
                instances.gt_boxes3d = torch.tensor(boxes3d, dtype=torch.float32)  # (N,6): x,y,z,w,l,h
                instances.gt_yaw_sc  = torch.tensor(yaw_sc,  dtype=torch.float32)  # (N,2): sin,cos
            else:
                instances.gt_boxes3d = torch.zeros((0,6), dtype=torch.float32)
                instances.gt_yaw_sc  = torch.zeros((0,2), dtype=torch.float32)

            d["instances"] = instances

            # you can drop annotations to save RAM
            d.pop("annotations", None)

        return d
