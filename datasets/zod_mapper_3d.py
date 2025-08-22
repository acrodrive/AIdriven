# datasets/zod_mapper_3d.py
import copy, torch, numpy as np
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, Instances

class ZOD3DMapper:
    def __init__(self, is_train=True, augment=True):
        self.is_train = is_train
        self.augs = T.AugmentationList([
            T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736),
                                 max_size=1333, sample_style="choice")
        ]) if (is_train and augment) else T.AugmentationList([])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        aug_input = T.AugInput(image)
        transforms = self.augs(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape
            ) for obj in dataset_dict.get("annotations", [])
        ]

        instances = Instances(image_shape)
        if len(annos):
            boxes = np.array([a["bbox"] for a in annos], dtype=np.float32)
            classes = np.array([a["category_id"] for a in annos], dtype=np.int64)
            instances.gt_boxes = Boxes(torch.from_numpy(boxes))
            instances.gt_classes = torch.from_numpy(classes)

            # --- 3D GT: ZOD center(lwh,yaw) -> bottom-center 변환 ---
            k3d = []
            y_sin_cos = []
            for a in annos:
                z3d = a["zod_3d"]
                cx, cy, cz = z3d["center"]
                l, w, h = z3d["lwh"]   # ZOD는 일반적으로 (l,w,h)
                yaw = z3d["yaw"]

                # bottom-center: z -= h/2
                bx, by, bz = cx, cy, cz - (h * 0.5)

                # 내부 회귀 순서는 (x,y,z,w,l,h)
                k3d.append([bx, by, bz, w, l, h])
                y_sin_cos.append([np.sin(yaw), np.cos(yaw)])

            instances.gt_boxes3d = torch.tensor(k3d, dtype=torch.float32)
            instances.gt_yaw_sincos = torch.tensor(y_sin_cos, dtype=torch.float32)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2,0,1).copy())
        dataset_dict["instances"] = instances
        return dataset_dict
