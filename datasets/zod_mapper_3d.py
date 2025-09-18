import copy
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, Boxes, Instances

class ZOD3DMapper:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736),
                max_size=1333,
                sample_style="choice"
            )
        ]) if is_train else T.AugmentationList([])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        if self.is_train:
            utils.check_image_size(dataset_dict, image)
            
            anns = dataset_dict.get("annotations", [])
            if self.is_train and not anns:
                return None  # 또는 raise SkipSample
            
            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]
            
            instances = Instances(image_shape)
            
            boxes = []
            classes = []
            boxes_3d = []
            
            for annotation in dataset_dict["annotations"]:
                bbox = BoxMode.convert(
                    annotation["bbox"],
                    annotation["bbox_mode"],
                    BoxMode.XYXY_ABS
                )
                bbox = transforms.apply_box([bbox])[0]
                boxes.append(bbox)
                classes.append(annotation["category_id"])
                
                if "box3d" in annotation:
                    box3d = annotation["box3d"]
                    boxes_3d.append([
                        *box3d["center"],
                        *box3d["lwh"],
                        box3d["yaw"]
                    ])
            
            if len(boxes):
                # Convert list to numpy array first
                boxes = np.asarray(boxes, dtype=np.float32)
                instances.gt_boxes = Boxes(torch.from_numpy(boxes))
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
                
                if boxes_3d:
                    # Convert 3D boxes to tensor
                    boxes_3d = np.asarray(boxes_3d, dtype=np.float32)
                    instances.gt_boxes_3d = torch.from_numpy(boxes_3d)
                    
            dataset_dict["instances"] = instances
        
        # Convert image to tensor format
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        
        return dataset_dict