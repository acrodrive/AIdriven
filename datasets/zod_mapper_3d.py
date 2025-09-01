import copy
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, Boxes, Instances

class ZOD3DMapper:
    def __init__(self, is_train=True, augment=True):
        self.is_train = is_train
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736),
                max_size=1333,
                sample_style="choice"
            )
        ]) if (is_train and augment) else T.AugmentationList([])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        if self.is_train:
            utils.check_image_size(dataset_dict, image)
            
            # Apply augmentations
            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]  # height, width
            
            # Create Instances object for gt_instances
            instances = Instances(image_shape)
            
            # Collect boxes, classes and 3D info
            boxes = []
            classes = []
            boxes_3d = []
            
            for annotation in dataset_dict["annotations"]:
                # Transform 2D bbox
                bbox = BoxMode.convert(
                    annotation["bbox"],
                    annotation["bbox_mode"],
                    BoxMode.XYXY_ABS
                )
                bbox = transforms.apply_box([bbox])[0]
                boxes.append(bbox)
                classes.append(annotation["category_id"])
                
                # Collect 3D information if available
                if "box3d" in annotation:
                    box3d = annotation["box3d"]
                    boxes_3d.append({
                        "center": torch.tensor(box3d["center"], dtype=torch.float32),
                        "lwh": torch.tensor(box3d["lwh"], dtype=torch.float32),
                        "yaw": torch.tensor([box3d["yaw"]], dtype=torch.float32),
                    })
            
            # Convert to tensor format
            if len(boxes):
                instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
                
                if boxes_3d:
                    instances.gt_boxes_3d = boxes_3d
                    
            dataset_dict["instances"] = instances
        
        # Convert image to tensor format
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        
        return dataset_dict