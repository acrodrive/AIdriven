"""import os
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader, build_detection_test_loader

# 데이터셋 등록/매퍼
from datasets.zod_register import register_all_zod
from datasets.zod_mapper_3d import ZOD3DMapper

# ROIHeads3D 등록(임포트만 해도 registry에 등록)
from aidriven.modeling.head.roi_heads_3d import ROIHeads3D  # noqa: F401

class ZOD3DTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ZOD3DMapper(is_train=True))
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=ZOD3DMapper(is_train=False, augment=False))

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file("configs/zod_config.yaml")
    # COCO R50-FPN 3x 가중치 초기화
    if args.weights and os.path.exists(args.weights):
        cfg.MODEL.WEIGHTS = args.weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    if args.ims_per_batch: cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    if args.base_lr:       cfg.SOLVER.BASE_LR = args.base_lr
    if args.output:
        cfg.OUTPUT_DIR = args.output
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)
    return cfg

def main(args):
    print("Registering ZOD dataset...")
    zod_root = "/home/appuser/AIdriven/datasets/zod"
    print(f"ZOD root path exists: {os.path.exists(zod_root)}")
    
    single_frames_dir = os.path.join(zod_root, "single_frames")
    print(f"Single frames dir exists: {os.path.exists(single_frames_dir)}")
    
    register_all_zod(zod_root=zod_root, version="full")
    
    from detectron2.data.catalog import DatasetCatalog
    dataset_dicts = DatasetCatalog.get("zod3d_train")
    print(f"\nDataset summary:")
    print(f"- Total samples found: {len(dataset_dicts)}")
    
    if len(dataset_dicts) > 0:
        n_annos = len(dataset_dicts[0].get('annotations', []))
        print(f"- First sample has {n_annos} annotations")
    
    cfg = setup(args)
    trainer = ZOD3DTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()    

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--weights", type=str, default="", help="COCO pkl/url or local pkl")
    parser.add_argument("--ims-per-batch", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    
    args = parser.parse_args()
    launch(main, args.num_gpus or 1, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))

# 코드 실행 방법
# python -m projects.cruise.train_net   --num-gpus 1   --weights pretrained/faster_rcnn_R_50_FPN_3x.pkl   --output aidriven/checkpoint/zod3d_r50fpn
# 보기 좋게 표현하자면
# python -m projects.cruise.train_net \
#   --num-gpus 1 \
#   --weights pretrained/faster_rcnn_R_50_FPN_3x.pkl \
#   --output aidriven/checkpoint/zod3d_r50fpn

# trainval-frames-full.json 경로 문제가 발생할 수 있음
# 최상위 폴더에만 해당 파일이 존재하는데 굳이 하위 경로에서 참조하려고 프로그램에서 시도하는 경우
# 아래 명령어로 심볼릭 링크 생성
# ln -sf ../trainval-frames-full.json \
#  /home/appuser/AIdriven/datasets/zod/single_frames/trainval-frames-full.json"""

import os
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from datasets.zod_register import register_all_zod
from datasets.zod_mapper_3d import ZOD3DMapper
from aidriven.modeling.head.roi_heads_3d import ROIHeads3D  # noqa: F401

class ZOD3DTrainer(DefaultTrainer):
    """Custom trainer for 3D object detection."""
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ZOD3DMapper(is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=ZOD3DMapper(is_train=False))

def setup(args):
    """Setup training configuration."""
    cfg = get_cfg()
    cfg.merge_from_file("configs/zod_config.yaml")

    # Model weights initialization
    if args.weights and os.path.exists(args.weights):
        cfg.MODEL.WEIGHTS = args.weights
    
    # Training parameters
    if args.output:
        cfg.OUTPUT_DIR = args.output
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    default_setup(cfg, args)
    return cfg

def main(args):
    """Main training function."""
    # Dataset registration should be minimal here
    register_all_zod("/home/appuser/AIdriven/datasets/zod")
    cfg = setup(args)
    cfg.DATASETS.TRAIN = ("zod_3class_train",)
    cfg.DATASETS.TEST  = ("zod_3class_val",)
    trainer = ZOD3DTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    
    launch(main, args.num_gpus or 1, num_machines=args.num_machines, 
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))