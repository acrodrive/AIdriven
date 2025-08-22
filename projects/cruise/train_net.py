import os
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
    # ZOD frames 루트 지정
    register_all_zod(zod_root="/home/appuser/AIdriven/datasets/zod", version="full")
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

# 이거 아래 중에 뭘 써야 하노..
# python projects/cruise/train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 1
# python projects/cruise/train_net.py --config-file aidriven/ZOD3D/zod3d_r50fpn.yaml --num-gpus 1
# python -m projects.cruise.train_zod_3d   --num-gpus 1   --weights pretrained/faster_rcnn_R_50_FPN_3x.pkl   --output aidriven/checkpoint/zod3d_r50fpn

# python -m projects.cruise.train_net \
#   --num-gpus 1 \
#   --weights pretrained/faster_rcnn_R_50_FPN_3x.pkl \
#   --output aidriven/checkpoint/zod3d_r50fpn

# trainval-frames-full.json 파일이 없다면 single_frames에 존재하지 않고 상위 폴더에 존재한다면 아래 명령어로 심볼릭 링크 생성
# ln -sf ../trainval-frames-full.json \
#  /home/appuser/AIdriven/datasets/zod/single_frames/trainval-frames-full.json