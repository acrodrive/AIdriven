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
    """
    학습할 때 데이터셋을 잘라서 모델에게 하나씩(or 여러 개씩) 공급하는 역할을 하는 Data Loader를 build함

    예를 들어 전방 영상을 10만 장 가지고 있다면 이것을 한 번에 다 모델에 넣을 수는 없음
    따라서 Data Loader가 데이터를 batch(예: 16장) 단위로 꺼내서 VRAM에 올림
    또한 매 epoch 마다 데이터 순서를 섞어서(shuffle) 모델이 데이터 순서에 의존하지 않도록 해줌.

    즉, 데이터 로더는 학습 데이터를 한 덩어리씩 모델에 전달하는 배급기라고 볼 수 있음.
    """
    
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
    
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0025 * (cfg.SOLVER.IMS_PER_BATCH / 4.0)
    scale = 4.0  # 4→1 로 줄였으니 4배
    cfg.SOLVER.MAX_ITER = int(9000 * scale)
    cfg.SOLVER.STEPS = (int(6000 * scale), int(8000 * scale))
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # (기본 512)
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500         # (기본 1000)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1000
    cfg.MODEL.RESNETS.FREEZE_AT = 2
    #cfg.INPUT.MIN_SIZE_TRAIN = (640, 720, 800)
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    #cfg.INPUT.MIN_SIZE_TEST  = 800
    #cfg.INPUT.MAX_SIZE_TEST  = 1333
    
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