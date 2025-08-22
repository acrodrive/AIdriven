# Cruise/datasets/zod/zod_register.py
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from zod import ZodFrames  # frames subset 사용 가정

ZOD_CATS = ["Car", "Pedestrian", "Cyclist"]
CAT_TO_ID = {k: i for i, k in enumerate(ZOD_CATS)}

def _objects_to_detectron(objects):
    annos = []
    for obj in objects:
        cls_name = getattr(obj, "category", None) or getattr(obj, "label", None)
        if cls_name not in CAT_TO_ID:
            continue

        # 2D bbox
        if getattr(obj, "box2d", None):
            x1, y1, x2, y2 = obj.box2d.xyxy
        elif getattr(obj, "bbox", None):
            x1, y1, x2, y2 = obj.bbox.xyxy
        else:
            continue

        # 3D
        if getattr(obj, "box3d", None):
            center = obj.box3d.center  # (x,y,z)
            lwh = obj.box3d.lwh        # (l,w,h)
            yaw = getattr(obj.box3d, "yaw", None)
            if yaw is None and getattr(obj.box3d, "quaternion", None):
                yaw = obj.box3d.quaternion.yaw  # SDK가 제공하는 경우
        else:
            continue

        annos.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": CAT_TO_ID[cls_name],
            "zod_3d": {
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "lwh": [float(lwh[0]), float(lwh[1]), float(lwh[2])],
                "yaw": float(yaw) if yaw is not None else 0.0,
            },
        })
    return annos

def load_zod_split(zod_root, split="train", version="mini", limit=None):
    """
    version: "mini" 또는 "full" (다운로드한 버전에 맞추세요)
    split: "train" / "val"
    """
    # ★ version 필수!
    zf = ZodFrames(dataset_root=zod_root, version=version)

    # 간단 80/20 split (원하면 공식 split API로 바꿔도 됨)
    frames = sorted(zf.get_all(), key=lambda f: f.id)
    cut = int(len(frames) * 0.8)
    sel = frames[:cut] if split == "train" else frames[cut:]
    if limit:
        sel = sel[:limit]

    dataset = []
    for fm in sel:
        # 이미지 경로/크기 (SDK 버전에 따라 접근자가 다를 수 있음)
        img_path = fm.image.path
        height, width = fm.image.size[1], fm.image.size[0]

        objects = fm.annotations.objects if hasattr(fm.annotations, "objects") else []
        annos = _objects_to_detectron(objects)
        if not annos:
            continue

        dataset.append({
            "file_name": img_path,
            "image_id": fm.id,
            "height": height,
            "width": width,
            "annotations": annos,
        })
    return dataset

def register_zod(name, zod_root, split, version):
    DatasetCatalog.register(name, lambda: load_zod_split(zod_root, split, version))
    MetadataCatalog.get(name).set(thing_classes=ZOD_CATS)

def register_all_zod(zod_root="/home/appuser/AIdriven/Cruise/datasets/zod/zoddata/single_frames", version=None):
    """
    프로젝트나 환경변수에서 버전을 지정할 수 있게 함.
    """
    ver = version or os.getenv("ZOD_VERSION", "mini")  # 기본 mini
    register_zod("zod3d_train", zod_root, "train", ver)
    register_zod("zod3d_val",   zod_root, "val",   ver)
