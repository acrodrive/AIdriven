# datasets/zod_register.py
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# (1) ZOD SDK 로딩
from zod import ZodFrames  # frames subset 사용 예

ZOD_CATS = ["Car", "Pedestrian", "Cyclist"]  # 사용 대상 (필요시 확장)
CAT_TO_ID = {k:i for i,k in enumerate(ZOD_CATS)}

def _objects_to_detectron(objects):
    annos = []
    for obj in objects:
        # ZOD SDK에서 제공하는 클래스/박스 접근 (SDK 버전에 따라 속성명이 약간 다를 수 있음)
        cls_name = getattr(obj, "category", None) or getattr(obj, "label", None)
        if cls_name not in CAT_TO_ID:
            continue

        # 2D bbox (xyxy)
        if getattr(obj, "box2d", None):
            x1, y1, x2, y2 = obj.box2d.xyxy
        elif getattr(obj, "bbox", None):  # 백업
            x1, y1, x2, y2 = obj.bbox.xyxy
        else:
            continue

        # 3D center/size/yaw  (center=(x,y,z), size=(l,w,h))
        # orientation: yaw or quaternion 제공. quaternion이면 yaw로 변환 필요.
        if getattr(obj, "box3d", None):
            center = obj.box3d.center  # (x,y,z)
            lwh = obj.box3d.lwh        # (l,w,h)
            yaw = getattr(obj.box3d, "yaw", None)
            if yaw is None and getattr(obj.box3d, "quaternion", None):
                # SDK가 yaw 변환 유틸을 제공하는 경우 사용. 여기서는 예시로 제공:
                yaw = obj.box3d.quaternion.yaw  # SDK가 제공할 경우
        else:
            continue

        annos.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": CAT_TO_ID[cls_name],
            "zod_3d": {
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "lwh": [float(lwh[0]), float(lwh[1]), float(lwh[2])],
                "yaw": float(yaw) if yaw is not None else 0.0
            }
        })
    return annos

def load_zod_split(zod_root, split="train", limit=None):
    """
    Frames split 예시: train/val 목록은 ZOD가 제공하는 메타를 따르거나
    임의로 나눌 수 있습니다. 여기선 SDK의 frames를 전부 불러온 뒤
    간단히 80/20로 나누는 예시를 보입니다.
    """
    zf = ZodFrames(dataset_root=zod_root)
    frames = zf.get_all()  # 모든 frame 메타
    frames = sorted(frames, key=lambda f: f.id)

    # 간단 split
    ratio = 0.8
    cut = int(len(frames) * ratio)
    sel = frames[:cut] if split == "train" else frames[cut:]
    if limit:
        sel = sel[:limit]

    dataset = []
    for fm in sel:
        # 이미지 경로/크기
        img_path = fm.image.path  # DNAT/BLUR 중 하나. 필요하면 설정으로 선택
        height, width = fm.image.size[1], fm.image.size[0]  # (w,h) → (h,w)

        # annotation 객체들
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
            # 필요시 calibration 파일 경로 포함:
            # "zod_calib": fm.calibration (SDK가 제공하는 경로/객체)
        })
    return dataset

def register_zod(name, zod_root, split):
    DatasetCatalog.register(name, lambda: load_zod_split(zod_root, split))
    MetadataCatalog.get(name).set(thing_classes=ZOD_CATS)

def register_all_zod(zod_root="/datasets/ZOD/frames"):
    register_zod("zod3d_train", zod_root, "train")
    register_zod("zod3d_val",   zod_root, "val")
