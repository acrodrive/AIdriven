import os, glob, json
from typing import List, Dict, Any, Tuple, Optional
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# ---- helpers added ----
def _bbox_from_geom_coordinates(obj):
    try:
        geom = obj.get("geometry", {})
        if not isinstance(geom, dict):
            return None, None
        gtype = str(geom.get("type", "")).lower()
        coords = geom.get("coordinates")
        if gtype not in ("multipoint", "polygon") or not isinstance(coords, (list, tuple)):
            return None, None
        # polygon nesting [[[x,y],...]]
        if gtype == "polygon" and coords and isinstance(coords[0], (list, tuple)) and len(coords) > 0 \
           and coords[0] and isinstance(coords[0][0], (list, tuple)):
            pts = coords[0]
        else:
            pts = coords
        xs, ys = [], []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0])); ys.append(float(p[1]))
            elif isinstance(p, dict) and ("x" in p and "y" in p):
                xs.append(float(p["x"])); ys.append(float(p["y"]))
        if not xs or not ys:
            return None, None
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2], BoxMode.XYXY_ABS
        return None, None
    except Exception:
        return None, None

def _bbox_from_simple_with_geom(obj):
    try:
        orig = globals().get("_bbox_from_simple", None) # 여기에서 props로 bbox랑 mode가 아니라 한번에 3dbox와 q rot 가져와야 함
        if callable(orig):
            bbox, mode = orig(obj)
            if bbox and mode is not None:
                return bbox, mode
    except Exception:
        pass
    return _bbox_from_geom_coordinates(obj) # 야 geo로 가져오면 안된다 여기로 들어가면 안됨

def _extract_3d_extras(obj):
    """Return (bbox3d, yaw_sincos) or (None, None)."""
    try:
        props = obj.get("properties", obj)
        loc3d = props.get("location_3d", {})
        sizeL = props.get("size_3d_length", None)
        sizeW = props.get("size_3d_width", None)
        sizeH = props.get("size_3d_height", None)
        qw = props.get("orientation_3d_qw", None)
        qx = props.get("orientation_3d_qx", None)
        qy = props.get("orientation_3d_qy", None)
        qz = props.get("orientation_3d_qz", None)

        if isinstance(loc3d, dict):
            coords3d = loc3d.get("coordinates", None)
        else:
            coords3d = None

        if coords3d and sizeL is not None and sizeW is not None and sizeH is not None:
            x, y, z = float(coords3d[0]), float(coords3d[1]), float(coords3d[2])
            w, l, h = float(sizeW), float(sizeL), float(sizeH)
            bbox3d = [x, y, z, w, l, h]

            if None not in (qw, qx, qy, qz):
                # yaw about Z
                yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                yaw_sincos = [math.sin(yaw), math.cos(yaw)]
            else:
                yaw_sincos = None
            return bbox3d, yaw_sincos
    except Exception:
        pass
    return None, None
# ---- end helpers ----

import math

THING_CLASSES = ["car", "pedestrian", "cyclist"]
CAT_TO_ID = {k:i for i,k in enumerate(THING_CLASSES)}

# 이미지: single_frames/<id>/(camera_front_blur|camera_front) 하위의 첫 이미지 선택
def _find_front_image(frame_dir: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    for sub in ("camera_front_blur", "camera_front"):
        root = os.path.join(frame_dir, sub)
        if not os.path.isdir(root):
            continue
        hits = []
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
            hits.extend(glob.glob(os.path.join(root, ext)))
        if hits:
            hits.sort()
            path = hits[0]
            # PIL 있으면 크기 읽기 (없으면 None으로 두어도 Detectron2가 처리함)
            W = H = None
            try:
                from PIL import Image
                with Image.open(path) as im:
                    W, H = im.size
            except Exception:
                pass
            return path, W, H
    return None, None, None

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

# see https://zod.zenseact.com/annotations
VEHICLE_TO_CAR = {"car","van","truck","trailer","bus","heavyequip","tramtrain","other","inconclusive"}
HUMAN_TO_PED   = {"pedestrian"}
VULN_TO_CYC    = {"bicycle","motorcycle","personaltransporter","other","inconclusive"}

def _to_std3(cls_raw: str, typ_raw: str) -> Optional[str]:
    cls = _norm(cls_raw)
    typ = _norm(typ_raw)

    # ZOD 쪽 라벨 예시: class=vehicle / type=car | class=human / type=pedestrian | class=vulnerablevehicle / type=cyclist
    if cls in {"vehicle", "vehicle_object"} and typ in VEHICLE_TO_CAR:
        return "car"
    if cls in {"pedestrian"} and ((not typ) or (typ in HUMAN_TO_PED)):
        return "pedestrian"
    if cls in {"vulnerablevehicle"} and (typ in VULN_TO_CYC):
        return "cyclist"

    return None

def _looks_like_xyxy(bb):
    return isinstance(bb,(list,tuple)) and len(bb)==4 and (bb[2] > bb[0]) and (bb[3] > bb[1])

def _bbox_from_simple(obj):
    props = obj.get("properties", obj)

    # 1) 리스트형 바로 치기
    for key in ("bbox","bbox2d","bbox_xyxy","bbox_xywh","box","rect","box2d"):
        for container in (props, obj):
            v = container.get(key) # 일을 왜 두번하노; ㄹㅈㄷ긴하다 근데 저런 키 존재 안할건데 이거 왜있는거냐? ㄹㅈㄷ긴해
            if isinstance(v, (list,tuple)) and len(v) == 4:
                return (list(map(float, v)),
                        BoxMode.XYXY_ABS if _looks_like_xyxy(v) else BoxMode.XYWH_ABS) # 아 mode라는게 bbox가 xyxy로 정의되는지 xywh로 정의되는지 모드구나 근데 아쉽게도 zod는 보통 단순한 직사각형이 아닐텐데 그리고 그럴 목적이면 prop를 가져올게 아니라 geometry를 가져와야 하는데; 일단 처음부터 잘못됨

    # 2) 딕트형 바로 치기
    def _from_dict(d):
        if not isinstance(d, dict):
            return None
        kk = {k.lower() for k in d.keys()}
        # 좌상-우하
        if {"xmin","ymin","xmax","ymax"} <= kk:
            return [float(d["xmin"]), float(d["ymin"]), float(d["xmax"]), float(d["ymax"])], BoxMode.XYXY_ABS
        if {"x1","y1","x2","y2"} <= kk:
            return [float(d["x1"]),  float(d["y1"]),  float(d["x2"]),  float(d["y2"])],  BoxMode.XYXY_ABS
        if {"left","top","right","bottom"} <= kk:
            return [float(d["left"]),float(d["top"]), float(d["right"]),float(d["bottom"])], BoxMode.XYXY_ABS
        if {"u0","v0","u1","v1"} <= kk:
            return [float(d["u0"]),  float(d["v0"]),  float(d["u1"]),  float(d["v1"])],  BoxMode.XYXY_ABS
        # 좌상 + 크기
        if {"x","y","w","h"} <= kk:
            return [float(d["x"]),   float(d["y"]),   float(d["w"]),   float(d["h"])],   BoxMode.XYWH_ABS
        # 중심 + 크기
        if {"cx","cy","w","h"} <= kk:
            cx,cy,w,h = float(d["cx"]), float(d["cy"]), float(d["w"]), float(d["h"])
            x1, y1, x2, y2 = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0
            return [x1,y1,x2,y2], BoxMode.XYXY_ABS
        return None

    for key in ("bbox","bbox2d","box2d","box","rect","region","geometry","shape"): # key로 geometry만 유효한데 다른 애들은 뭐냐; 그리고 geometry 얘는 직사각형 xyxy도 아니고 꼭짓점이어서 그냥 쓰면 안 될텐데 사각형 bbox로 만들려면 → x_min, y_min, x_max, y_max 계산해야 하는데
        for container in (props, obj):
            val = container.get(key)
            # dict로 오는 경우
            got = _from_dict(val) if isinstance(val, dict) else None
            if got:
                return got
            # list로 오는 경우 (중첩 형태)
            if isinstance(val, (list,tuple)) and len(val) == 4:
                bb = list(map(float, val))
                return bb, (BoxMode.XYXY_ABS if _looks_like_xyxy(bb) else BoxMode.XYWH_ABS)
            # 한 단계 더 중첩: geometry/region 안의 bbox/box2d 등
            if isinstance(val, dict):
                for k2 in ("bbox","bbox2d","box","rect"):
                    if k2 in val:
                        inner = val[k2]
                        got2 = _from_dict(inner) if isinstance(inner, dict) else None
                        if got2:
                            return got2
                        if isinstance(inner, (list,tuple)) and len(inner) == 4:
                            bb = list(map(float, inner))
                            return bb, (BoxMode.XYXY_ABS if _looks_like_xyxy(bb) else BoxMode.XYWH_ABS)

    # 3) 폴리곤류가 있다면 bounding box로 환원
    for key in ("poly2d","polygon","vertices","points"): # 너는 진짜 뭐냐;
        for container in (props, obj):
            pts = container.get(key)
            if isinstance(pts, (list,tuple)) and len(pts) >= 3:
                xs, ys = [], []
                for p in pts:
                    if isinstance(p, (list,tuple)) and len(p) >= 2:
                        xs.append(float(p[0])); ys.append(float(p[1]))
                    elif isinstance(p, dict) and ("x" in p and "y" in p):
                        xs.append(float(p["x"])); ys.append(float(p["y"]))
                if xs and ys:
                    x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
                    if (x2>x1) and (y2>y1):
                        return [x1,y1,x2,y2], BoxMode.XYXY_ABS

    return None, None

def _bbox_valid(bbox, mode):
    if bbox is None or mode is None: return False
    if mode == BoxMode.XYWH_ABS:
        x,y,w,h = bbox; return (w>0) and (h>0)
    if mode == BoxMode.XYXY_ABS:
        x1,y1,x2,y2 = bbox; return (x2-x1>0) and (y2-y1>0)
    return False

def extract_dataset_from_annotation(ann_files: List[str]) -> List[Dict[str, Any]]:
    from collections import Counter
    dataset = []
    num_per_class = Counter(); dropped = Counter()
    num_valid = num_invalid = 0
    img_id = 0

    print(f"[ZOD DEBUG] annotation files = {len(ann_files)}")
    for ann_path in ann_files:
        # …/single_frames/<id>/annotations/object_detection.json → <id> 디렉토리
        frame_dir = os.path.dirname(os.path.dirname(ann_path))
        img_dir = os.path.join(frame_dir, "camera_front_blur")
        hits = sorted(sum([glob.glob(os.path.join(img_dir, ext)) 
                        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")], []))
        if not hits:
            print(f"[ZOD WARN] no image under {img_dir}"); 
            continue
        file_name = hits[0]
        W = H = None  # (원하면 Pillow로 im.size 읽어도 됨)

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                root = json.load(f)
        except Exception as e:
            print(f"[ZOD WARN] json load failed: {ann_path}: {e}")
            continue

        # 당신 파일은 "리스트(객체들)" 라고 가정 (업로드 파일 기준)
        objs = root if isinstance(root, list) else root.get("annotations", [])

        record = dict(file_name=file_name, image_id=img_id, annotations=[])
        if H is not None and W is not None:
            record['height'] = H
            record['width'] = W
        img_id += 1

        for obj in objs: # json 파일 열어서 {}를 하나씩 꺼내봄. 하나의 scope 안에 geometry와 properties로 또 나누어짐.
            props = obj.get("properties", obj) # 주로 알고 싶은 정보는 properties에 존재
            cls_raw = props.get("class") or props.get("category") or "" # 보통 class에 대분류가 있음
            typ_raw = props.get("type")  or props.get("subclass")  or "" # 보통 type에 소분류가 있음

            std = _to_std3(cls_raw, typ_raw) # ⭐ only dynamic object
            if std is None: # car, ped, cyc 빼고 다 드롭
                dropped[f"class={_norm(cls_raw)}|type={_norm(typ_raw)}"] += 1
                continue

            bbox, mode = _bbox_from_simple_with_geom(obj) # 야 이거 뭐냐; bbox필요없다고 2d는; 근데 여기서 왜 geo랑 prop 둘다 봄? bbox는 geo만 봐라
            if not _bbox_valid(bbox, mode):
                num_invalid += 1
                continue
            num_valid += 1

            record["annotations"].append({
                "bbox": bbox, # bbox 없는데 뭔 개솔? 3dbox형태밖에 없는데 얘는 뭘 추출하고 있는거임?;
                "bbox_mode": mode,
                "category_id": CAT_TO_ID[std],
                "iscrowd": int(props.get("iscrowd", 0)), # 이거 뭔 개솔?
            })
            
            bbox3d, yaw_sincos = _extract_3d_extras(obj) # 이거지 3D 박스를 뽑아야지; 이거지 이거지 여기서 x, y, z, w, l, h랑 yaw 뽑네 (아마 중심은 차량 전체 부피의 한가운데 일거임)
            if bbox3d is not None and yaw_sincos is not None:
                record['annotations'][-1]['bbox3d'] = bbox3d
                record['annotations'][-1]['yaw_sincos'] = yaw_sincos
            num_per_class[std] += 1 # class별로 몇개 뽑혔는지 카운트

        if record["annotations"]:
            dataset.append(record)

    print(f"[ZOD DEBUG] valid object = {num_valid}")
    print(f"[ZOD DEBUG] invalid object due to wrong annotation GT data = {num_invalid}")
    print(f"[ZOD DEBUG] number per class = {dict(num_per_class)}")
    if dropped:
        print(f"[ZOD DEBUG] dropped top5 = {dropped.most_common(5)}")
    print(f"[ZOD DEBUG] built records = {len(dataset)}")
    return dataset

def register_zod(zod_root: str): # Detectron2의 DatasetCatalog에 등록함
    assert os.path.isdir(zod_root), f"Invalid zod_root: {zod_root}"
    annotation_glob = os.path.join(zod_root, "single_frames", "*", "annotations", "object_detection.json")
    annotation_files = sorted(glob.glob(annotation_glob))
    print(f"[ZOD DEBUG] found ann files: {len(annotation_files)} by {annotation_glob}")
    if not annotation_files:
        raise FileNotFoundError(f"No annotation json found by: {annotation_glob}")

    import random
    # Split ann_files into train and val (80/20)
    annotation_files = annotation_files[:]  # shallow copy
    random.seed(42)  # for reproducibility
    random.shuffle(annotation_files)

    split_idx = int(len(annotation_files) * 0.8)
    annotation_files_for_training = annotation_files[:split_idx]
    annotation_files_for_val   = annotation_files[split_idx:]

    print(f"[ZOD DEBUG] train/val split = {len(annotation_files_for_training)}/{len(annotation_files_for_val)}")

    def regist_dataset(name, loader):
        if name in DatasetCatalog.list():
            print(f"[ZOD] skip already-registered: {name}"); return
        DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(thing_classes=THING_CLASSES)
        print(f"[ZOD] Registered {name}")

    regist_dataset("zod_train", lambda: extract_dataset_from_annotation(annotation_files_for_training))
    regist_dataset("zod_val",   lambda: extract_dataset_from_annotation(annotation_files_for_val))