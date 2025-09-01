import os, json, glob
from typing import List, Dict
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from zod import ZodFrames  # frames subset 사용 가정

ZOD_CATS = ["car", "pedestrian", "cyclist"]  # lowercase로 통일
CAT_TO_ID = {k: i for i, k in enumerate(ZOD_CATS)}

def _objects_to_detectron(objects):
    annos = []
    # print(f"Processing {len(objects)} objects")  # Debug print
    
    for obj in objects:
        # Debug print object structure
        # print(f"Object attributes: {dir(obj)}")
        
        cls_name = getattr(obj, "category", None) or getattr(obj, "label", None)
        if cls_name not in CAT_TO_ID:
            # print(f"Skipping unknown category: {cls_name}")  # Debug print
            continue

        # Get 2D bbox
        box2d = None
        if hasattr(obj, "box2d"):
            box2d = obj.box2d
        elif hasattr(obj, "bbox"):
            box2d = obj.bbox
        
        if not box2d:
            continue
            
        x1, y1, x2, y2 = box2d

        # Get 3D box
        box3d = getattr(obj, "box3d", None)
        if not box3d:
            continue
            
        # Extract 3D parameters
        center = getattr(box3d, "center", [0, 0, 0])
        lwh = getattr(box3d, "lwh", [0, 0, 0])
        yaw = getattr(box3d, "yaw", 0.0)

        annos.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": CAT_TO_ID[cls_name.lower()],  # Convert to lowercase
            "box3d": {
                "center": [float(c) for c in center],
                "lwh": [float(d) for d in lwh],
                "yaw": float(yaw),
            }
        })
    return annos

def _iter_frames(zf, split):
    """
    Make ZodFrames iterable across versions (0.3.x~).
    Tries several APIs in order and falls back to internal indices.
    """
    # 1) 옛 API: get_all()
    if hasattr(zf, "get_all"):
        yield from zf.get_all()
        return

    # 2) split별 ID를 얻는 API가 있는 경우
    if hasattr(zf, "get_split"):
        def ids_for(s):
            try:
                return list(zf.get_split(s))
            except Exception:
                return []
        if split in ("train", "val", "test"):
            ids = ids_for(split)
        else:  # 'trainval' 등
            ids = ids_for("train") + ids_for("val")
        for fid in ids:
            if hasattr(zf, "get_frame"):
                yield zf.get_frame(fid)
            elif hasattr(zf, "__getitem__"):
                yield zf[fid]
        return

    # 3) 내부 인덱스가 노출되는 구버전(0.3.x)들: _train_ids/_val_ids
    has_tv = hasattr(zf, "_train_ids") or hasattr(zf, "_val_ids")
    if has_tv:
        if split == "train":
            ids = list(getattr(zf, "_train_ids", []))
        elif split == "val":
            ids = list(getattr(zf, "_val_ids", []))
        else:  # 'trainval' 등
            ids = list(getattr(zf, "_train_ids", [])) + list(getattr(zf, "_val_ids", []))
        for fid in ids:
            if hasattr(zf, "get_frame"):
                yield zf.get_frame(fid)
            elif hasattr(zf, "__getitem__"):
                yield zf[fid]
        return

    # 4) 최후 수단: _infos 키 순회
    if hasattr(zf, "_infos"):
        for fid in sorted(zf._infos.keys()):
            if hasattr(zf, "get_frame"):
                yield zf.get_frame(fid)
            elif hasattr(zf, "__getitem__"):
                yield zf[fid]
        return

    raise AttributeError("Unsupported ZodFrames API for this version")

def _fid_list(zf, split: str):
    ids = []
    if hasattr(zf, "get_split"):
        def ids_for(s):
            try: return list(zf.get_split(s))
            except Exception: return []
        if split in ("train","val","test"):
            ids = ids_for(split)
        else:  # 'trainval'
            ids = ids_for("train") + ids_for("val")
    if not ids and (hasattr(zf, "_train_ids") or hasattr(zf, "_val_ids")):
        if split == "train": ids = list(getattr(zf, "_train_ids", []))
        elif split == "val": ids = list(getattr(zf, "_val_ids", []))
        else: ids = list(getattr(zf, "_train_ids", [])) + list(getattr(zf, "_val_ids", []))
    if not ids and hasattr(zf, "_infos"):
        ids = sorted(zf._infos.keys())
    def _key(x):
        try: return int(x)
        except: return str(x)
    return sorted(ids, key=_key)

def _find_front_image(frame_dir: str) -> str:
    """
    frame_dir = .../single_frames/<fid>
    front 카메라 후보 폴더를 우선순위로 탐색하고, 그 안에서 jpg/png 파일 1장을 선택.
    """
    cand_dirs = [
        "camera_front_blur",  # 사용자가 보여준 폴더
        "camera_front",       # 블러 없는 경우
    ]
    for d in cand_dirs:
        img_dir = os.path.join(frame_dir, d)
        if not os.path.isdir(img_dir):
            continue
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            imgs.extend(glob.glob(os.path.join(img_dir, ext)))
        if imgs:
            imgs.sort()
            return imgs[0]  # 가장 앞의 한 장
    return ""

def _load_2d_boxes_if_any(ann_dir: str) -> List[Dict]:
    """
    ZOD 버전별/내부 구성에 따라 파일명이 다를 수 있으므로,
    흔히 쓰이는 이름 후보를 순차적으로 시도.
    없으면 빈 리스트 반환(=annotation 없음).
    """
    if not os.path.isdir(ann_dir):
        return []
    # 후보 파일명(필요시 여기에 실제 파일명 추가)
    candidates = [
        "2d_boxes.json",
        "boxes_2d.json",
        "annotations_2d.json",
        "instances_2d.json",
        "coco_instances.json",  # coco 포맷일 수 있음
    ]
    path = ""
    for fn in candidates:
        p = os.path.join(ann_dir, fn)
        if os.path.exists(p):
            path = p
            break
    if not path:
        # 주석: 폴더 안에 뭐가 있는지 탐색해서 알려주고 빈 리스트 반환
        # print("[WARN] No known 2D box json in:", ann_dir, "files:", os.listdir(ann_dir))
        return []

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    # ① per-frame 단일 파일 구조 예시(프레임 안에 boxes 배열이 있는 경우)
    if isinstance(data, dict) and "boxes" in data:
        anns = []
        for obj in data.get("boxes", []):
            # 포맷이 [x,y,w,h] 또는 [x0,y0,x1,y1]일 수 있음 → 최대한 방어적으로 처리
            bbox = obj.get("bbox") or obj.get("box") or obj.get("bbox_xywh") or obj.get("bbox_xyxy")
            if not bbox:
                continue
            if len(bbox) == 4:
                # xywh인지 xyxy인지 모르면, 일단 xyxy로 가정하고 틀리면 나중에 교정
                x0, y0, x1, y1 = bbox
            else:
                continue
            cat = obj.get("category_id", 0)
            anns.append({
                "bbox": [x0, y0, x1, y1],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(cat),
            })
        return anns

    # ② coco 형식일 경우(프레임 폴더별로 coco가 들어있는 케이스는 드뭅니다만 방어)
    if isinstance(data, dict) and "annotations" in data and "images" in data and "categories" in data:
        # coco의 image_id와 현재 frame의 매칭이 필요 → per-frame coco라면 annotation 전부 사용
        anns = []
        for a in data["annotations"]:
            if "bbox" not in a:
                continue
            x, y, w, h = a["bbox"]
            anns.append({
                "bbox": [x, y, x+w, y+h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(a.get("category_id", 0)),
            })
        return anns

    # 그 외 구조는 미지원 → 빈 리스트
    return []

def load_zod_split(zod_root: str, split: str, version: str):
    from zod import ZodFrames
    import json
    
    zf = ZodFrames(dataset_root=zod_root, version=version)
    fids = _fid_list(zf, split)
    dataset_dicts = []
    
    print(f"Processing {len(fids)} frames...")
    valid = 0
    
    for fid in fids:
        frame_dir = os.path.join(zod_root, "single_frames", str(fid))
        img_path = _find_front_image(frame_dir)
        if not img_path:
            continue
            
        # Read object detection annotations
        ann_file = os.path.join(frame_dir, "annotations", "object_detection.json")
        if not os.path.exists(ann_file):
            continue
            
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
            
        annos = []
        for obj in annotations:
            if "properties" not in obj or "geometry" not in obj:
                continue
                
            props = obj["properties"]
            geom = obj["geometry"]
            
            # Skip if not one of our target classes
            obj_class = props.get("class", "").lower()
            if obj_class not in CAT_TO_ID:
                continue
                
            # Get 2D bbox from geometry
            if geom["type"] != "MultiPoint" or len(geom["coordinates"]) != 4:
                continue
                
            # Convert quadrilateral to bbox
            points = geom["coordinates"]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Get 3D information
            loc_3d = props.get("location_3d", {}).get("coordinates", [0,0,0])
            size_3d = [
                props.get("size_3d_length", 0),
                props.get("size_3d_width", 0),
                props.get("size_3d_height", 0)
            ]
            yaw = props.get("orientation_3d_qz", 0)
            
            annos.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": CAT_TO_ID[obj_class],
                "box3d": {
                    "center": [float(x) for x in loc_3d],
                    "lwh": [float(x) for x in size_3d],
                    "yaw": float(yaw)
                }
            })
            
        if not annos:
            continue
            
        # Get image size
        with Image.open(img_path) as img:
            width, height = img.size
            
        valid += 1
        dataset_dicts.append({
            "file_name": img_path,
            "image_id": fid,
            "height": height,
            "width": width,
            "annotations": annos,
        })
    
    print(f"Found {valid} valid samples")
    return dataset_dicts

def register_zod(name: str, zod_root: str, split: str, version: str = "full"):
    DatasetCatalog.register(name, lambda: load_zod_split(zod_root, split, version))
    # 실제 클래스 이름으로 바꾸세요
    MetadataCatalog.get(name).set(thing_classes=["car","pedestrian","cyclist"])
    
def register_all_zod(zod_root="/home/appuser/AIdriven/datasets/zod", version=None):
    """
    프로젝트나 환경변수에서 버전을 지정할 수 있게 함.
    """
    ver = version or os.getenv("ZOD_VERSION", "full")  # 기본 mini
    register_zod("zod3d_train", zod_root, "train", ver)
    register_zod("zod3d_val",   zod_root, "val",   ver)
