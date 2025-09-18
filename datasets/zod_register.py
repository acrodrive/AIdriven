import os, glob, json
from typing import List, Dict, Any, Tuple, Optional
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# --- Added helper: robust bbox extraction from geometry.coordinates ---
def _bbox_from_geom_coordinates(obj):
    try:
        geom = obj.get("geometry", {})
        if not isinstance(geom, dict):
            return None, None
        gtype = str(geom.get("type", "")).lower()
        coords = geom.get("coordinates")
        if gtype not in ("multipoint", "polygon") or not isinstance(coords, (list, tuple)):
            return None, None

        # Handle Polygon nesting: [[[x,y], ...]] -> take outer ring
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
    # Try original function if present
    try:
        bbox_mode = globals().get("_bbox_from_simple", None)
        if callable(bbox_mode):
            bbox, mode = bbox_mode(obj)
            # If it returned a valid bbox, keep it
            if bbox and mode is not None:
                return bbox, mode
    except Exception:
        pass
    # Fallback to geometry.coordinates-based bbox
    return _bbox_from_geom_coordinates(obj)
# --- end added helper ---


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

VEHICLE_TO_CAR = {"car","bus","truck","van","trailer","pickup","suv","jeep","minivan"}
HUMAN_TO_PED   = {"pedestrian","person","people","human"}
VULN_TO_CYC    = {"cyclist","bicyclist","bicycle","rider","motorcycle","motorcyclist","kick_scooter","personal_mobility"}

def _to_std3(cls_raw: str, typ_raw: str) -> Optional[str]:
    cls = _norm(cls_raw)
    typ = _norm(typ_raw)

    # ZOD 쪽 라벨 예시: class=vehicle / type=car | class=human / type=pedestrian | class=vulnerablevehicle / type=cyclist
    if cls in {"vehicle", "vehicle_object"} and typ in VEHICLE_TO_CAR:
        return "car"
    if cls in {"human", "pedestrian"} and ((not typ) or (typ in HUMAN_TO_PED)):
        return "pedestrian"
    if cls in {"vulnerablevehicle", "vulnerable_vehicle"} and (typ in VULN_TO_CYC):
        return "cyclist"

    # 보조 판단 (type만으로)
    if typ in VEHICLE_TO_CAR: return "car"
    if typ in HUMAN_TO_PED:   return "pedestrian"
    if typ in VULN_TO_CYC:    return "cyclist"

    # 마지막 보조 (class만으로)
    if cls in VEHICLE_TO_CAR: return "car"
    if cls in HUMAN_TO_PED:   return "pedestrian"
    if cls in VULN_TO_CYC:    return "cyclist"

    return None

def _looks_like_xyxy(bb):
    return isinstance(bb,(list,tuple)) and len(bb)==4 and (bb[2] > bb[0]) and (bb[3] > bb[1])

def _bbox_from_simple(obj):
    """
    당신 파일 한 타입만 잘 먹게 하되, 가장 흔한 변형들을 모두 흡수:
    - 리스트형: bbox=[x1,y1,x2,y2] 또는 [x,y,w,h]
    - 딕트형:  {xmin,ymin,xmax,ymax} / {x1,y1,x2,y2} / {left,top,right,bottom}
              {u0,v0,u1,v1} / {x,y,w,h} / {cx,cy,w,h}
    - 위치: properties 또는 최상위, 그리고 box2d/bbox/box/rect/geometry/region/shape 아래
    """
    props = obj.get("properties", obj)

    # 1) 리스트형 바로 치기
    for key in ("bbox","bbox2d","bbox_xyxy","bbox_xywh","box","rect","box2d"):
        for container in (props, obj):
            v = container.get(key)
            if isinstance(v, (list,tuple)) and len(v) == 4:
                return (list(map(float, v)),
                        BoxMode.XYXY_ABS if _looks_like_xyxy(v) else BoxMode.XYWH_ABS)

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

    for key in ("bbox","bbox2d","box2d","box","rect","region","geometry","shape"):
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
    for key in ("poly2d","polygon","vertices","points"):
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

def load_zod_simple(ann_files: List[str]) -> List[Dict[str, Any]]:
    from collections import Counter
    dataset = []
    kept = Counter(); dropped = Counter()
    good = bad = 0
    img_id = 0

    print(f"[ZOD DEBUG] ann files = {len(ann_files)}")
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

        for obj in objs:
            # class/type 는 최상위 또는 properties 밑에 존재 (둘 다 시도)
            props = obj.get("properties", obj)
            cls_raw = props.get("class") or props.get("category") or ""
            typ_raw = props.get("type")  or props.get("subclass")  or ""

            std = _to_std3(cls_raw, typ_raw)
            if std is None:
                # car/ped/cyc 외는 드롭
                dropped[f"class={_norm(cls_raw)}|type={_norm(typ_raw)}"] += 1
                continue

            bbox, mode = _bbox_from_simple_with_geom(obj)
            if not _bbox_valid(bbox, mode):
                bad += 1
                continue
            good += 1

            record["annotations"].append({
                "bbox": bbox,
                "bbox_mode": mode,
                "category_id": CAT_TO_ID[std],
                "iscrowd": int(props.get("iscrowd", 0)),
            })
            kept[std] += 1

        if record["annotations"]:
            dataset.append(record)

    print(f"[ZOD DEBUG] bbox good/bad = {good}/{bad}")
    print(f"[ZOD DEBUG] kept per-class = {dict(kept)}")
    if dropped:
        print(f"[ZOD DEBUG] dropped top5 = {dropped.most_common(5)}")
    print(f"[ZOD DEBUG] built records = {len(dataset)}")
    return dataset

def register_all_zod(zod_root: str):
    assert os.path.isdir(zod_root), f"Invalid zod_root: {zod_root}"
    ann_glob = os.path.join(zod_root, "single_frames", "*", "annotations", "object_detection.json")
    ann_files = sorted(glob.glob(ann_glob))
    print(f"[ZOD DEBUG] found ann files: {len(ann_files)} by {ann_glob}")
    if not ann_files:
        raise FileNotFoundError(f"No annotation json found by: {ann_glob}")

    def _safe(name, loader):
        if name in DatasetCatalog.list():
            print(f"[ZOD] skip already-registered: {name}"); return
        DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(thing_classes=THING_CLASSES)
        print(f"[ZOD] Registered {name}")

    _safe("zod_3class_train", lambda: load_zod_simple(ann_files))
    _safe("zod_3class_val",   lambda: load_zod_simple(ann_files))
