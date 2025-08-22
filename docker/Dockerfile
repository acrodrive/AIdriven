# ---- Base --------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# ---- Arguments ---------------------------------------------------------------
ARG USER_ID=1000
ARG TORCH_CUDA_ARCH_LIST="8.6"

# ---- Environment -------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

# ---- System packages ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-distutils \
        git build-essential cmake ninja-build \
        libjpeg-dev libpng-dev libgl1 libglib2.0-0 python3-opencv ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- PyPI packages -----------------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118

# ---- Detectron2 dependencies -------------------------------------------------
RUN python -m pip install --no-cache-dir \
    fvcore \
    iopath \
    yacs \
    termcolor \
    tqdm \
    tabulate \
    matplotlib \
    pycocotools

# ---- Non-root user (선택) ----------------------------------------------------
RUN useradd -m --uid ${USER_ID} appuser
USER appuser
WORKDIR /home/appuser

# ---- Detectron2 --------------------------------------------------------------
RUN git clone --depth 1 https://github.com/acrodrive/AIdriven.git
WORKDIR /home/appuser/AIdriven
RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git
RUN cp -r /home/appuser/AIdriven/detectron2/demo /home/appuser/AIdriven/Cruise/inference
COPY faster_rcnn_R_50_FPN_3x.pkl /home/appuser/AIdriven/Cruise/pretrained
COPY rsds.py /home/appuser/AIdriven/Cruise/inference
WORKDIR /home/appuser/AIdriven/detectron2
RUN python -m pip install -e .
WORKDIR /home/appuser/AIdriven
# faster_rcnn_R_50_FPN_3x.pkl와 inference.py는 dockerfile과 같은 폴더에 준비할 것(뭐 굳이 없어도 됨)


# ---- Sanity check ------------------------------------------------------------
CMD ["/bin/bash"]


# 도커 이미지 생성 및 컨테이너 생성
# docker build -t aidriven:v0 .
# docker run -it --gpus all --name cruise aidriven:v0

# 컨테이너를 한 번 생성하고 나면 그 뒤로는 컨테이너를 또 생성할 필요 없이 껐다/켰다 하기
# docker start cruise (백그라운드에서 돔, 백그라운드에서 돌리다가 필요에 따라 터미널에 접근할 수는 있음)
# docker start -ai cruise (터미널로 들어감)
# docker stop cruise

# 주의사항
# demo.py의 17번 째줄의 from vision.fair.detectron2.demo.predictor import VisualizationDemo는
# 페이스북 내부 경로에서 네임스페이스를 사용하던 흔적이고
# 원래는 아래와 같이 가져와야 한다.
# from predictor import VisualizationDemo

# 실행 방법
# python3 demo/demo.py \
#   --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
#   --input demo/test_images/* \
#   --output demo/test_output \
#   --confidence-threshold 0.5 \
#   --opts MODEL.WEIGHTS demo/faster_rcnn_R_50_FPN_3x.pkl

# 데이터 다운로드 방법
# zod download -y \
#   --url "https://www.dropbox.com/scl/fo/q81qqpiqygaeys7mppgoe/AFuqa-QrSkGzHmnkhhpvbBE?rlkey=ocr9n0gq3u083zj8sn1yo1ak6&e=1&dl=0" \
#   --subset frames \
#   --version full \
#   --output-dir /home/appuser/zoddata \
#   --no-lidar --no-oxts --no-vehicle-data