FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN apt-get update && apt-get install -y \
    python3-opencv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true
RUN pip3 install --no-deps "ultralytics==8.0.196"
RUN pip3 install PyYAML tqdm matplotlib scipy pillow psutil py-cpuinfo pandas

ENV PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

WORKDIR /app
COPY . .

RUN python3 -c "import cv2; print('cv2 OK:', cv2.__version__)"
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD ["python3", "camera.py", "live", "--headless", "--interval", "10"]
