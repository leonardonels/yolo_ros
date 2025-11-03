# yolo

Purpose
-------
Detect colored street cones using a YOLO-based model. Supports ONNX and
PyTorch artifacts (see `yolo/models/`). Designed to run on an NVIDIA GPU for
real-time performance.

Key files
- `src/cone_detection_node.py` — main detector node.
- `models/` — contains `best.onnx`, `yolov8m.onnx`, `best.pt`, etc.
- `requirements.txt`, `setup.py` — python dependencies and packaging.
- `launch/yolo_launch.py` — launch file to start the detector with
  parameters (model path, device, topics).

Topics
- Subscribes: camera image topic (check launch args)
- Publishes: bounding boxes (class, confidence, bbox coords), and optionally
  visualization images.

Run (example)
--------------
Ensure python deps are installed (see `requirements.txt`) and run:

```bash
ros2 launch yolo yolo_launch.py
```

Notes
- For best performance use the ONNX model with an accelerated runtime (ONNX
  Runtime with CUDA, TensorRT, or PyTorch on GPU).
- Tune model path and input topics in the launch file.