from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("../models/best.pt")

# Export the model to ONNX format
model.export(format="onnx")

# Load the exported ONNX model
onnx_model = YOLO("../models/yolov8m.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")