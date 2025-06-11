import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D, Point2D
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection_node')

        self.bridge = CvBridge()
        self.model_type = self.declare_parameter('model_type', 'pt').get_parameter_value().string_value  # 'pt' or 'onnx'
        self.model = self.load_yolo_model()

        # Publisher and subscriber
        self.detection_pub = self.create_publisher(Detection2DArray, '/cone_detections', 10)
        self.masked_image_pub = self.create_publisher(Image, '/cone_detection/image_with_masks', 10)
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left_raw/image_raw_color', self.image_callback, 10)

    def load_yolo_model(self):
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        if self.model_type == 'onnx':
            model_path = os.path.join(model_dir, 'yolov8m.onnx')
            self.get_logger().info(f"Using ONNX model: {model_path}")
            return cv2.dnn.readNetFromONNX(model_path)
        else:
            from ultralytics import YOLO
            model_path = os.path.join(model_dir, 'best.pt')
            self.get_logger().info(f"Using PyTorch model: {model_path}")
            return YOLO(model_path)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        start = time.time()
        detections, image_with_masks = self.run_inference(cv_image)
        end = time.time()
        self.get_logger().info(f"Inference time: {end - start:.3f}s")

        # Publish detections
        self.detection_pub.publish(detections)

        # Publish annotated image
        mask_msg = self.bridge.cv2_to_imgmsg(image_with_masks, encoding='bgr8')
        mask_msg.header = msg.header
        self.masked_image_pub.publish(mask_msg)

    def run_inference(self, image):
        h, w = image.shape[:2]
        detections = Detection2DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera_frame'

        if self.model_type == 'onnx':
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.model.setInput(blob)
            outputs = self.model.forward()

            for output in outputs[0]:
                confidence = float(output[4])
                if confidence < 0.5:
                    continue

                class_id = int(np.argmax(output[5:]))
                score = float(output[5 + class_id])
                cx, cy, bw, bh = output[0:4]

                cx *= w
                cy *= h
                bw *= w
                bh *= h

                self._append_detection(detections, class_id, score, cx, cy, bw, bh, image)

        else:  # 'pt'
            results = self.model(image)[0]
            for det in results.boxes:
                class_id = int(det.cls[0])
                score = float(det.conf[0])
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1

                self._append_detection(detections, class_id, score, cx, cy, bw, bh, image)

        return detections, image

    def _append_detection(self, detections, class_id, score, cx, cy, bw, bh, image):
        bbox = BoundingBox2D()
        bbox.center.position = Point2D(x=float(cx), y=float(cy))
        bbox.center.theta = 0.0
        bbox.size_x = float(bw)
        bbox.size_y = float(bh)

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(class_id)
        hypothesis.hypothesis.score = score

        det = Detection2D()
        det.bbox = bbox
        det.results.append(hypothesis)
        detections.detections.append(det)

        # Draw bounding box
        x1 = int(cx - bw/2)
        y1 = int(cy - bh/2)
        x2 = int(cx + bw/2)
        y2 = int(cy + bh/2)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"Class {class_id}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
