import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D, Pose2D, Point2D
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection_node')

        # Initialize bridge and YOLO model (load your model here)
        self.bridge = CvBridge()
        self.model = self.load_yolo_model()

        # Publisher for detections and mask images
        self.detection_pub = self.create_publisher(Detection2DArray, '/cone_detections', 10)
        self.masked_image_pub = self.create_publisher(Image, '/cone_detection/image_with_masks', 10)

        # Subscriber to image input
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
    
    def load_yolo_model(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8m.onnx')
        model_path = os.path.abspath(model_path)
        return cv2.dnn.readNetFromONNX(model_path)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections, image_with_masks = self.run_inference(cv_image)

        # Publish detections
        self.detection_pub.publish(detections)

        # Publish annotated image with masks
        mask_msg = self.bridge.cv2_to_imgmsg(image_with_masks, encoding='bgr8')
        mask_msg.header = msg.header
        self.masked_image_pub.publish(mask_msg)

    def run_inference(self, image):
        # Resize, normalize and prepare image for inference
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward()

        h, w = image.shape[:2]
        detections = Detection2DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera_frame'

        for output in outputs[0]:
            confidence = float(output[4])
            if confidence < 0.5:
                continue

            class_id = int(np.argmax(output[5:]))
            score = float(output[5 + class_id])

            cx, cy, bw, bh = output[0:4]  # center x, center y, box width/height (normalized)
            cx *= w
            cy *= h
            bw *= w
            bh *= h

            bbox = BoundingBox2D()
            bbox.center.position = Point2D(x=cx, y=cy)
            bbox.center.theta = 0.0
            bbox.size_x = bw
            bbox.size_y = bh

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(class_id)
            hypothesis.hypothesis.score = score

            det = Detection2D()
            det.bbox = bbox
            det.results.append(hypothesis)
            detections.detections.append(det)

            # Draw bounding box and class label on image (for mask image)
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"Class {class_id}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return detections, image

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
