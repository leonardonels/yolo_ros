import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection2DArray
from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header
import numpy as np


class BoundingBoxPointCloudFilter(Node):
    def __init__(self):
        super().__init__('bbox_pointcloud_filter')

        self.declare_parameter('cloud_width', 448)
        self.declare_parameter('cloud_height', 256)

        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)

        self.cloud_width = self.get_parameter('cloud_width').get_parameter_value().integer_value
        self.cloud_height = self.get_parameter('cloud_height').get_parameter_value().integer_value

        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value


        self.detections = None
        self.latest_cloud = None

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/cone_detections',
            self.detection_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10
        )

        self.filtered_pub = self.create_publisher(
            PointCloud2,
            '/filtered_pointcloud',
            10
        )

    def detection_callback(self, msg: Detection2DArray):
        self.detections = msg

    def pointcloud_callback(self, msg: PointCloud2):
        if self.detections is None:
            return

        '''cloud_points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))'''
        
        # Create a 2D grid of (cloud_height x cloud_width)
        organized_cloud = np.full((self.cloud_height, self.cloud_width, 3), np.nan, dtype=np.float32)

        # Fill organized cloud
        idx = 0
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False):
            u = idx % self.cloud_width
            v = idx // self.cloud_width
            if v >= self.cloud_height:
                break
            x, y, z = point[0], point[1], point[2]
            if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                organized_cloud[v, u] = [x, y, z]
            idx += 1

        # Collect points inside all bounding boxes
        selected_points = []

        scale_x = self.cloud_width / self.image_width
        scale_y = self.cloud_height / self.image_height

        for detection in self.detections.detections:
            bbox = detection.bbox

            cx = bbox.center.position.x * scale_x
            cy = bbox.center.position.y * scale_y
            bw = bbox.size_x * scale_x
            bh = bbox.size_y * scale_y

            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(self.cloud_width, cx + bw / 2))
            y2 = int(min(self.cloud_height, cy + bh / 2))

            for v in range(y1, y2):
                for u in range(x1, x2):
                    point = organized_cloud[v, u]
                    if not np.isnan(point).any():
                        selected_points.append(tuple(point))


        if not selected_points:
            self.get_logger().warn("No points found in bounding boxes.")
            return

        # Create new PointCloud2 message
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        filtered_cloud = point_cloud2.create_cloud_xyz32(header, selected_points)
        self.filtered_pub.publish(filtered_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxPointCloudFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
