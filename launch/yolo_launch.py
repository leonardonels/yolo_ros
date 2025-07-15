from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os

def generate_launch_description():

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    cone_detection = os.path.join(pkg_dir, '../src/cone_detection_node.py')

    cone_detectio_node = ExecuteProcess(
        cmd=['python3', cone_detection],
        output='screen'
    )

    return LaunchDescription([
        cone_detectio_node,
    ])
