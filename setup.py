from setuptools import setup
import os
from glob import glob

package_name = 'cone_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonardo Nels',
    maintainer_email='leo.nels.2000@gmail.com',
    description='YOLO for ROS 2',
    license='GPL-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cone_detection_node = cone_detection.cone_detection_node:main',
        ],
    },
)
