cmake_minimum_required(VERSION 3.5)
project(cone_detection)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(OpenCV REQUIRED)

#############
## Include ##
#############

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(DIRECTORY cone_detection/
  DESTINATION share/${PROJECT_NAME}/cone_detection
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

ament_package()