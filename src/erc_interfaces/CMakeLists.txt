cmake_minimum_required(VERSION 3.8)
project(erc_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/NavigateToColumn.srv"
  "srv/NavigateToBin.srv"
  "srv/GraspBook.srv"
  "srv/PlaceInBin.srv"
)

ament_package()
