cmake_minimum_required(VERSION 3.10)
project(planner)

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)
find_package(OsqpEigen REQUIRED)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    include
)

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs)


add_executable(learning_planning src/learning_planning.cpp)

target_link_libraries(learning_planning
  ${OMPL_LIBRARIES}
  ${catkin_LIBRARIES}
  OsqpEigen::OsqpEigen
  "${TORCH_LIBRARIES}")
set_property(TARGET learning_planning PROPERTY CXX_STANDARD 14)


# ---- 键盘遥操作与数据记录脚本 -------------------------------------------
# 注册成可执行文件，使 `rosrun planner teleop_keyboard.py` 在
# devel 与 install 两种空间下都能找到（install 空间必须显式安装）。
catkin_install_python(PROGRAMS
  scripts/teleop_keyboard.py
  scripts/record_trajectory.py
  scripts/plot_trajectory.py
  scripts/offline_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


