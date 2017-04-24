cmake_minimum_required(VERSION 2.8.3)
project(kalman_sense)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

find_package(Eigen3 REQUIRED )

catkin_package()

set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

message( STATUS "*******************************************")
message( STATUS "Eigen include:  ${EIGEN3_INCLUDE_DIR}")
message( STATUS "*******************************************")

add_executable(node src/node.cpp
                    src/QuadUkf.cpp
                    src/UnscentedKf.cpp
)

target_link_libraries( node
   ${catkin_LIBRARIES}
)
