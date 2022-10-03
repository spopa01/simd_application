cmake_minimum_required(VERSION 3.23)
project(gmm)

set(CMAKE_CXX_STANDARD 23)

#set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gmm main.cpp)
target_link_libraries(gmm ${OpenCV_LIBS})