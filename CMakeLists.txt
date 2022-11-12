cmake_minimum_required(VERSION 3.20)

project(mandelbrot-greyscale cxx
    VERSION 0.1
    DESCRIPTION "Generate 8-bit greyscale mandelbrot sets")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mandelbrot mandelbrot.cpp)
target_link_libraries(mandelbrot ${OpenCV_LIBS})

