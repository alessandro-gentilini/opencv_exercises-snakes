cmake_minimum_required(VERSION 2.8)
PROJECT( snake )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( snake snake.cpp )
TARGET_LINK_LIBRARIES( snake ${OpenCV_LIBS} )
SET(CMAKE_CXX_FLAGS "-std=c++0x")

