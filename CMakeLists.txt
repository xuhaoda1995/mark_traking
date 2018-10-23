cmake_minimum_required(VERSION 2.8.3)
project(mark_track)

add_compile_options(-std=c++11)

SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

MESSAGE(STATUS ${OpenCV_DIR})
MESSAGE(STATUS ${OpenCV_VERSION})

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

MESSAGE(STATUS ${OpenCV_INCLUDE_DIRS})

# Add the source in project root directory
aux_source_directory(. DIRSRCS)

add_executable(${PROJECT_NAME} ${DIRSRCS})

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
)