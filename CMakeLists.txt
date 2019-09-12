project(CCL C CXX)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wpedantic -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG")

set(WERROR ON CACHE BOOL "Option to disable -Werror flag.")
set(PGO_DIR "${PROJECT_BINARY_DIR}/profiling"
  CACHE STRING "Directory where PGO data is placed.")

if (WERROR)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

find_package(Boost "1.55.0" COMPONENTS filesystem program_options REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem regex)
find_package(Ceres REQUIRED)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_definitions(
  ${OpenCV_DEFINITIONS}
  ${Boost_DEFINITIONS}
  ${CERES_DEFINITIONS})
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS})
link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS})
link_libraries(
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES})
