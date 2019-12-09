set(SRCS
  ${CMAKE_CURRENT_LIST_DIR}/multi_camera_calibration.cpp)

add_executable(multi_cam_calib ${SRCS})

set(EX_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/ba_example.cpp
  ${CMAKE_CURRENT_LIST_DIR}/bal_problem.cc
  )
add_executable(ba_example ${EX_SRCS})


set(EX2_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/ba_example2.cpp
  )
add_executable(ba_example2 ${EX2_SRCS})
