cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(Tf_worker)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_executable(main main.cpp)

# hiredis dependency
find_path(HIREDIS_HEADER hiredis)
find_library(HIREDIS_LIB hiredis)

target_link_libraries(main ${HIREDIS_LIB})

find_package(Protobuf)
target_link_libraries(main ${Protobuf_LIBRARIES})

# Tensorflow library.
find_package(TensorflowCC REQUIRED)
target_link_libraries(main TensorflowCC::TensorflowCC)

# You may also link cuda if it is available.
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(main ${CUDA_LIBRARIES})
endif()
