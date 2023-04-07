cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main hiredis)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
