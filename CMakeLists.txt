cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(libmongocxx REQUIRED)
# find_package(libbsoncxx REQUIRED)

add_executable(main main.cpp)

# include_directories(${LIBMONGOCXX_INCLUDE_DIRS})
# include_directories(${LIBBSONCXX_INCLUDE_DIRS})

target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main hiredis)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
# target_link_libraries(main ${LIBMONGOCXX_LIBRARIES})
# target_link_libraries(main ${LIBBSONCXX_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 14)
