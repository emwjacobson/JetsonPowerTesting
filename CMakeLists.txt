project(JetsonPowerTesting LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)

add_executable(power_monitor main.cpp)
target_link_libraries(power_monitor pthread)

add_executable(constant_load constant_load.cu)
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
set_target_properties(constant_load PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(constant_load PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}
)
target_link_libraries(constant_load
    ${CUBLAS_LIBRARY}
)
