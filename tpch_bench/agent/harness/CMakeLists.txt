# SPDX-License-Identifier: Apache-2.0
# Minimal project to compile the agent-generated query against the installed libcudf.
cmake_minimum_required(VERSION 3.26)
project(agent_query LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(cudf REQUIRED)

add_executable(agent_query main.cpp run_query.cpp)
set_target_properties(agent_query PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20)
target_link_libraries(agent_query PRIVATE cudf::cudf)
