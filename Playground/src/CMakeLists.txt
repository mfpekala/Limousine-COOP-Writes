cmake_minimum_required(VERSION 3.10.2)

project(
  Playground
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Get debug symbols
set(CMAKE_BUILD_TYPE Debug)

# clangd completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR})
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/*.cpp")
include_directories(include)

add_executable(${PROJECT_NAME} ${SOURCES})
