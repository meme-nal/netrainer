cmake_minimum_required(VERSION 3.31.3)
project(netrainer)

find_package(Torch REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/*.cpp"
  "${CMAKE_SOURCE_DIR}/*.h"
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

set(DEBUG_OPTIONS
  -g
  -Wall
  -Wextra
  -D_GLIBCXX_ASSERTIONS)

set(RELEASE_OPTIONS
  -O2
  -march=native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_MODE)
  target_compile_options(${PROJECT_NAME} PRIVATE ${DEBUG_OPTIONS})
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE_MODE)
  target_compile_options(${PROJECT_NAME} PRIVATE ${RELEASE_OPTIONS})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
CXX_STANDARD 20
CXX_STANDARD_REQUIRED ON)

target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries (${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)