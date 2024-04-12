# https://github.com/SFML/cmake-sfml-project/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(ltct LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_SILENT OFF)

# libtorch - get the libtorch shared binaries (THIS IS FOR LINUX - CPU)
# To use GPU and Support for Windows or Mac replace the link below with the other
# links found here https://pytorch.org/get-started/locally/ for CUDA GPUs
message("--------------------------")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("Operating System Type: LINUX")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcpu.zip
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("Operating System Type: APPLE")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.2.zip
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("Operating System Type: WINDOWS")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.2%2Bcpu.zip
  )
else()
  message("Operating System Type: Unkown")
  message(FATAL_ERROR "Cannot be supported")
endif()
message("--------------------------")

# libtorch - make the libtorch library available to the project
FetchContent_MakeAvailable(Torch)

# libtorch - add the lib paths of the downloaded library from FetchContent
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")

# libtorch - find the libtorch lib and require it for the project
find_package(Torch REQUIRED)

# libtorch - link the libtorch libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(CMakePrintHelpers)
message("=====================================================")
cmake_print_variables(torch_SOURCE_DIR)
cmake_print_variables(TORCH_CXX_FLAGS)
cmake_print_variables(TORCH_LIBRARIES)
cmake_print_variables(TORCH_INCLUDE_DIRS)
cmake_print_variables(CMAKE_CXX_FLAGS)
message("=====================================================")

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} main.cpp ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)