cmake_minimum_required(VERSION 3.18)
project(ltct LANGUAGES CXX)

if(WIN32)
    # disable windows asan for now since I don't know how to make it work
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
else()
    # asan seems to be broken in my machine for some reason - segfaulting in some cases even there is no errors
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

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
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("Operating System Type: APPLE")
  if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
      message(STATUS "x86_64 architecture detected")
      FetchContent_Declare(Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.5.0.zip
      )
  elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
      message(STATUS "arm64 architecture detected")
      FetchContent_Declare(Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.0.zip
      )
  else()
      message(FATAL_ERROR "Other architecture detected")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("Operating System Type: WINDOWS")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.0%2Bcpu.zip
  )
else()
  message("Operating System Type: Unknown")
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

# ################# utility binaries : start #################

file(GLOB RAW_DATA_WRITER_SOURCES "raw_data_writer/*.cpp")

add_library(RawDataWriter STATIC ${RAW_DATA_WRITER_SOURCES})
set_property(TARGET RawDataWriter PROPERTY CXX_STANDARD 17)

cmake_print_variables(RawDataWriter)
message("=====================================================")


# ################# utility binaries : end ################# 


function(compile_binary OUT_EXE SRC_LIST)
  add_executable(${OUT_EXE} ${SRC_LIST})

  target_link_libraries(${OUT_EXE} "${TORCH_LIBRARIES}")
  target_link_libraries(${OUT_EXE} RawDataWriter)
  set_property(TARGET ${OUT_EXE} PROPERTY CXX_STANDARD 17)

  if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${OUT_EXE}
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:${OUT_EXE}>)
  endif (MSVC)
endfunction()

compile_binary(xor_net "XOR/train.cpp;XOR/model.cpp")
compile_binary(mcc_cnn "MultiClassClassification/train.cpp;MultiClassClassification/model.cpp")
compile_binary(jit_mcc_cnn "MultiClassClassificationJIT/train.cpp")
# compile_binary(load_example "PytorchJIT/load-example.cpp")