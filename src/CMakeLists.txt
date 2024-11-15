cmake_minimum_required(VERSION 3.18)
project(ltct LANGUAGES CXX)

include(CMakePrintHelpers)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(WIN32)
    # disable windows asan for now since I don't know how to make it work
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
else()
    # asan seems to be broken in my machine for some reason - segfaulting in some cases even there is no errors
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

##############################################################################
                    # DOWNLOAD AND BUILD LIBJPEG-TURBO
##############################################################################

# Optionally build shared or static libraries based on the BUILD_SHARED_LIBS flag
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

if (BUILD_SHARED_LIBS)
    set(LIBJPEGTURBO_STATIC_LIBRARY_FLAG OFF)
    set(LIBJPEGTURBO_SHARED_LIBRARY_FLAG ON)
else()
    set(LIBJPEGTURBO_STATIC_LIBRARY_FLAG ON)
    set(LIBJPEGTURBO_SHARED_LIBRARY_FLAG OFF)
endif()

# Include the ExternalProject module
include(ExternalProject)

# Define the libjpeg-turbo external project
ExternalProject_Add(LibJpegTurbo
    URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.0.4.zip
    PREFIX ${CMAKE_CURRENT_LIST_DIR}/turbojpeg
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/turbojpeg
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DENABLE_STATIC=${LIBJPEGTURBO_STATIC_LIBRARY_FLAG}
      -DENABLE_SHARED=${LIBJPEGTURBO_SHARED_LIBRARY_FLAG}
      -DENABLE_TESTING=OFF
)

# Delay finding the library until after LibJpegTurbo is built
ExternalProject_Get_Property(LibJpegTurbo install_dir)

# Specify include and library paths after the build of LibJpegTurbo
set(LIBJPEG_TURBO_INCLUDE_DIR ${install_dir}/include)
set(LIBJPEG_TURBO_LIB_DIR ${install_dir}/lib)

# Set the paths of the libraries for specific platforms
if (BUILD_SHARED_LIBS)
    if(WIN32 AND NOT MSVC)
        set(LIBJPEG_TURBO_LIB "${LIBJPEG_TURBO_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}turbojpeg${CMAKE_SHARED_LIBRARY_SUFFIX}.a")
    elseif(MSVC)
        set(LIBJPEG_TURBO_LIB "${LIBJPEG_TURBO_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}turbojpeg.lib")
    else()
        set(LIBJPEG_TURBO_LIB "${LIBJPEG_TURBO_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}turbojpeg${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
else()
    if (MSVC)
        set(LIBJPEG_TURBO_LIB "${LIBJPEG_TURBO_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg-static${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(LIBJPEG_TURBO_LIB "${LIBJPEG_TURBO_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
endif()

##############################################################################
                    # DOWNLOAD AND BUILD LIBTORCH
##############################################################################

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

##############################################################################
                  # BUILD CUSTOM C++ RAW DATA WRITER
##############################################################################

file(GLOB RAW_DATA_WRITER_SOURCES "raw_data_writer/*.cpp")

add_library(RawDataWriter STATIC ${RAW_DATA_WRITER_SOURCES})
set_property(TARGET RawDataWriter PROPERTY CXX_STANDARD 17)

##############################################################################
                  # BUILD CUSTOM IMAGE READER
##############################################################################

file(GLOB IMAGE_READER_SOURCES "ImageReader/*.cpp")

add_library(ImageReader STATIC ${IMAGE_READER_SOURCES})
add_dependencies(ImageReader LibJpegTurbo)

target_include_directories(ImageReader PRIVATE ${LIBJPEG_TURBO_INCLUDE_DIR})
target_link_libraries(ImageReader ${LIBJPEG_TURBO_LIB})
target_link_libraries(ImageReader "${TORCH_LIBRARIES}")

set_property(TARGET ImageReader PROPERTY CXX_STANDARD 17)

##############################################################################
                    # FUNCTION FOR COMPILING BINs
##############################################################################

function(compile_binary OUT_EXE SRC_LIST)
  add_executable(${OUT_EXE} ${SRC_LIST})

  add_dependencies(${OUT_EXE} LibJpegTurbo)

  target_link_libraries(${OUT_EXE} RawDataWriter)
  target_link_libraries(${OUT_EXE} ImageReader)

  set_property(TARGET ${OUT_EXE} PROPERTY CXX_STANDARD 17)

  if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${OUT_EXE}
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:${OUT_EXE}>)
  endif (MSVC)

  if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/bin/*.dll")
    add_custom_command(TARGET ${OUT_EXE}
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:${OUT_EXE}>)
  endif (MSVC)

  # Copy DLLs to the output directory on windows if there are
  if (WIN32)
  add_custom_command(TARGET main POST_BUILD
      COMMENT "Copying DLL files to main bin directory"

      # copy needed dll file to main binary path
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${install_dir}/bin $<TARGET_FILE_DIR:main>

      # remove not needed binaries
      COMMAND ${CMAKE_COMMAND} -E rm $<TARGET_FILE_DIR:main>/cjpeg${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E rm $<TARGET_FILE_DIR:main>/djpeg${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E rm $<TARGET_FILE_DIR:main>/jpegtran${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E rm $<TARGET_FILE_DIR:main>/tjbench${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E rm $<TARGET_FILE_DIR:main>/rdjpgcom${CMAKE_EXECUTABLE_SUFFIX}
      COMMAND ${CMAKE_COMMAND} -E rm $<TARGET_FILE_DIR:main>/wrjpgcom${CMAKE_EXECUTABLE_SUFFIX}
  )
  endif ()
endfunction()

##############################################################################
                # CALL THE FUNCTION AND COMPILE THE BINARIES
##############################################################################

# compile_binary(xor_net "XOR/train.cpp;XOR/model.cpp")
# compile_binary(mcc_cnn "MultiClassClassification/train.cpp;MultiClassClassification/model.cpp")
# compile_binary(jit_mcc_cnn "MultiClassClassificationJIT/train.cpp")
compile_binary(data_loader "BinaryClassificationJIT/dataloader.cpp")