cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)  # MSVC runtime library flags
project(node_stable_diffusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- GPU auto-detection ---
if(APPLE)
    set(SD_METAL ON CACHE BOOL "" FORCE)
    message(STATUS "node-sd-cpp: Metal backend enabled (macOS)")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(SD_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "node-sd-cpp: CUDA backend enabled")
endif()

find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    set(SD_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "node-sd-cpp: Vulkan backend enabled")
endif()

# --- stable-diffusion.cpp as static lib ---
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Disable native CPU optimizations for portability and to avoid
# feature detection failures with some compiler configurations.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/stable-diffusion.cpp)

# --- Node.js / N-API ---
include_directories(${CMAKE_JS_INC})

execute_process(
    COMMAND node -p "require('node-addon-api').include_dir"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NAPI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
# Strip quotes from the output
string(REPLACE "\"" "" NAPI_INCLUDE_DIR ${NAPI_INCLUDE_DIR})
string(REPLACE "'" "" NAPI_INCLUDE_DIR ${NAPI_INCLUDE_DIR})

# --- Addon sources ---
file(GLOB_RECURSE ADDON_SOURCES
    "src/cpp/*.cpp"
    "src/cpp/*.h"
)

add_library(${PROJECT_NAME} SHARED
    ${ADDON_SOURCES}
    ${CMAKE_JS_SRC}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${NAPI_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/deps/stable-diffusion.cpp/include
    ${CMAKE_SOURCE_DIR}/src/cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    stable-diffusion
    ${CMAKE_JS_LIB}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    NAPI_VERSION=8
    NAPI_DISABLE_CPP_EXCEPTIONS
    NODE_ADDON_API_DISABLE_DEPRECATED
)
