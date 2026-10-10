cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)  # MSVC runtime library flags
project(node_stable_diffusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Backend-DL mode: build ggml backends (CPU variants, Vulkan, …) as
# runtime-loadable libraries placed next to the addon instead of linking one
# CPU backend statically. The addon then picks the best CPU variant at
# runtime and skips GPU backends whose system libraries are missing (e.g. no
# Vulkan loader) instead of failing to load entirely. This is how the
# published linux/win prebuilds are built. Accepted as a cmake option or as
# an environment variable (cmake-js does not reliably forward -D vars to
# cmake configure).
option(SD_NODE_BACKEND_DL "Build ggml backends as runtime-loadable libraries" OFF)
if(NOT SD_NODE_BACKEND_DL AND DEFINED ENV{SD_NODE_BACKEND_DL}
   AND NOT "$ENV{SD_NODE_BACKEND_DL}" STREQUAL ""
   AND NOT "$ENV{SD_NODE_BACKEND_DL}" STREQUAL "0")
    set(SD_NODE_BACKEND_DL ON)
endif()

# Force the ggml Vulkan backend on, independent of the find_package
# auto-detection below (CI installs the Vulkan SDK, but explicit beats
# detection for reproducible prebuilds). Requires the Vulkan SDK (glslc +
# headers) at build time only; at runtime just the system Vulkan loader.
# Combine with SD_NODE_BACKEND_DL so machines without the loader fall back
# to CPU instead of failing to load the addon.
option(SD_NODE_VULKAN "Enable the ggml Vulkan backend" OFF)
if(NOT SD_NODE_VULKAN AND DEFINED ENV{SD_NODE_VULKAN}
   AND NOT "$ENV{SD_NODE_VULKAN}" STREQUAL ""
   AND NOT "$ENV{SD_NODE_VULKAN}" STREQUAL "0")
    set(SD_NODE_VULKAN ON)
endif()
if(SD_NODE_VULKAN)
    set(SD_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "node-sd-cpp: Vulkan backend enabled (SD_NODE_VULKAN)")
endif()

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
if(SD_NODE_BACKEND_DL)
    # GGML_BACKEND_DL requires shared ggml: ggml/ggml-base become .so/.dll
    # files shipped next to the addon, and each backend becomes a MODULE
    # library that ggml dlopens at runtime. SD_BUILD_SHARED_GGML_LIB stops
    # sd.cpp from forcing BUILD_SHARED_LIBS OFF while the sd library itself
    # stays STATIC inside the addon (SD_BUILD_SHARED_LIBS above).
    set(SD_BUILD_SHARED_GGML_LIB ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS ON  CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL   ON  CACHE BOOL "" FORCE)
    # One module per CPU microarchitecture (SSE4.2 … AVX512/AMX on x86,
    # armv8.x/armv9.x on ARM); ggml scores them at runtime and loads the
    # best one. Unsupported on Apple platforms (mac prebuilds stay static
    # anyway — Metal covers acceleration there).
    if(NOT APPLE)
        set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    endif()
    # Everything is shipped flat next to node_stable_diffusion.node: resolve
    # the shared-lib deps relative to each binary, not the build tree.
    if(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN")
    endif()
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()
add_subdirectory(deps/stable-diffusion.cpp)

# --- Node.js / N-API ---
include_directories(${CMAKE_JS_INC})

# On Windows, generate node.lib from the .def file provided by cmake-js
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
                    /out:${CMAKE_JS_NODELIB_TARGET}
                    ${CMAKE_STATIC_LINKER_FLAGS})
endif()

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

# Backend-DL packaging: collect every deps-tree shared/MODULE library (ggml,
# ggml-base, and all dlopen'd backends — whatever this sd.cpp/ggml version
# produces) and stage them next to node_stable_diffusion.node so the
# directory is self-contained. The addon preloads backends from its own
# directory at runtime (src/cpp/addon.cpp).
if(SD_NODE_BACKEND_DL)
    # Tells addon.cpp to load backends from the addon's directory instead of
    # relying on sd.cpp's default-path scan (executable dir + cwd), which
    # would never find them inside node_modules.
    target_compile_definitions(${PROJECT_NAME} PRIVATE SD_NODE_BACKEND_DL)

    function(sd_node_collect_runtime_targets out_var dir)
        set(result "")
        get_property(dir_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(tgt IN LISTS dir_targets)
            get_target_property(tgt_type ${tgt} TYPE)
            if(tgt_type STREQUAL "SHARED_LIBRARY" OR tgt_type STREQUAL "MODULE_LIBRARY")
                list(APPEND result ${tgt})
            endif()
        endforeach()
        get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
        foreach(sub IN LISTS subdirs)
            sd_node_collect_runtime_targets(sub_result "${sub}")
            list(APPEND result ${sub_result})
        endforeach()
        set(${out_var} ${result} PARENT_SCOPE)
    endfunction()

    sd_node_collect_runtime_targets(SD_NODE_RUNTIME_TARGETS
        "${CMAKE_SOURCE_DIR}/deps/stable-diffusion.cpp")

    foreach(tgt IN LISTS SD_NODE_RUNTIME_TARGETS)
        # Strip VERSION/SOVERSION: npm packages cannot contain symlinks, so
        # every library must be a single real file with its plain name
        # (libggml-vulkan.so, not libggml-vulkan.so.0.9.5 + symlinks) — the
        # ggml backend scan also only matches the plain ".so" extension.
        set_property(TARGET ${tgt} PROPERTY VERSION)
        set_property(TARGET ${tgt} PROPERTY SOVERSION)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${tgt}>" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach()
    list(LENGTH SD_NODE_RUNTIME_TARGETS _n)
    message(STATUS "node-sd-cpp: backend-DL mode, staging ${_n} runtime libs next to the addon: ${SD_NODE_RUNTIME_TARGETS}")
endif()
