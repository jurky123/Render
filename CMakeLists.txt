cmake_minimum_required(VERSION 3.18)
project(PathTracer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_REGENERATION ON)

# Force Visual Studio toolchain on Windows to avoid MSYS64 interference
if(WIN32 AND NOT CMAKE_C_COMPILER_FORCED)
    # Only set this once to avoid repeated forced changes
    set(CMAKE_C_COMPILER_FORCED TRUE CACHE BOOL "")
    # Let CMake auto-detect MSVC, but ensure it doesn't use MSYS64
    if(NOT DEFINED CMAKE_C_COMPILER)
        # Force MSVC compiler identification
        set(CMAKE_C_COMPILER_ID "MSVC")
    endif()
endif()

# -------------------------------------------------------------------
# Options
# -------------------------------------------------------------------
option(PATHTRACER_ENABLE_OPTIX "Enable NVIDIA OptiX path tracing" ON)

# Allow enabling CUDA only when OptiX is requested
if(PATHTRACER_ENABLE_OPTIX)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# -------------------------------------------------------------------
# CUDA (only required when OptiX is on)
# -------------------------------------------------------------------
if(PATHTRACER_ENABLE_OPTIX)
    find_package(CUDAToolkit REQUIRED)
endif()

# -------------------------------------------------------------------
# OptiX
# -------------------------------------------------------------------
if(PATHTRACER_ENABLE_OPTIX)
    find_path(OPTIX_INCLUDE_DIR
        NAMES optix.h
        PATHS
            $ENV{OPTIX_ROOT}/include
            $ENV{OPTIX8_PATH}/include
            $ENV{OPTIX7_PATH}/include
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0/include"
            /usr/local/optix/include
            /opt/optix/include
        NO_DEFAULT_PATH
    )

    if(NOT OPTIX_INCLUDE_DIR)
        message(WARNING
            "OptiX headers not found. "
            "Set OPTIX_ROOT to your OptiX SDK install directory. "
            "Building without OptiX support.")
        set(PATHTRACER_ENABLE_OPTIX OFF)
    endif()
endif()

# -------------------------------------------------------------------
# FetchContent setup
# -------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# -------------------------------------------------------------------
# OpenGL
# -------------------------------------------------------------------
find_package(OpenGL REQUIRED)

# -------------------------------------------------------------------
# GLFW
# -------------------------------------------------------------------
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "System GLFW not found – fetching from source.")
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.8
        GIT_SHALLOW    TRUE
    )
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

# -------------------------------------------------------------------
# GLAD (OpenGL loader)
# -------------------------------------------------------------------
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v0.1.36
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    FetchContent_Populate(glad)
    set(GLAD_PROFILE      "core"  CACHE STRING "" FORCE)
    set(GLAD_API          "gl=4.5" CACHE STRING "" FORCE)
    set(GLAD_GENERATOR    "c"      CACHE STRING "" FORCE)
    set(GLAD_EXTENSIONS   ""       CACHE STRING "" FORCE)
    set(GLAD_SPEC         "gl"     CACHE STRING "" FORCE)
    set(GLAD_ALL_EXTENSIONS OFF    CACHE BOOL   "" FORCE)
    add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR})
endif()

# -------------------------------------------------------------------
# GLM (header-only math library)
# -------------------------------------------------------------------
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.8
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# -------------------------------------------------------------------
# Dear ImGui
# -------------------------------------------------------------------
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.89.9
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# -------------------------------------------------------------------
# yaml-cpp (always fetch from source for consistency)
# -------------------------------------------------------------------
message(STATUS "Fetching yaml-cpp from source...")
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.7.0
    GIT_SHALLOW TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)
message(STATUS "yaml-cpp fetched successfully")

# -------------------------------------------------------------------
# Assimp (kept for model loading)
# -------------------------------------------------------------------
find_package(assimp QUIET)
if(NOT assimp_FOUND)
    message(STATUS "System Assimp not found – fetching from source.")
    set(ASSIMP_LOCAL_SRC "${CMAKE_SOURCE_DIR}/assimp-5.3.1")
    if(EXISTS ${ASSIMP_LOCAL_SRC})
        message(STATUS "Using local assimp source: ${ASSIMP_LOCAL_SRC}")
        set(ASSIMP_URL "file://${ASSIMP_LOCAL_SRC}")
    else()
        set(ASSIMP_URL https://github.com/assimp/assimp/archive/refs/tags/v5.3.1.zip)
    endif()
    FetchContent_Declare(
        assimp
        URL ${ASSIMP_URL}
    )
    set(ASSIMP_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INSTALL            OFF CACHE BOOL "" FORCE)
    set(ASSIMP_NO_EXPORT          ON  CACHE BOOL "" FORCE)
    set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

    FetchContent_GetProperties(assimp)
    if(NOT assimp_POPULATED)
        FetchContent_Populate(assimp)

        # assimp's CMake adds TARGET_COMPILE_FEATURES(assimp PRIVATE c_std_99), which
        # fails on MSVC because C compile features are not modeled. Strip that line
        # so the project can configure with MSVC.
        if(MSVC)
            set(_assimp_cmakelists "${assimp_SOURCE_DIR}/code/CMakeLists.txt")
            if(EXISTS ${_assimp_cmakelists})
                file(READ ${_assimp_cmakelists} _assimp_cmake_contents)
                string(REPLACE "TARGET_COMPILE_FEATURES(assimp PRIVATE c_std_99)" "" _assimp_cmake_contents "${_assimp_cmake_contents}")
                file(WRITE ${_assimp_cmakelists} "${_assimp_cmake_contents}")
            endif()
        endif()

        add_subdirectory(${assimp_SOURCE_DIR} ${assimp_BINARY_DIR})
    endif()

    if(MSVC)
        if(TARGET assimp)
            target_compile_options(assimp PRIVATE /WX- /wd4819)
        elseif(TARGET assimp::assimp)
            target_compile_options(assimp::assimp PRIVATE /WX- /wd4819)
        endif()
    endif()
endif()

# -------------------------------------------------------------------
# Path Tracer executable
# -------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/Window.cpp
    src/Camera.cpp
    src/Material.cpp
    src/OptixRenderer.cpp
    src/stb_image_impl.cpp
    src/app/Application.cpp
    src/app/SceneController.cpp
    src/app/ViewportController.cpp
    src/scene/Scene.cpp
    src/scene/SceneImporter.cpp
    src/scene/ScenePostProcessor.cpp
    src/scene/importers/TextureRepository.cpp
    src/scene/importers/MaterialLoader.cpp
    src/scene/importers/AssimpSceneImporter.cpp
    src/scene/importers/YamlSceneImporter.cpp
    src/render/Renderer.cpp
    src/render/CpuFallbackBackend.cpp
    src/render/optix/OptixBackend.cpp
)

add_executable(PathTracer ${SOURCES})

# Ensure UTF-8 source encoding on Windows (fixes C4819 in MSVC/NVCC)
if(MSVC)
    target_compile_options(PathTracer PRIVATE /utf-8)
    if(PATHTRACER_ENABLE_OPTIX)
        target_compile_options(PathTracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

target_include_directories(PathTracer PRIVATE
    src
    src/app
    src/scene
    src/scene/importers
    src/render
    src/render/optix
    shaders
    ${glad_SOURCE_DIR}/include
    ${glad_BINARY_DIR}/include
)

# OptiX headers
if(PATHTRACER_ENABLE_OPTIX AND OPTIX_INCLUDE_DIR)
    target_include_directories(PathTracer PRIVATE ${OPTIX_INCLUDE_DIR})
    target_compile_definitions(PathTracer PRIVATE PATHTRACER_OPTIX_ENABLED=1)
endif()

# Include yaml-cpp headers if available
if(TARGET yaml-cpp)
    get_target_property(_yaml_includes yaml-cpp INTERFACE_INCLUDE_DIRECTORIES)
    if(_yaml_includes)
        target_include_directories(PathTracer PRIVATE ${_yaml_includes})
    endif()
endif()

target_link_libraries(PathTracer PRIVATE
    OpenGL::GL
    glfw
    glad
    glm
    imgui
    yaml-cpp
    $<IF:$<TARGET_EXISTS:assimp::assimp>,assimp::assimp,assimp>
)

# Ensure glad headers are visible to this target (some glad CMake setups don't propagate includes reliably)
if(TARGET glad)
    get_target_property(_glad_includes glad INTERFACE_INCLUDE_DIRECTORIES)
    if(_glad_includes)
        target_include_directories(PathTracer PRIVATE ${_glad_includes})
    endif()
endif()

if(PATHTRACER_ENABLE_OPTIX)
    target_include_directories(PathTracer PRIVATE
        ${OPTIX_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(PathTracer PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
    )
    target_compile_definitions(PathTracer PRIVATE PATHTRACER_OPTIX_ENABLED)
endif()

# -------------------------------------------------------------------
# Compile CUDA device programs to PTX
# -------------------------------------------------------------------
if(PATHTRACER_ENABLE_OPTIX)
    add_library(ptx_shaders OBJECT shaders/path_tracer.cu)
    set_target_properties(ptx_shaders PROPERTIES
        CUDA_PTX_COMPILATION      ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(ptx_shaders PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OPTIX_INCLUDE_DIR}
        shaders
        shaders/device
    )
    target_compile_options(ptx_shaders PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>
    )

    target_compile_definitions(PathTracer PRIVATE
        PTX_DIR=\"${CMAKE_BINARY_DIR}/ptx_shaders.dir/$<CONFIG>\"
    )
endif()

# -------------------------------------------------------------------
# Optional: install targets
# -------------------------------------------------------------------
install(TARGETS PathTracer DESTINATION bin)
