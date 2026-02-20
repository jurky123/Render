cmake_minimum_required(VERSION 3.18)
project(PathTracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
            $ENV{OPTIX7_PATH}/include
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
# Assimp
# -------------------------------------------------------------------
find_package(assimp QUIET)
if(NOT assimp_FOUND)
    message(STATUS "System Assimp not found – fetching from source.")
    FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
        GIT_TAG        v5.3.1
        GIT_SHALLOW    TRUE
    )
    set(ASSIMP_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INSTALL            OFF CACHE BOOL "" FORCE)
    set(ASSIMP_NO_EXPORT          ON  CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(assimp)
endif()

# -------------------------------------------------------------------
# Path Tracer executable
# -------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/Application.cpp
    src/Window.cpp
    src/Camera.cpp
    src/Scene.cpp
    src/Material.cpp
    src/Renderer.cpp
    src/OptixRenderer.cpp
)

add_executable(PathTracer ${SOURCES})

target_include_directories(PathTracer PRIVATE
    src
    shaders
)

target_link_libraries(PathTracer PRIVATE
    OpenGL::GL
    glfw
    glad
    glm
    imgui
    $<IF:$<TARGET_EXISTS:assimp::assimp>,assimp::assimp,assimp>
)

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
    )
    target_compile_options(ptx_shaders PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>
    )

    add_custom_command(
        TARGET PathTracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:ptx_shaders>
            $<TARGET_FILE_DIR:PathTracer>/path_tracer.ptx
        COMMENT "Copying PTX shaders"
    )

    target_compile_definitions(PathTracer PRIVATE
        PTX_DIR=\"$<TARGET_FILE_DIR:PathTracer>\"
    )
endif()

# -------------------------------------------------------------------
# Optional: install targets
# -------------------------------------------------------------------
install(TARGETS PathTracer DESTINATION bin)
