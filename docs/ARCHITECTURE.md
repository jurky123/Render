# PathTracer Architecture

## Overview

The runtime is organized as:

- `app`: application shell, UI, scene selection, camera/input control
- `scene`: scene data model, importers, and scene post-processing
- `render`: renderer facade plus backend abstraction
- `render/optix`: OptiX-facing backend adapter
- `shaders`: OptiX device programs and shared launch data

The main flow is:

`main -> app::Application -> SceneController / ViewportController -> Renderer -> OptixBackend -> OptixRenderer -> shaders/path_tracer.cu`

## Directory Layout

### `src/app`

- `Application`: owns the window, ImGui lifecycle, and frame loop
- `AppState`: shared UI/runtime state
- `SceneController`: scans scenes and loads them through `SceneImporter`
- `ViewportController`: maps GLFW input to camera movement and accumulation reset

### `src/scene`

- `Scene`: pure in-memory scene container
- `SceneImporter`: high-level importer entrypoint
- `ScenePostProcessor`: default light/camera synthesis and stats rebuild

### `src/scene/importers`

- `AssimpSceneImporter`: mesh/material/light/camera import through Assimp
- `YamlSceneImporter`: YAML scene composition
- `MaterialLoader`: converts external material descriptions to internal `Material`
- `TextureRepository`: owns texture loading helpers

### `src/render`

- `Renderer`: OpenGL display facade and backend host
- `RenderSettings`: frame render settings passed as one object
- `IRenderBackend`: backend contract
- `CpuFallbackBackend`: simple fallback output path

### `src/render/optix`

- `OptixBackend`: backend adapter used by `Renderer`
- Current implementation delegates to legacy `OptixRenderer` while the OptiX internals are being further split

### `shaders`

- `launch_params.h`: host/device shared ABI structs
- `path_tracer.cu`: OptiX entrypoints and main path tracing implementation
- `device/*.cuh`: extracted device-side responsibility boundaries for payload, math, textures, materials, curves, and closest-hit helpers

## Runtime Responsibilities

### Application Layer

- Initializes GLFW, GLAD, and ImGui
- Drives the per-frame update/render loop
- Builds the UI panels
- Delegates scene actions and camera/input behavior to controllers

### Scene Layer

- Produces a normalized `Scene` regardless of source format
- Keeps import logic out of the scene container itself
- Applies post-load defaults centrally instead of scattering them through loaders

### Render Layer

- Accepts `RenderSettings` as a single struct
- Keeps OpenGL presentation separate from backend rendering
- Makes backend replacement possible without changing the app layer

### Shader Layer

- Uses `launch_params.h` for host/device ABI stability
- Keeps entrypoints in `path_tracer.cu`
- Gradually moves shared device logic into `shaders/device`

## Compatibility Notes

- Root-level headers `src/Application.h`, `src/Scene.h`, and `src/Renderer.h` remain as compatibility wrappers.
- The legacy `OptixRenderer` is still in use under `OptixBackend` to avoid breaking GPU rendering while the deeper OptiX split is completed.

## Build Notes

- `CMAKE_SUPPRESS_REGENERATION` is enabled to avoid Visual Studio/MSBuild stamp regeneration issues in this workspace.
- PTX is now loaded directly from `build/ptx_shaders.dir/<Config>` instead of being copied with a post-build step.
