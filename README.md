# PathTracer 渲染器项目

## 项目简介
本项目为基于 OptiX 的高性能物理路径追踪渲染器，支持加载和渲染多种 3D 场景格式，具备现代 OpenGL/OptiX 渲染管线、材质与贴图支持、可交互相机与光源配置。

## 主要特性
- 支持 YAML 场景描述文件（自定义相机、光源、模型组合）
- 支持 Assimp 加载主流网格格式：OBJ、FBX、GLTF/GLB、DAE、PLY、STL 等
- 材质系统支持 PBR（金属度/粗糙度）、基础漫反射、镜面、玻璃等类型
- 贴图支持 JPG/PNG（自动转 RGBA，shader 采样一致）
- 可配置环境光、定向光、点光源
- 交互式窗口，支持 ImGui UI 调节参数
- CUDA/OptiX 加速路径追踪，支持多光线反弹

## 目录结构
```
├─models/           # 场景与模型资源（YAML/OBJ/贴图等）
├─src/              # 源码（Application/Scene/Renderer/OptixRenderer 等）
├─shaders/          # CUDA/OptiX shader 内核
├─include/          # 头文件（如 tinyexr.h）
├─build/            # CMake 构建输出
├─CMakeLists.txt    # 构建脚本，自动拉取依赖
├─README.md         # 项目说明
```

## 场景加载说明
- YAML 场景：如 `models/breakfast_room/breakfast_room.yaml`，可指定相机、光源、模型组合，支持 `Models`/`ComplexModels` 节点。
- 网格文件：直接加载 OBJ/FBX/GLTF/DAE/PLY/等，自动解析材质与贴图。
- 贴图：支持 JPG/PNG，自动转 RGBA，shader 采样一致。

## 构建与运行
1. 安装 CUDA/OptiX SDK（建议 CUDA 11+，OptiX 7.7+/8.1+）
2. Windows 推荐 VS2022 + CMake 3.18+
3. 进入项目根目录，执行：
   ```
   cmake -S . -B build
   cmake --build build --config Release
   ```
4. 运行 `build/Debug/PathTracer.exe` 或 `build/Release/PathTracer.exe`

## 依赖说明
- GLFW、GLAD、GLM、ImGui、yaml-cpp、assimp（均自动拉取，无需手动下载）
- stb_image（本地实现，支持贴图加载）

## 常见问题
- 场景贴图异常：请确保图片为常见格式（JPG/PNG），且与模型/mtl文件路径一致
- OptiX/CUDA 环境未配置：请检查 SDK 路径与环境变量
- 其他问题可查阅 `PathTracer_debug.log` 日志

## 贡献与扩展
欢迎提交 issue 或 PR，支持更多格式、材质类型、渲染特性扩展。

---

如需自定义场景，请参考 `models/cornell_box/cornell_box.yaml` 或 `models/breakfast_room/breakfast_room.yaml`。
