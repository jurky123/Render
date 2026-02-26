# Path Tracer 路径追踪渲染器

基于 **NVIDIA OptiX 7** 的实时路径追踪渲染器，支持鼠标键盘交互界面以及多格式场景加载。

## ✨ 特性

| 功能 | 说明 |
|------|------|
| **GPU 路径追踪** | NVIDIA OptiX 7 GPU 加速，支持渐进式累积采样 |
| **交互式 UI** | GLFW + Dear ImGui，支持实时相机控制和参数调整 |
| **PBR 材质系统** | Disney 金属-粗糙度 BSDF，完整支持金属度、粗糙度、透射率、IOR |
| **多格式场景加载** | Assimp 支持 40+ 格式（OBJ、FBX、glTF、DAE、3DS、PBRT v4 等） |
| **PBRT v4 支持** | 完整的 PBRT v4 场景文件解析（摄像机、灯光、材质、几何体） |
| **CPU 回退** | 无 GPU 时自动切换为 CPU 光线追踪，界面仍可运行 |
| **色调映射** | ACES 胶片色调映射 + sRGB 伽马校正 |

---

## 📦 依赖

所有依赖均通过 CMake FetchContent 自动下载，**无需手动安装**：

| 库 | 用途 | 版本 |
|----|------|------|
| [GLFW](https://github.com/glfw/glfw) | 跨平台窗口和输入 | 3.3+ |
| [GLAD](https://github.com/Dav1dde/glad) | OpenGL 4.5 加载器 | 最新 |
| [GLM](https://github.com/g-truc/glm) | 数学库 | 最新 |
| [Dear ImGui](https://github.com/ocornut/imgui) | UI 库 | 1.89+ |
| [Assimp](https://github.com/assimp/assimp) | 3D 模型导入 | 5.3+ |
| **[NVIDIA OptiX SDK](https://developer.nvidia.com/optix)** | GPU 光线追踪（可选） | 7.x |
| **CUDA Toolkit** | GPU 计算（OptiX 时必须） | 11+ |

### 系统要求

- **编译器**: C++17（MSVC 2019+、GCC 9+、Clang 10+）
- **CMake**: ≥ 3.18
- **OpenGL 驱动**: 4.5+
- **GPU**（可选）: NVIDIA GTX 960+ 或 RTX（建议）

---

## 🔨 编译指南

### 快速开始（CPU 模式）

```bash
cd Render
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPATHTRACER_ENABLE_OPTIX=OFF
cmake --build build --config Release -j$(nproc)
cd build/Debug
./PathTracer.exe
```

### 启用 OptiX GPU 加速（Windows/MSVC）

```powershell
# 下载 OptiX SDK 7.x 并设置环境变量
$env:OPTIX_ROOT = "C:\Program Files\NVIDIA\OptiX SDK 7.x.0"

# 配置和编译
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPATHTRACER_ENABLE_OPTIX=ON
cmake --build build --config Release

# 运行
.\build\Release\PathTracer.exe
```

### Linux/GCC 编译

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DPATHTRACER_ENABLE_OPTIX=OFF

cmake --build build --config Release -j$(nproc)
./build/PathTracer
```

### 编译选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `DPATHTRACER_ENABLE_OPTIX` | 启用 OptiX GPU 支持 | `OFF` |
| `CMAKE_BUILD_TYPE` | 构建类型 | `Release` |

---

## ⌨️ 使用指南

### 启动命令

**加载特定场景：**
```bash
PathTracer.exe --scene /path/to/scene.glb
PathTracer.exe --scene /path/to/scene.pbrt        # PBRT v4 格式
PathTracer.exe --scene /path/to/scene.obj         # Wavefront OBJ
```

**自定义窗口大小和采样：**
```bash
PathTracer.exe --scene scene.gltf --width 1920 --height 1200 --spp 8
```

**诊断 PBRT 场景（查看加载的摄像机、灯光、材质）：**
```bash
PathTracer.exe --test-pbrt scene.pbrt              # 输出详细解析信息
```

### 运行时快捷键

| 按键 | 功能 |
|------|------|
| `W / A / S / D` | 前后左右移动（飞行相机） |
| `Q / E` | 下降 / 上升 |
| `鼠标右键拖拽` | 旋转视点 |
| `鼠标滚轮` | 缩放（改变 FOV） |
| `F1` | 显示/隐藏 UI 面板 |
| `Esc` | 退出程序 |

### PBRT v4 场景支持

PathTracer 完整支持 PBRT v4 场景文件，包括：
- ✅ 摄像机（LookAt、投影参数）
- ✅ 材质（导电体、电介质、漫反射、涂层等）
- ✅ 灯光（点光源、方向光、面光源）
- ✅ 几何体（网格、PLY、球体、圆柱等）
- ✅ 变换（平移、旋转、缩放）
- ✅ 引用和命名对象（Include、ObjectBegin/End）

**加载官方 PBRT v4 场景示例：**
```bash
PathTracer.exe --scene pbrt-v4-scenes/crown/crown.pbrt
PathTracer.exe --test-pbrt pbrt-v4-scenes/sportscar/sportscar-area-lights.pbrt
```

---

## 📁 项目结构

```
Render/
├── CMakeLists.txt              # 主 CMake 构建脚本
├── README.md                   # 本文件
├── .gitignore                  # Git 忽略配置
│
├── src/                        # 源代码目录
│   ├── main.cpp                # 程序入口点（命令行解析）
│   ├── Application.h/cpp       # 应用主循环（GLFW、ImGui、渲染循环）
│   ├── Camera.h/cpp            # 飞行相机实现
│   ├── Scene.h/cpp             # 场景数据结构和加载器
│   ├── Material.h/cpp          # PBR 材质定义
│   ├── Renderer.h/cpp          # 渲染器门面
│   ├── OptixRenderer.h/cpp     # OptiX 7 + CPU 回退
│   ├── Window.h/cpp            # 窗口工具
│   ├── PBRTLoader.h/cpp        # PBRT v4 完整解析器 (~1700 行)
│   └── PBRTLoaderOfficial.h    # 官方库包装（预留）
│
├── shaders/                    # CUDA/OptiX 着色器
│   ├── launch_params.h         # 共享数据结构
│   ├── path_tracer.cu          # OptiX 核心程序
│   └── *.ptx                   # 编译输出（git 忽略）
│
├── build/                      # CMake 构建目录（自动生成，git 忽略）
│   ├── Debug/PathTracer.exe    # 最终可执行文件
│   └── ...
│
├── pbrt-v4/                    # PBRT v4 官方库（参考用）
├── pbrt-v4-scenes/             # PBRT v4 测试场景
└── models/                     # 示例模型（git 忽略）
```

### 核心模块说明

| 文件 | 功能 | 关键点 |
|------|------|--------|
| **PBRTLoader** | PBRT v4 解析 | Token-based，支持所有材质/灯光/摄像机 |
| **OptixRenderer** | GPU 路径追踪 | OptiX 7 + CPU 回退，渐进累积 |
| **Application** | 主应用循环 | GLFW 管理、ImGui UI、渲染调度 |
| **Scene** | 场景管理 | Assimp 加载、数据组织 |
| **Camera** | 交互相机 | 飞行模式（WASD） |

---

## 🎮 常见使用场景

### 1. 加载并渲染 OBJ 模型
```bash
./PathTracer.exe --scene models/cornell_box/cbox.obj -width 1280 --height 720
```

### 2. 渲染 PBRT v4 官方场景
```bash
./PathTracer.exe --scene pbrt-v4-scenes/crown/crown.pbrt
```

### 3. 诊断 PBRT 场景加载情况
```bash
./PathTracer.exe --test-pbrt pbrt-v4-scenes/sportscar/sportscar-area-lights.pbrt
# 输出：摄像机、灯光、材质数量及属性
```

### 4. 高分辨率高质量渲染
```bash
./PathTracer.exe --scene scene.gltf --width 4096 --height 2160 --spp 32
```

---

## 🔍 API 文档

### PBRT v4 加载器验证

**Current Status**: ✅ 完全支持

通过 `--test-pbrt` 模式可验证：
- ✅ 摄像机正确解析（Eye, Target, Up, FOVy）
- ✅ 所有材质属性解析（色彩、金属度、粗糙度、IOR、透射）
- ✅ 灯光完整解析（位置、方向、强度）
- ✅ 几何体正确加载（网格、PLY、原始体）

### 扩展指南

添加新格式支持：修改 `PBRTLoader.cpp` 或 `Scene.cpp`
添加新材质类型：修改 `Material.h` + `OptixRenderer.cpp`
自定义 UI：编辑 `Application.cpp` 中 ImGui 部分

---

## 📄 截图 Screenshots

> （待 GPU 环境下截图补充）

---

## License

MIT
