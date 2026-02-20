# Path Tracer 路径追踪渲染器

基于 **NVIDIA OptiX 7** 的实时路径追踪渲染器，支持鼠标键盘交互界面以及高兼容性的场景/模型/材质加载系统。

---

## 特性 Features

| 功能 | 说明 |
|------|------|
| 路径追踪管线 | NVIDIA OptiX 7 GPU 加速路径追踪，支持渐进式累积采样 |
| 交互式界面 | GLFW + Dear ImGui，支持鼠标旋转、WASD 飞行相机、滚轮缩放 |
| PBR 材质 | Disney 金属-粗糙度 BSDF（基础色、金属度、粗糙度、自发光、透射、IOR） |
| 场景加载 | Assimp 支持 OBJ、FBX、glTF/GLB、DAE、3DS、PLY、STL 等 40+ 格式 |
| CPU 回退 | 无 GPU/OptiX 时自动切换为 CPU 软件渲染器，界面仍可正常运行 |
| 色调映射 | ACES 胶片色调映射 + sRGB 伽马校正 |

---

## 依赖 Dependencies

所有依赖均通过 CMake FetchContent 自动下载，无需手动安装：

| 库 | 用途 |
|----|------|
| [GLFW 3.3](https://github.com/glfw/glfw) | 跨平台窗口 / 输入 |
| [GLAD](https://github.com/Dav1dde/glad) | OpenGL 4.5 加载器 |
| [GLM](https://github.com/g-truc/glm) | 数学库 |
| [Dear ImGui v1.89](https://github.com/ocornut/imgui) | 即时模式 GUI |
| [Assimp v5.3](https://github.com/assimp/assimp) | 模型/场景导入 |
| NVIDIA OptiX 7.x SDK | GPU 光线追踪（可选） |
| CUDA Toolkit 11+ | GPU 计算（OptiX 开启时必须） |

---

## 构建 Build

### 前置条件

- CMake ≥ 3.18
- C++17 编译器（MSVC 2019+、GCC 9+、Clang 10+）
- OpenGL 4.5 驱动
- （可选）NVIDIA GPU + [OptiX 7 SDK](https://developer.nvidia.com/optix) + CUDA Toolkit 11+

### 编译步骤

```bash
# 克隆仓库
git clone https://github.com/jurky123/Render.git
cd Render

# 配置（OptiX 可选，SDK 路径通过环境变量传入）
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPATHTRACER_ENABLE_OPTIX=ON \
      -DOPTIX_ROOT=/path/to/OptiX-SDK-7.x.0

# 编译
cmake --build build --config Release -j$(nproc)

# 运行
./build/PathTracer --scene /path/to/scene.gltf
```

**不带 OptiX（CPU 模式）：**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPATHTRACER_ENABLE_OPTIX=OFF
cmake --build build --config Release -j$(nproc)
./build/PathTracer
```

---

## 操作说明 Controls

| 输入 | 功能 |
|------|------|
| `W / A / S / D` | 前后左右移动 |
| `Q / E` | 下降 / 上升 |
| 鼠标右键拖拽 | 旋转视角 |
| 鼠标滚轮 | 缩放（调整 FoV） |
| `F1` | 显示/隐藏 GUI 面板 |
| `Esc` | 退出 |

---

## 命令行参数 CLI

```
PathTracer [OPTIONS]

Options:
  --scene <path>   启动时加载场景文件
  --width  <px>    窗口宽度（默认 1280）
  --height <px>    窗口高度（默认 720）
  --spp   <n>      每像素初始采样数（默认 4）
  --help           显示帮助
```

---

## 项目结构 Project Structure

```
Render/
├── CMakeLists.txt          # CMake 构建脚本
├── src/
│   ├── main.cpp            # 程序入口，命令行解析
│   ├── Application.h/cpp   # 应用主循环，GLFW 回调，ImGui UI
│   ├── Camera.h/cpp        # 第一人称相机（飞行模式）
│   ├── Scene.h/cpp         # 场景加载（Assimp 包装）
│   ├── Material.h/cpp      # PBR 材质定义与工具函数
│   ├── Renderer.h/cpp      # 渲染器门面（OpenGL 全屏 quad 输出）
│   ├── OptixRenderer.h/cpp # OptiX 7 管线 + CPU 回退渲染器
│   └── Window.h/cpp        # 窗口工具（预留扩展）
└── shaders/
    ├── launch_params.h     # CPU/GPU 共享数据结构
    └── path_tracer.cu      # OptiX 设备程序（raygen / miss / closesthit）
```

---

## 截图 Screenshots

> （待 GPU 环境下截图补充）

---

## License

MIT
