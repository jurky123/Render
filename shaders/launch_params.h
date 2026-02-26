#pragma once

/**
 * @file launch_params.h
 * @brief 主机(C++)和设备(CUDA/OptiX)代码之间共享的结构体
 *
 * 保持此头文件不包含 C++ STL 类型，以便可以从
 * .cpp 和 .cu 编译单元中包含它。
 */

#ifdef __CUDACC__
#  include <vector_types.h>
#  include <optix_types.h>
#else
// CPU端存根，使头文件在非-CUDA编译单元中能编译
// 仅当 CUDA/OptiX 头文件未定义时才定义
#  include <cstdint>
#  ifndef __VECTOR_TYPES_H__
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
static inline float3 make_float3(float x, float y, float z) { return {x,y,z}; }
static inline float4 make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
#  endif
#  ifndef OPTIX_TYPES_H
using OptixTraversableHandle = unsigned long long;
#  endif
#endif

// ---------------------------------------------------------------------------
// 相机参数（所有均在世界空间）
// ---------------------------------------------------------------------------
struct CameraParams
{
    float3 origin;
    float3 lowerLeft;   ///< 图像平面的左下角
    float3 horizontal;  ///< 图像平面的完整水平跨度
    float3 vertical;    ///< 图像平面的完整垂直跨度
};

// ---------------------------------------------------------------------------
// Material Type Enumeration (matching pbrt-v4)
// ---------------------------------------------------------------------------
enum class MaterialTypeData {
    Diffuse,
    Conductor,
    Dielectric,
    RoughDielectric,
    CoatedDiffuse,
    CoatedConductor,
    Subsurface,
    BasicPBR  // Legacy metallic-roughness
};

// ---------------------------------------------------------------------------
// Curve Type Enumeration
// ---------------------------------------------------------------------------
enum class CurveTypeData {
    Flat,
    Cylinder,
    Ribbon
};

// ---------------------------------------------------------------------------
// 存储在命中组 SBT 记录中的每个网格材质数据
// ---------------------------------------------------------------------------
struct HitGroupData
{
    // Material type identifier
    MaterialTypeData materialType;
    
    // Legacy PBR parameters (for BasicPBR type)
    float3 baseColor;
    float  metallic;
    float  roughness;
    float3 emission;
    float  ior;           ///< 折射率（默认1.45玻璃，水1.33）
    float  transmission;  ///< 透射度 0=不透射 1=完全透射
    
    // pbrt-v4 material parameters
    float  sigma;         ///< Oren-Nayar roughness for diffuse
    float3 eta;           ///< Conductor IOR (complex, real part)
    float3 k;             ///< Conductor extinction coefficient
    float  uroughness;    ///< U-direction roughness
    float  vroughness;    ///< V-direction roughness
    int    remapRoughness; ///< Whether to remap roughness values

    // 用于 closesthit 中法线计算的几何数据
    float3*       vertices;      ///< 指向顶点位置的设备指针
    float3*       normals;       ///< 指向顶点法线的设备指针（可能为null）
    unsigned int* indices;       ///< 指向索引缓冲区的设备指针
    int           hasNormals;    ///< 如果法线缓冲区有效则为1
    int           materialIndex; ///< 此网格的材质索引

    // 材质的纹理数据
    // Base Color / Albedo 纹理
    uint8_t*      baseColorTexPixels; ///< 指向RGBA格式纹理像素的设备指针
    int           baseColorTexWidth;  ///< 纹理宽度
    int           baseColorTexHeight; ///< 纹理高度
    int           hasBaseColorTex;    ///< 如果 baseColorTexPixels 有效则为1

    // Metallic + Roughness 纹理（通常合并：R=金属度，G=粗糙度）
    uint8_t*      metallicRoughnessTexPixels;
    int           metallicRoughnessTexWidth;
    int           metallicRoughnessTexHeight;
    int           hasMetallicRoughnessTex;

    // 法线贴图（用于表面细节）
    uint8_t*      normalTexPixels;
    int           normalTexWidth;
    int           normalTexHeight;
    int           hasNormalTex;

    // 发光贴图
    uint8_t*      emissionTexPixels;
    float4*               envmapPixelsF;  ///< HDR环境贴图像素 (float4, linear)
    int           emissionTexWidth;
    int           emissionTexHeight;
    int           hasEmissionTex;
    int                   envmapIsHDR;

    // 透射率/透明度贴图（用于玻璃、冰等）
    uint8_t*      transmissionTexPixels;
    int           transmissionTexWidth;
    int           transmissionTexHeight;
    int           hasTransmissionTex;

    // 顶点纹理坐标
    float2*       texCoords;      ///< 指向顶点纹理坐标的设备指针
    int           hasTexCoords;   ///< 如果 texCoords 缓冲区有效则为1

    int           isCurve;        ///< 1 if this hit record is for a curve primitive
};

// ---------------------------------------------------------------------------
// Curve Data (custom primitives)
// ---------------------------------------------------------------------------
struct CurveData
{
    float3        cp[4];
    float         width0;
    float         width1;
    CurveTypeData type;
    float3        n0;
    float3        n1;
    int           hasNormals;
    int           materialIndex;
};

// ---------------------------------------------------------------------------
// 光源参数
// ---------------------------------------------------------------------------
enum class LightTypeData { Point, Directional, Spot };

struct LightData
{
    LightTypeData type;
    float3 position;
    float3 direction;
    float3 intensity;
    float  cosInner;
    float  cosOuter;
};

// ---------------------------------------------------------------------------
// 方向光结构体
// ---------------------------------------------------------------------------
struct DirectionalLight
{
    float3 direction;
    float3 intensity;
};

// ---------------------------------------------------------------------------
// 顶层启动参数（绑定到 PTX 中的 "params"）
// ---------------------------------------------------------------------------
struct LaunchParams
{
    CameraParams          camera;
    float4*               output;     ///< 线性HDR输出缓冲区（CUDA设备指针）
    float4*               accum;      ///< 累积缓冲区（CUDA设备指针）
    float4*               albedo;     ///< 降噪用albedo引导缓冲区
    float4*               normal;     ///< 降噪用法线引导缓冲区
    int                   width;
    int                   height;
    unsigned int          sampleIndex;///< 用于RNG种子的帧计数器
    int                   maxBounces;
    OptixTraversableHandle traversable;

    LightData*            lights;     ///< 指向光源数组的设备指针
    int                   numLights;
    float3                ambientIntensity;

    int                   debugDirectional;
    int                   debugMode;      ///< 0=完整路径追踪, 1=法线可视化, 2=直接光, 3=材质颜色

    DirectionalLight directionalLights[8]; // 支持最多8个方向光
    int numDirectionalLights;

    // 环境贴图
    uint8_t*              envmapPixels;   ///< RGBA环境贴图像素
    float4*               envmapPixelsF;  ///< HDR环境贴图像素 (float4, linear)
    int                   envmapWidth;
    int                   envmapHeight;
    int                   hasEnvmap;
    int                   envmapIsHDR;
    float3                environmentScale; ///< 环境贴图强度倍数 (RGB)
    float3                envLightRight;  ///< 环境光旋转矩阵：右向量
    float3                envLightUp;     ///< 环境光旋转矩阵：上向量
    float3                envLightFwd;    ///< 环境光旋转矩阵：前向量

    // Curves
    CurveData*            curves;
    int                   numCurves;
};
