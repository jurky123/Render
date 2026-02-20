#pragma once

/**
 * @file launch_params.h
 * @brief Structures shared between the host (C++) and device (CUDA/OptiX) code.
 *
 * Keep this header free of C++ STL types so that it can be included from
 * both .cpp and .cu translation units.
 */

#ifdef __CUDACC__
#  include <vector_types.h>
#  include <optix_types.h>
#else
// CPU-side stubs so the header compiles in non-CUDA TUs
#  include <cstdint>
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
static inline float3 make_float3(float x, float y, float z) { return {x,y,z}; }
static inline float4 make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
using OptixTraversableHandle = unsigned long long;
#endif

// ---------------------------------------------------------------------------
// Camera parameters (all in world space)
// ---------------------------------------------------------------------------
struct CameraParams
{
    float3 origin;
    float3 lowerLeft;   ///< lower-left corner of the image plane
    float3 horizontal;  ///< full horizontal span of the image plane
    float3 vertical;    ///< full vertical span of the image plane
};

// ---------------------------------------------------------------------------
// Per-mesh material data stored in the hit-group SBT record
// ---------------------------------------------------------------------------
struct HitGroupData
{
    float3 baseColor;
    float  metallic;
    float  roughness;
    float3 emission;
    float  ior;
    float  transmission;
};

// ---------------------------------------------------------------------------
// Top-level launch parameters (bound to "params" in PTX)
// ---------------------------------------------------------------------------
struct LaunchParams
{
    CameraParams          camera;
    float4*               output;     ///< RGBA output buffer (CUDA device ptr)
    float4*               accum;      ///< accumulation buffer (CUDA device ptr)
    int                   width;
    int                   height;
    unsigned int          sampleIndex;///< frame counter for RNG seeding
    int                   maxBounces;
    OptixTraversableHandle traversable;
};
