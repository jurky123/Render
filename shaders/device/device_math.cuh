#pragma once

#include <cuda_runtime.h>

// float3 arithmetic operators
static __forceinline__ __device__ float3 operator+(float3 a, float3 b)
{ return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static __forceinline__ __device__ float3 operator-(float3 a, float3 b)
{ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static __forceinline__ __device__ float3 operator*(float3 a, float3 b)
{ return {a.x*b.x, a.y*b.y, a.z*b.z}; }
static __forceinline__ __device__ float3 operator*(float s, float3 a)
{ return {s*a.x, s*a.y, s*a.z}; }
static __forceinline__ __device__ float3 operator*(float3 a, float s)
{ return {a.x*s, a.y*s, a.z*s}; }
static __forceinline__ __device__ float3 operator/(float3 a, float s)
{ return {a.x/s, a.y/s, a.z/s}; }
static __forceinline__ __device__ float3 operator-(float3 a)
{ return {-a.x, -a.y, -a.z}; }
static __forceinline__ __device__ float3& operator+=(float3& a, float3 b)
{ a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
static __forceinline__ __device__ float3& operator*=(float3& a, float s)
{ a.x*=s; a.y*=s; a.z*=s; return a; }

// float3 math functions
static __forceinline__ __device__ float dot(float3 a, float3 b)
{ return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float length(float3 a)
{ return sqrtf(dot(a, a)); }
static __forceinline__ __device__ float3 normalize(float3 a)
{ const float inv = rsqrtf(fmaxf(dot(a,a), 1e-10f)); return {a.x*inv, a.y*inv, a.z*inv}; }
static __forceinline__ __device__ float3 cross(float3 a, float3 b)
{ return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
static __forceinline__ __device__ float3 lerp(float3 a, float3 b, float t)
{ return {a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t}; }
static __forceinline__ __device__ float lerpf(float a, float b, float t)
{ return a + (b - a) * t; }
static __forceinline__ __device__ float clampf(float v, float lo, float hi)
{ return v < lo ? lo : (v > hi ? hi : v); }
static __forceinline__ __device__ float3 clamp3(float3 v, float lo, float hi)
{ return {clampf(v.x,lo,hi), clampf(v.y,lo,hi), clampf(v.z,lo,hi)}; }

// Rendering utilities
static __forceinline__ __device__ float3 reflectVec(float3 v, float3 n)
{ return v - 2.f * dot(v, n) * n; }

static __forceinline__ __device__ float3 safeNormalize(float3 v)
{ return normalize(v); }

static __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 f0)
{
    float t = fmaxf(0.f, 1.f - cosTheta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return f0 + (make_float3(1.f, 1.f, 1.f) - f0) * t5;
}

static __forceinline__ __device__ float ggxD(float NoH, float alpha)
{
    float a2 = alpha * alpha;
    float d = (NoH * NoH) * (a2 - 1.f) + 1.f;
    return a2 / (3.14159265f * d * d);
}
