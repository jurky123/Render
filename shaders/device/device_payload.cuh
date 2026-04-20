#pragma once

struct ShadingState
{
    float t;
    float3 normal;
    unsigned int materialType;
};

static __forceinline__ __device__ void payloadSetShadowBlocked(float visibility)
{
    optixSetPayload_0(__float_as_uint(visibility));
}

static __forceinline__ __device__ void payloadSetMiss()
{
    optixSetPayload_0(__float_as_uint(1e10f));
    optixSetPayload_1(__float_as_uint(0.f));
    optixSetPayload_2(__float_as_uint(0.f));
    optixSetPayload_3(__float_as_uint(0.f));
    optixSetPayload_4(__float_as_uint(0.f));
    optixSetPayload_5(__float_as_uint(0.f));
    optixSetPayload_6(__float_as_uint(0.f));
    optixSetPayload_7(__float_as_uint(0.f));
    optixSetPayload_8(__float_as_uint(0.f));
}

static __forceinline__ __device__ void payloadSetHitHeader(float t, float3 normal, unsigned int materialType)
{
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(normal.x));
    optixSetPayload_2(__float_as_uint(normal.y));
    optixSetPayload_3(__float_as_uint(normal.z));
    optixSetPayload_4(materialType);
}
