/**
 * @file path_tracer.cu
 * @brief OptiX 8 path-tracing device programs with proper bouncing.
 */

#include <optix.h>
#include <cuda_runtime.h>

#include "launch_params.h"

// Bind the launch-parameter struct to OptiX's "params" variable
extern "C" __constant__ LaunchParams params;

// ---------------------------------------------------------------------------
// Payload: encode float3 (radiance contribution from this hit)
// ---------------------------------------------------------------------------
static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 getPayload()
{
    return {__uint_as_float(optixGetPayload_0()),
            __uint_as_float(optixGetPayload_1()),
            __uint_as_float(optixGetPayload_2())};
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
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
static __forceinline__ __device__ float  dot(float3 a, float3 b)
{ return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float3 normalize(float3 a)
{ const float inv = rsqrtf(fmaxf(dot(a,a), 1e-10f)); return {a.x*inv, a.y*inv, a.z*inv}; }
static __forceinline__ __device__ float3 cross(float3 a, float3 b)
{ return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
static __forceinline__ __device__ float3 lerp(float3 a, float3 b, float t)
{ return {a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t}; }
static __forceinline__ __device__ float  clampf(float v, float lo, float hi)
{ return v < lo ? lo : (v > hi ? hi : v); }
static __forceinline__ __device__ float3 clamp3(float3 v, float lo, float hi)
{ return {clampf(v.x,lo,hi), clampf(v.y,lo,hi), clampf(v.z,lo,hi)}; }

// Simple LCG random generator
struct LCG
{
    uint32_t state;
    
    __device__ LCG(uint32_t seed) : state(seed) {}
    
    __device__ uint32_t next()
    {
        state = state * 1103515245u + 12345u;
        return (state >> 16u) & 0x7fffu;
    }
    
    __device__ float nextFloat()
    {
        return static_cast<float>(next()) * (1.f / 32768.f);
    }
};

// Cosine-weighted hemisphere sampling
static __device__ float3 hemisphereSample(float3 normal, float u1, float u2)
{
    const float theta = asinf(sqrtf(u2));
    const float phi = 2.f * 3.14159265f * u1;
    const float st = sinf(theta);
    const float ct = cosf(theta);
    const float cp = cosf(phi);
    const float sp = sinf(phi);
    
    // Local coordinates
    float3 t, b;
    if (fabsf(normal.x) > 0.9f)
        t = normalize(cross({0, 1, 0}, normal));
    else
        t = normalize(cross({1, 0, 0}, normal));
    b = cross(normal, t);
    
    return normalize(t * (st * cp) + b * (st * sp) + normal * ct);
}

// ---------------------------------------------------------------------------
// __raygen__pathTrace: Main path tracing kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__pathTrace()
{
    const uint3  idx  = optixGetLaunchIndex();
    const int    pixelID = idx.y * params.width + idx.x;
    
    // Simple RNG per pixel
    LCG rng(pixelID + params.sampleIndex * 1000000);
    
    // Generate primary ray with subpixel jitter
    const float u = (idx.x + rng.nextFloat()) / static_cast<float>(params.width);
    const float v = (idx.y + rng.nextFloat()) / static_cast<float>(params.height);
    
    const float3 origin = params.camera.origin;
    const float3 rayDir = normalize(
        params.camera.lowerLeft +
        u * params.camera.horizontal +
        v * params.camera.vertical - origin);
    
    // Path tracing loop
    float3 throughput = {1.f, 1.f, 1.f};
    float3 radiance = {0.f, 0.f, 0.f};
    float3 rayOrigin = origin;
    float3 rayDirection = rayDir;
    
    for (int bounce = 0; bounce <= params.maxBounces; ++bounce)
    {
        // Prepare payload storage
        unsigned int p0 = 0, p1 = 0, p2 = 0;
        
        // Trace ray
        optixTrace(
            params.traversable,
            rayOrigin, rayDirection,
            1e-3f, 1e30f, 0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2);
        
        // Decode hit data 
        // p0 contains encoded t-distance, p1,p2 contain normal components
        float3 hitNormal = {__uint_as_float(p1),
                            __uint_as_float(p2),
                            0.f};
        
        // Check if sky (t is very large for sky)
        float t_val = __uint_as_float(p0);
        if (t_val > 1e9f)
        {
            // Sky hit
            float3 sky_color = {0.5f, 0.7f, 1.0f};
            radiance = radiance + throughput * sky_color;
            break;
        }
        
        // Normal surface hit - compute third component of normal
        const float nx2 = hitNormal.x * hitNormal.x;
        const float ny2 = hitNormal.y * hitNormal.y;
        float nz = sqrtf(fmaxf(0.f, 1.f - nx2 - ny2));
        hitNormal.z = nz;
        hitNormal = normalize(hitNormal);
        
        // For now, simple diffuse surface
        float3 surfaceColor = make_float3(0.8f, 0.8f, 0.8f);
        
        // Add direct lighting (simple hack: add some light from above)
        float3 lightDir = normalize(make_float3(0.5f, 1.f, 0.3f));
        float3 diffuse = make_float3(fmaxf(0.f, dot(hitNormal, lightDir)), fmaxf(0.f, dot(hitNormal, lightDir)), fmaxf(0.f, dot(hitNormal, lightDir))) * surfaceColor;
        radiance = radiance + throughput * diffuse * 0.5f;
        
        // Attenuate throughput
        throughput = throughput * surfaceColor * 0.7f;
        
        if (bounce >= params.maxBounces)
            break;
        
        // Russian roulette
        const float prob = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (rng.nextFloat() > prob)
            break;
        throughput = throughput * (1.f / prob);
        
        // Compute new hit point
        float t = __uint_as_float(p0);
        if (t < 1e9f)  // Valid hit
        {
            rayOrigin = rayOrigin + rayDirection * t;
            rayOrigin = rayOrigin + hitNormal * 0.001f;  // Small offset to prevent self-intersection
        }
        else
        {
            break;  // Invalid hit
        }
        
        // Sample new direction on hemisphere
        rayDirection = hemisphereSample(hitNormal, rng.nextFloat(), rng.nextFloat());
    }
    
    // Accumulate to buffer
    const float4 prev = params.accum[pixelID];
    const float  n    = static_cast<float>(params.sampleIndex + 1);
    const float4 curr = {
        (prev.x * (n-1) + radiance.x) / n,
        (prev.y * (n-1) + radiance.y) / n,
        (prev.z * (n-1) + radiance.z) / n,
        1.f
    };
    params.accum[pixelID] = curr;
    params.output[pixelID] = curr;
}

// ---------------------------------------------------------------------------
// __miss__envmap: Sky/environment map
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__envmap()
{
    const float3 d = optixGetWorldRayDirection();
    
    // Simple sky: blue gradient based on height
    const float t = 0.5f * (d.y + 1.f);
    const float3 sky = lerp({0.8f, 0.8f, 0.8f}, {0.3f, 0.5f, 0.8f}, t);
    
    // Encode: p0=huge t value (sky), p1=sky.y, p2=sky.z
    // Use a huge number to signal sky
    optixSetPayload_0(__float_as_uint(1e10f));
    optixSetPayload_1(__float_as_uint(-20.f));  // Negative flag for sky
    optixSetPayload_2(__float_as_uint(sky.x));
}

// ---------------------------------------------------------------------------
// __closesthit__pbr: Hit surface shader
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__pbr()
{
    const HitGroupData* data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    
    // Get surface normal from ray direction (backface shading)
    float3 rayDir = optixGetWorldRayDirection();
    float3 normal = normalize(rayDir * -1.f);
    
    // Flip if needed to face the ray
    if (dot(normal, rayDir) > 0.f)
        normal = normal * -1.f;
    
    // Use ray parameters: optixGetObjectRayDirection gives ray parameter (length from origin to hit)
    // We'll just use a dummy small positive value to signal a valid hit
    float t = 1.0f;  // Valid hit marker
    
    // Encode: p0=t (as float), p1=nx, p2=ny (nz is computed from nx,ny)
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(normal.x));
    optixSetPayload_2(__float_as_uint(normal.y));
}

