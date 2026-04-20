/**
 * @file path_tracer.cu
 * @brief OptiX 8 路径追踪设备程序，支持完整的光线反弹
 */

#include <optix.h>
#include <cuda_runtime.h>

#include "launch_params.h"
#include "device/device_payload.cuh"
#include "device/device_math.cuh"
#include "device/device_texture.cuh"
#include "device/device_material.cuh"
#include "device/device_curve.cuh"
#include "device/device_closesthit.cuh"

// 将启动参数结构体绑定到 OptiX 的 "params" 变量
extern "C" __constant__ LaunchParams params;

// 调试模式：0=完整路径追踪, 1=可视化法线, 2=可视化直接光照, 3=可视化基础颜色
// 注意：debugMode 现在是运行时设置的 - 见 launch_params.h

// ---------------------------------------------------------------------------
static __forceinline__ __device__ float ggxG1(float NoV, float alpha)
{
    float a2 = alpha * alpha;
    float denom = NoV + sqrtf(a2 + (1.f - a2) * NoV * NoV);
    return (2.f * NoV) / fmaxf(denom, 1e-6f);
}

static __forceinline__ __device__ float ggxG(float NoV, float NoL, float alpha)
{
    return ggxG1(NoV, alpha) * ggxG1(NoL, alpha);
}

static __forceinline__ __device__ void buildTangentBasis(float3 n, float3& t, float3& b)
{
    if (fabsf(n.x) > 0.9f)
        t = normalize(cross({0.f, 1.f, 0.f}, n));
    else
        t = normalize(cross({1.f, 0.f, 0.f}, n));
    b = cross(n, t);
}

static __forceinline__ __device__ float3 sampleGGX(float3 n, float alpha, float u1, float u2)
{
    float phi = 2.f * 3.14159265f * u1;
    float cosTheta = sqrtf((1.f - u2) / (1.f + (alpha * alpha - 1.f) * u2));
    float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));

    float3 t, b;
    buildTangentBasis(n, t, b);

    float3 h = normalize(t * (sinTheta * cosf(phi)) + b * (sinTheta * sinf(phi)) + n * cosTheta);
    return h;
}

// Forward declaration
static __device__ float3 sampleTexture(const uint8_t* pixels, int width, int height, float2 uv);
static __device__ float3 sampleTextureHDR(const float4* pixels, int width, int height, float2 uv);
static __device__ float3 sampleTextureFloat(const float4* pixels, int width, int height, float2 uv);

static __forceinline__ __device__ float3 sampleEnvmap(float3 dir)
{
    if (!params.hasEnvmap || params.envmapWidth <= 0 || params.envmapHeight <= 0)
        return make_float3(0.f, 0.f, 0.f);

    dir = normalize(dir);
    
    // Temporarily disable light transform to debug whiteness issue
    // float3 transformedDir;
    // transformedDir.x = dot(dir, params.envLightRight);
    // transformedDir.y = dot(dir, params.envLightUp);
    // transformedDir.z = dot(dir, params.envLightFwd);
    // dir = normalize(transformedDir);
    
    float u = atan2f(-dir.z, dir.x) * (0.5f / 3.14159265f) + 0.5f;
    float v = acosf(clampf(dir.y, -1.f, 1.f)) * (1.f / 3.14159265f);
    
    // HDR path
    if (params.envmapIsHDR) {
        if (!params.envmapPixelsF)
            return make_float3(0.f, 0.f, 0.f);
        return sampleTextureHDR(params.envmapPixelsF, params.envmapWidth, params.envmapHeight, make_float2(u, v));
    }
    
    // LDR path
    if (!params.envmapPixels)
        return make_float3(0.f, 0.f, 0.f);
    return sampleTexture(params.envmapPixels, params.envmapWidth, params.envmapHeight, make_float2(u, v));
}

// ---------------------------------------------------------------------------
// pbrt-v4 Material BRDFs
// ---------------------------------------------------------------------------

// Oren-Nayar Diffuse BRDF (for rough diffuse surfaces)
static __forceinline__ __device__ float3 orenNayarBRDF(float3 wo, float3 wi, float3 n, float sigma)
{
    if (sigma == 0.f) {
        // Lambertian fallback
        return make_float3(1.f / 3.14159265f, 1.f / 3.14159265f, 1.f / 3.14159265f);
    }
    
    float sinThetaI = sqrtf(fmaxf(0.f, 1.f - dot(wi, n) * dot(wi, n)));
    float sinThetaO = sqrtf(fmaxf(0.f, 1.f - dot(wo, n) * dot(wo, n)));
    
    float maxCos = 0.f;
    if (sinThetaI > 1e-4f && sinThetaO > 1e-4f) {
        float3 wi_perp = normalize(wi - dot(wi, n) * n);
        float3 wo_perp = normalize(wo - dot(wo, n) * n);
        float dCos = dot(wi_perp, wo_perp);
        maxCos = fmaxf(0.f, dCos);
    }
    
    float sinAlpha, tanBeta;
    float absCosI = fabsf(dot(wi, n));
    float absCosO = fabsf(dot(wo, n));
    if (absCosI > absCosO) {
        sinAlpha = sinThetaO;
        tanBeta = sinThetaI / absCosI;
    } else {
        sinAlpha = sinThetaI;
        tanBeta = sinThetaO / absCosO;
    }
    
    float sigma2 = sigma * sigma;
    float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
    float B = 0.45f * sigma2 / (sigma2 + 0.09f);
    
    float rho = (1.f / 3.14159265f) * (A + B * maxCos * sinAlpha * tanBeta);
    return make_float3(rho, rho, rho);
}

// Conductor Fresnel (complex IOR)
static __forceinline__ __device__ float3 conductorFresnel(float cosTheta, float3 eta, float3 k)
{
    float cosThetaSq = cosTheta * cosTheta;
    float sinThetaSq = 1.f - cosThetaSq;
    float3 etaSq = eta * eta;
    float3 kSq = k * k;
    
    float3 t0 = etaSq - kSq - make_float3(sinThetaSq, sinThetaSq, sinThetaSq);
    float3 aSqplusbSq = make_float3(
        sqrtf(t0.x * t0.x + 4.f * etaSq.x * kSq.x),
        sqrtf(t0.y * t0.y + 4.f * etaSq.y * kSq.y),
        sqrtf(t0.z * t0.z + 4.f * etaSq.z * kSq.z)
    );
    float3 a = make_float3(
        sqrtf(0.5f * (aSqplusbSq.x + t0.x)),
        sqrtf(0.5f * (aSqplusbSq.y + t0.y)),
        sqrtf(0.5f * (aSqplusbSq.z + t0.z))
    );
    
    float3 term1 = aSqplusbSq + make_float3(cosThetaSq, cosThetaSq, cosThetaSq);
    float3 term2 = 2.f * a * cosTheta;
    // Component-wise division for float3
    float3 Rs = make_float3(
        (term1.x - term2.x) / (term1.x + term2.x),
        (term1.y - term2.y) / (term1.y + term2.y),
        (term1.z - term2.z) / (term1.z + term2.z)
    );
    
    float3 term3 = aSqplusbSq * cosThetaSq + make_float3(sinThetaSq * sinThetaSq, sinThetaSq * sinThetaSq, sinThetaSq * sinThetaSq);
    float3 term4 = term2 * sinThetaSq;
    // Component-wise division for float3
    float3 Rp = make_float3(
        Rs.x * ((term3.x - term4.x) / (term3.x + term4.x)),
        Rs.y * ((term3.y - term4.y) / (term3.y + term4.y)),
        Rs.z * ((term3.z - term4.z) / (term3.z + term4.z))
    );
    
    return 0.5f * (Rs + Rp);
}

// Dielectric Fresnel (single IOR)
static __forceinline__ __device__ float dielectricFresnel(float cosThetaI, float eta)
{
    cosThetaI = clampf(cosThetaI, -1.f, 1.f);
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        eta = 1.f / eta;
        cosThetaI = fabsf(cosThetaI);
    }
    
    float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = sinThetaI / eta;
    
    if (sinThetaT >= 1.f) return 1.f;  // Total internal reflection
    
    float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
    
    float Rparl = ((eta * cosThetaI) - cosThetaT) / ((eta * cosThetaI) + cosThetaT);
    float Rperp = (cosThetaI - (eta * cosThetaT)) / (cosThetaI + (eta * cosThetaT));
    
    return 0.5f * (Rparl * Rparl + Rperp * Rperp);
}

// Roughness remapping (pbrt-v4 style)
static __forceinline__ __device__ float remapRoughness(float r)
{
    r = fmaxf(r, 1e-3f);
    float x = logf(r);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 
           0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

// Cosine-weighted hemisphere sampling
static __device__ float3 hemisphereSample(float3 normal, float u1, float u2)
{
    const float theta = asinf(sqrtf(u2));
    const float phi = 2.f * 3.14159265f * u1;
    const float st = sinf(theta);
    const float ct = cosf(theta);
    const float cp = cosf(phi);
    const float sp = sinf(phi);
    
    // Build local coordinate system
    float3 t, b;
    if (fabsf(normal.x) > 0.9f)
        t = normalize(cross({0, 1, 0}, normal));
    else
        t = normalize(cross({1, 0, 0}, normal));
    b = cross(normal, t);
    
    return normalize(t * (st * cp) + b * (st * sp) + normal * ct);
}

// ---------------------------------------------------------------------------
// Material Evaluation and Sampling (pbrt-v4 unified interface)
// ---------------------------------------------------------------------------

struct MaterialSample {
    float3 wi;           // Sampled incident direction
    float3 f;            // BRDF/BSDF value
    float pdf;           // Probability density
    bool isDelta;        // Is this a delta distribution (mirror/glass)?
};

// Evaluate Diffuse BRDF (with optional Oren-Nayar)
static __device__ float3 evaluateDiffuse(float3 wo, float3 wi, float3 n, float3 albedo, float sigma)
{
    if (dot(n, wi) <= 0.f || dot(n, wo) <= 0.f) return make_float3(0.f, 0.f, 0.f);
    
    float3 brdf = orenNayarBRDF(wo, wi, n, sigma);
    return albedo * brdf;
}

// Sample Diffuse BRDF
static __device__ MaterialSample sampleDiffuse(float3 wo, float3 n, float3 albedo, float sigma, float u1, float u2)
{
    MaterialSample ms;
    ms.isDelta = false;
    
    // Cosine-weighted hemisphere sampling
    float3 wi = hemisphereSample(n, u1, u2);
    float NoL = dot(n, wi);
    
    if (NoL <= 0.f) {
        ms.f = make_float3(0.f, 0.f, 0.f);
        ms.pdf = 0.f;
        ms.wi = wi;
        return ms;
    }
    
    ms.wi = wi;
    ms.f = evaluateDiffuse(wo, wi, n, albedo, sigma);
    ms.pdf = NoL * (1.f / 3.14159265f);
    return ms;
}

// Evaluate Conductor BRDF (microfacet with complex Fresnel)
static __device__ float3 evaluateConductor(float3 wo, float3 wi, float3 n, float3 eta, float3 k, 
                                            float urough, float vrough, bool remap)
{
    if (dot(n, wi) <= 0.f || dot(n, wo) <= 0.f) return make_float3(0.f, 0.f, 0.f);
    
    float alpha = remap ? remapRoughness(urough) : urough;
    alpha = fmaxf(alpha, 1e-3f);
    
    float3 wh = normalize(wo + wi);
    float NoH = fmaxf(dot(n, wh), 0.f);
    float NoV = fmaxf(dot(n, wo), 0.f);
    float NoL = fmaxf(dot(n, wi), 0.f);
    float VoH = fmaxf(dot(wo, wh), 0.f);
    
    float D = ggxD(NoH, alpha);
    float G = ggxG(NoV, NoL, alpha);
    float3 F = conductorFresnel(VoH, eta, k);
    
    return F * (D * G / fmaxf(4.f * NoV * NoL, 1e-6f));
}

// Sample Conductor BRDF
static __device__ MaterialSample sampleConductor(float3 wo, float3 n, float3 eta, float3 k,
                                                  float urough, float vrough, bool remap, 
                                                  float u1, float u2)
{
    MaterialSample ms;
    ms.isDelta = false;
    
    float alpha = remap ? remapRoughness(urough) : urough;
    alpha = fmaxf(alpha, 1e-3f);
    
    float3 wh = sampleGGX(n, alpha, u1, u2);
    float3 wi = reflectVec(-wo, wh);
    
    if (dot(n, wi) <= 0.f) {
        ms.f = make_float3(0.f, 0.f, 0.f);
        ms.pdf = 0.f;
        ms.wi = wi;
        return ms;
    }
    
    ms.wi = wi;
    ms.f = evaluateConductor(wo, wi, n, eta, k, urough, vrough, remap);
    
    float NoH = fmaxf(dot(n, wh), 0.f);
    float VoH = fmaxf(dot(wo, wh), 0.f);
    float D = ggxD(NoH, alpha);
    ms.pdf = (D * NoH) / fmaxf(4.f * VoH, 1e-6f);
    
    return ms;
}

// Evaluate Dielectric BRDF/BTDF
static __device__ float3 evaluateDielectric(float3 wo, float3 wi, float3 n, float eta, bool reflect)
{
    if (reflect) {
        // Reflection lobe
        if (dot(n, wi) <= 0.f || dot(n, wo) <= 0.f) return make_float3(0.f, 0.f, 0.f);
        float F = dielectricFresnel(dot(n, wo), eta);
        return make_float3(F, F, F);
    } else {
        // Transmission lobe
        return make_float3(1.f, 1.f, 1.f);
    }
}

// Sample Dielectric BRDF/BTDF (specular only for now)
static __device__ MaterialSample sampleDielectric(float3 wo, float3 n, float eta, float u)
{
    MaterialSample ms;
    ms.isDelta = true;
    
    float cosTheta = dot(n, wo);
    float F = dielectricFresnel(cosTheta, eta);
    
    if (u < F) {
        // Reflection
        ms.wi = reflectVec(-wo, n);
        ms.f = make_float3(F, F, F);
        ms.pdf = F;
    } else {
        // Refraction
        bool entering = cosTheta > 0.f;
        float etaRatio = entering ? (1.f / eta) : eta;
        float3 refractN = entering ? n : (n * -1.f);
        
        float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
        float sinThetaT = etaRatio * sinThetaI;
        
        if (sinThetaT >= 1.f) {
            // Total internal reflection
            ms.wi = reflectVec(-wo, n);
            ms.f = make_float3(1.f, 1.f, 1.f);
            ms.pdf = 1.f;
        } else {
            float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
            ms.wi = normalize(wo * (-etaRatio) + refractN * (etaRatio * fabsf(cosTheta) - cosThetaT));
            ms.f = make_float3(1.f - F, 1.f - F, 1.f - F);
            ms.pdf = 1.f - F;
        }
    }
    
    return ms;
}

// Sample RoughDielectric (GGX microfacet with refraction)
static __device__ MaterialSample sampleRoughDielectric(float3 wo, float3 n, float eta,
                                                         float urough, float vrough, bool remap,
                                                         float u, float u1, float u2)
{
    MaterialSample ms;
    ms.isDelta = false;
    
    float alpha = remap ? remapRoughness(urough) : urough;
    alpha = fmaxf(alpha, 1e-3f);
    
    float3 wh = sampleGGX(n, alpha, u1, u2);
    float cosTheta = dot(wo, wh);
    float F = dielectricFresnel(cosTheta, eta);
    
    if (u < F) {
        // Reflection
        float3 wi = reflectVec(-wo, wh);
        if (dot(n, wi) <= 0.f) {
            ms.f = make_float3(0.f, 0.f, 0.f);
            ms.pdf = 0.f;
            ms.wi = wi;
            return ms;
        }
        
        float NoV = fmaxf(dot(n, wo), 0.f);
        float NoL = fmaxf(dot(n, wi), 0.f);
        float NoH = fmaxf(dot(n, wh), 0.f);
        float VoH = fmaxf(dot(wo, wh), 0.f);
        
        float D = ggxD(NoH, alpha);
        float G = ggxG(NoV, NoL, alpha);
        
        ms.wi = wi;
        ms.f = make_float3(F, F, F) * (D * G / fmaxf(4.f * NoV * NoL, 1e-6f));
        ms.pdf = F * (D * NoH) / fmaxf(4.f * VoH, 1e-6f);
    } else {
        // Refraction
        bool entering = dot(n, wo) > 0.f;
        float etaRatio = entering ? (1.f / eta) : eta;
        float3 refractN = entering ? n : (n * -1.f);
        
        float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
        float sinThetaT = etaRatio * sinThetaI;
        
        if (sinThetaT >= 1.f) {
            ms.f = make_float3(0.f, 0.f, 0.f);
            ms.pdf = 0.f;
            ms.wi = make_float3(0.f, 0.f, 0.f);
        } else {
            float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
            float3 wi = normalize(wo * (-etaRatio) + refractN * (etaRatio * fabsf(cosTheta) - cosThetaT));
            
            ms.wi = wi;
            ms.f = make_float3(1.f - F, 1.f - F, 1.f - F);
            ms.pdf = 1.f - F;
        }
    }
    
    return ms;
}

// Sample CoatedDiffuse (clearcoat over diffuse)
static __device__ MaterialSample sampleCoatedDiffuse(float3 wo, float3 n, float3 albedo,
                                                      float interfaceEta, float interfaceRough,
                                                      float u, float u1, float u2)
{
    MaterialSample ms;
    
    float F = dielectricFresnel(dot(n, wo), interfaceEta);
    
    if (u < F) {
        // Sample clearcoat reflection
        float alpha = fmaxf(interfaceRough, 1e-3f);
        float3 wh = sampleGGX(n, alpha, u1, u2);
        float3 wi = reflectVec(-wo, wh);
        
        if (dot(n, wi) <= 0.f) {
            ms.f = make_float3(0.f, 0.f, 0.f);
            ms.pdf = 0.f;
            ms.wi = wi;
            ms.isDelta = false;
            return ms;
        }
        
        float NoV = fmaxf(dot(n, wo), 0.f);
        float NoL = fmaxf(dot(n, wi), 0.f);
        float NoH = fmaxf(dot(n, wh), 0.f);
        float VoH = fmaxf(dot(wo, wh), 0.f);
        
        float D = ggxD(NoH, alpha);
        float G = ggxG(NoV, NoL, alpha);
        
        ms.wi = wi;
        ms.f = make_float3(F, F, F) * (D * G / fmaxf(4.f * NoV * NoL, 1e-6f));
        ms.pdf = F * (D * NoH) / fmaxf(4.f * VoH, 1e-6f);
        ms.isDelta = false;
    } else {
        // Sample diffuse base layer
        ms = sampleDiffuse(wo, n, albedo * (1.f - F), 0.f, u1, u2);
        ms.f = ms.f * (1.f - F);
        ms.pdf = ms.pdf * (1.f - F);
    }
    
    return ms;
}

// CoatedConductor: Clearcoat interface over conductor base (complex Fresnel)
__device__ MaterialSample sampleCoatedConductor(
    float3 wo, float3 n,
    float3 eta, float3 k,              // Conductor base layer optical properties
    float interfaceIOR,                // Clearcoat interface refractive index
    float interfaceURoughness,         // Clearcoat u-roughness
    float interfaceVRoughness,         // Clearcoat v-roughness
    int shouldRemapRoughness,          // Whether to remap roughness
    float baseURoughness,              // Conductor base u-roughness
    float baseVRoughness,              // Conductor base v-roughness
    float u0, float u1, float u2)
{
    MaterialSample ms;
    ms.wi = make_float3(0.f, 0.f, 0.f);
    ms.f = make_float3(0.f, 0.f, 0.f);
    ms.pdf = 0.f;
    ms.isDelta = false;
    
    // Fresnel for dielectric interface (1.0 -> interfaceIOR)
    float cosTheta = fmaxf(dot(wo, n), 0.f);
    float F = dielectricFresnel(cosTheta, interfaceIOR);
    
    // Sample interface or base layer based on Fresnel
    if (u0 < F) {
        // Sample clearcoat interface (dielectric specular reflection)
        // Apply roughness remapping if requested
        float alpha_u = shouldRemapRoughness ? remapRoughness(interfaceURoughness) : interfaceURoughness;
        float alpha_v = shouldRemapRoughness ? remapRoughness(interfaceVRoughness) : interfaceVRoughness;
        float alpha = (alpha_u + alpha_v) * 0.5f; // isotropic approximation
        
        float3 H = sampleGGX(n, alpha, u1, u2);
        ms.wi = reflectVec(-wo, H);
        
        float NoL = fmaxf(dot(n, ms.wi), 0.f);
        float NoV = fmaxf(dot(n, wo), 0.f);
        float NoH = fmaxf(dot(n, H), 0.f);
        float VoH = fmaxf(dot(wo, H), 0.f);
        
        if (NoL <= 0.f || NoV <= 0.f) return ms;
        
        float D = ggxD(NoH, alpha);
        float G = ggxG(NoV, NoL, alpha);
        
        ms.f = make_float3(F, F, F) * (D * G / fmaxf(4.f * NoV * NoL, 1e-6f));
        ms.pdf = F * (D * NoH) / fmaxf(4.f * VoH, 1e-6f);
        ms.isDelta = false;
    } else {
        // Sample conductor base layer (complex Fresnel)
        MaterialSample baseSample = sampleConductor(wo, n, eta, k, baseURoughness, baseVRoughness, remapRoughness, u1, u2);
        
        // Attenuate by transmission through interface: (1 - F)
        ms.wi = baseSample.wi;
        ms.f = baseSample.f * (1.f - F);
        ms.pdf = baseSample.pdf * (1.f - F);
        ms.isDelta = baseSample.isDelta;
    }
    
    return ms;
}

// 改进的 PCG 随机数生成器
struct RNG
{
    uint32_t state;
    
    __device__ RNG(uint32_t seed) 
    {
        // 使用 Wang hash 混合初始种子
        seed = (seed ^ 61u) ^ (seed >> 16u);
        seed *= 9u;
        seed = seed ^ (seed >> 4u);
        seed *= 0x27d4eb2du;
        seed = seed ^ (seed >> 15u);
        state = seed;
    }
    
    __device__ uint32_t next()
    {
        // PCG-XSH-RR
        uint32_t oldstate = state;
        state = oldstate * 747796405u + 2891336453u;
        uint32_t word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
        return (word >> 22u) ^ word;
    }
    
    __device__ float nextFloat()
    {
        return static_cast<float>(next()) * (1.f / 4294967296.f);
    }
};

// 阴影光线可见性测试
// 如果光线路径未被遮挡（光源可见）则返回true
// 返回值: 0 = 遮挡(closesthit命中), 1 = 可见(miss), -1 = 未知
static __device__ int traceShadowRayDebug(float3 origin, float3 direction, float tmin, float tmax)
{
    // 使用特殊初始值以便调试
    uint32_t u0 = __float_as_uint(-999.0f);  // 初始值（不应该保持）
    uint32_t u1 = 0;
    uint32_t u2 = 0;
    uint32_t u3 = 0;
    uint32_t u4 = 0;
    uint32_t u5 = 0;
    uint32_t u6 = 0;
    uint32_t u7 = 0;
    uint32_t u8 = 0;
    
    optixTrace(params.traversable,
               origin,
               direction,
               tmin,              // 最小距离
               tmax,              // 最大距离
               0.0f,              // 光线时间
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,  // 首次命中即终止
               0,                 // SBT偏移
               1,                 // SBT步幅
               0,                 // miss SBT索引
               u0, u1, u2, u3, u4, u5, u6, u7, u8);
    
    float result = __uint_as_float(u0);
    // 0.25f = closesthit 命中 (遮挡)
    // 1.0f = miss (可见)
    // -999.0f = 未修改 (错误)
    if (result > 0.2f && result < 0.3f) return 0;  // closesthit -> 遮挡
    if (result > 0.9f && result < 1.1f) return 1;  // miss -> 可见
    return -1;  // 未知
}

// 阴影光线可见性测试（简化版，用于正常渲染）
static __device__ bool traceShadowRay(float3 origin, float3 direction, float tmin, float tmax)
{
    uint32_t u0 = __float_as_uint(0.0f);  // 初始为遮挡
    uint32_t u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0, u6 = 0, u7 = 0, u8 = 0;
    
    optixTrace(params.traversable,
               origin,
               direction,
               tmin, tmax, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 1, 0,
               u0, u1, u2, u3, u4, u5, u6, u7, u8);
    
    float result = __uint_as_float(u0);
    // closesthit 设置 0.25f，miss 设置 1.0f
    return (result > 0.5f);
}

// 方向光计算 - direction 指向光源
static __device__ float3 computeDirectionalLight(float3 normal, float3 albedo, float3 lightDir, float3 lightIntensity)
{
    // lightDir 是指向光源的方向
    float3 L = normalize(lightDir);
    float cosTheta = fmaxf(0.f, dot(normal, L));
    // 使用 Lambertian BRDF: albedo/π 乘以余弦项
    return albedo * lightIntensity * cosTheta * (0.5f / 3.14159265f);
}

// 点光源计算，带距离平方衰减
static __device__ float3 computePointLight(float3 hitPos, float3 normal, float3 albedo, 
                                            float3 lightPos, float3 lightIntensity)
{
    float3 toLightVec = lightPos - hitPos;
    float distSq = dot(toLightVec, toLightVec);
    float dist = sqrtf(distSq);
    float3 toLight = toLightVec / dist;
    
    float cosTheta = fmaxf(0.f, dot(normal, toLight));
    // 点光源使用平滑衰减：1 / (1 + 距离²)
    // 缩小强度，因为YAML中的值非常高
    float attenuation = 1.f / (1.f + distSq * distSq);  // 针对典型强度的缩放因子
    return albedo * lightIntensity * cosTheta * attenuation * (1.f / 3.14159265f);
}

// 全局命中结果缓冲：为每个线程存储详细的命中信息
struct HitResult
{
    float t;              // 命中距离
    float3 normal;        // 表面法线
    float3 albedo;        // 材质颜色
    int isMiss;           // 1表示未命中, 0表示命中
};
extern "C" __device__ __constant__ HitResult* g_hitResults = nullptr;

// ---------------------------------------------------------------------------
// __raygen__pathTrace: 主路径追踪内核，支持完整的光线反弹
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__pathTrace()
{
    const uint3  idx  = optixGetLaunchIndex();
    const int    pixelID = idx.y * params.width + idx.x;
    
    // 每像素的随机数生成器，使用更好的种子组合
    uint32_t seed = pixelID ^ (params.sampleIndex * 0x9E3779B9u) ^ (idx.x * 0x85EBCA6Bu) ^ (idx.y * 0xC2B2AE35u);
    RNG rng(seed);
    
    // 生成主光线
    const float u = (static_cast<float>(idx.x) + rng.nextFloat()) / static_cast<float>(params.width);
    const float v = (static_cast<float>(idx.y) + rng.nextFloat()) / static_cast<float>(params.height);
    
    float3 rayOrg = params.camera.origin;
    float3 rayDir = normalize(
        params.camera.lowerLeft +
        u * params.camera.horizontal +
        v * params.camera.vertical - rayOrg);
    
    float3 radiance = {0, 0, 0};
    float3 throughput = {1, 1, 1};
    float3 aovAlbedo = make_float3(0.f, 0.f, 0.f);
    float3 aovNormal = make_float3(0.f, 0.f, 1.f);
    int aovValid = 0;
    
    // 路径追踪主循环
    for (int bounce = 0; bounce < params.maxBounces; ++bounce)
    {
        // 追踪光线以找到交点 - 增加payload槽以传递emission, transmission, ior
        unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0, p13 = 0;
        optixTrace(
            params.traversable,
            rayOrg, rayDir,
            1e-3f, 1e30f, 0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,  // SBT偏移=0, SBT步幅=1 (光线类型数), miss SBT索引=0
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);

        // Decode hit result from payload
        float t = __uint_as_float(p0);
        float nx = __uint_as_float(p1);
        float ny = __uint_as_float(p2);
        float nz = __uint_as_float(p3);
        MaterialTypeData matType = static_cast<MaterialTypeData>(p4);
        
        // Decode material-specific parameters from p5-p13
        float3 albedo = make_float3(0.5f, 0.5f, 0.5f);
        float3 emission = make_float3(0.f, 0.f, 0.f);
        float sigma = 0.f;
        float3 eta = make_float3(1.5f, 1.5f, 1.5f);
        float3 k = make_float3(1.f, 1.f, 1.f);
        float uroughness = 0.1f;
        float vroughness = 0.1f;
        bool remapRoughness = true;
        float ior = 1.5f;
        float metallic = 0.f;
        float roughness = 0.5f;
        float transmission = 0.f;
        
        switch (matType) {
            case MaterialTypeData::Diffuse:
                albedo = make_float3(__uint_as_float(p5), __uint_as_float(p6), __uint_as_float(p7));
                sigma = __uint_as_float(p8);
                emission = make_float3(__uint_as_float(p9), __uint_as_float(p10), __uint_as_float(p11));
                break;
            
            case MaterialTypeData::Conductor:
                eta = make_float3(__uint_as_float(p5), __uint_as_float(p6), __uint_as_float(p7));
                k = make_float3(__uint_as_float(p8), __uint_as_float(p9), __uint_as_float(p10));
                uroughness = __uint_as_float(p11);
                vroughness = __uint_as_float(p12);
                remapRoughness = (__uint_as_float(p13) > 0.5f);
                break;
            
            case MaterialTypeData::Dielectric:
            case MaterialTypeData::RoughDielectric:
                ior = __uint_as_float(p5);
                uroughness = __uint_as_float(p6);
                vroughness = __uint_as_float(p7);
                remapRoughness = (__uint_as_float(p8) > 0.5f);
                emission = make_float3(__uint_as_float(p9), __uint_as_float(p10), __uint_as_float(p11));
                break;
            
            case MaterialTypeData::CoatedDiffuse:
                albedo = make_float3(__uint_as_float(p5), __uint_as_float(p6), __uint_as_float(p7));
                ior = __uint_as_float(p8);  // interface eta
                uroughness = __uint_as_float(p9);
                vroughness = __uint_as_float(p10);
                emission = make_float3(__uint_as_float(p11), __uint_as_float(p12), __uint_as_float(p13));
                break;
            
            case MaterialTypeData::CoatedConductor:
                eta = make_float3(__uint_as_float(p5), __uint_as_float(p6), __uint_as_float(p7));
                k = make_float3(__uint_as_float(p8), __uint_as_float(p9), __uint_as_float(p10));
                ior = __uint_as_float(p11);  // interface IOR
                uroughness = __uint_as_float(p12);  // interface uroughness
                vroughness = __uint_as_float(p13);  // interface vroughness
                // Note: using same roughness for base and interface due to payload slot limit
                break;
            
            case MaterialTypeData::BasicPBR:
            default:
                albedo = make_float3(__uint_as_float(p5), __uint_as_float(p6), __uint_as_float(p7));
                metallic = __uint_as_float(p8);
                roughness = __uint_as_float(p9);
                emission = make_float3(__uint_as_float(p10), __uint_as_float(p11), __uint_as_float(p12));
                transmission = __uint_as_float(p13);
                break;
        }
        
        // 检查是否未命中（天空）
        if (t > 1e9f)
        {
            float3 envColor = make_float3(0.f, 0.f, 0.f);
            if (params.hasEnvmap)
            {
                envColor = sampleEnvmap(rayDir) * params.environmentScale;
            }
            else
            {
                float skyT = 0.5f * (rayDir.y + 1.f);
                float3 skyColor = lerp({0.9f, 0.9f, 0.9f}, {0.3f, 0.5f, 0.8f}, skyT);
                envColor = skyColor * params.environmentScale;
            }
            radiance = radiance + throughput * envColor;
            break;
        }
        
        // 使用完整法线分量
        float3 normal = make_float3(nx, ny, nz);

        if (bounce == 0)
        {
            aovAlbedo = albedo;
            aovNormal = normal;
            aovValid = 1;
        }
        
        // 调试：可视化从 closesthit 返回的材质 baseColor
        if (params.debugMode == 3)
        {
            params.output[pixelID] = make_float4(albedo.x, albedo.y, albedo.z, 1.f);
            params.accum[pixelID] = make_float4(albedo.x, albedo.y, albedo.z, 1.f);
            break;
        }
        
        // 调试：可视化法线方向
        if (params.debugMode == 1)
        {
            float3 normalViz = (normal + make_float3(1.f, 1.f, 1.f)) * 0.5f;
            params.output[pixelID] = make_float4(normalViz.x, normalViz.y, normalViz.z, 1.f);
            params.accum[pixelID] = make_float4(normalViz.x, normalViz.y, normalViz.z, 1.f);
            break;  // 跳过到下一像素
        }
        
        //  命中位置
        float3 hitPos = rayOrg + rayDir * t;
        
        // 为直接光照计算准备（使用albedo的近似）
        float3 directAlbedo = albedo;
        if (matType == MaterialTypeData::BasicPBR) {
            // For legacy PBR, use the existing computation
            directAlbedo = albedo * (1.f - metallic);
            // Highly transmissive surfaces should not contribute diffuse direct light
            if (transmission > 0.2f) directAlbedo = make_float3(0.f, 0.f, 0.f);
        }
        // Remove energy that should go to transmission for direct lighting
        float transCut = clampf(transmission, 0.f, 1.f);
        directAlbedo = directAlbedo * (1.f - transCut);
        
        // Clamp roughness for stability
        float clampedRoughness = clampf(matType == MaterialTypeData::BasicPBR ? roughness : uroughness, 0.02f, 1.f);

        if (params.debugDirectional != 0 && bounce == 0)
        {
            if (params.numDirectionalLights == 0)
            {
                // 洋红色：没有方向光
                radiance = make_float3(1.f, 0.f, 1.f);
                break;
            }

            // direction 表示光源所在方向（指向光源），不是光线传播方向
            float3 L = normalize(params.directionalLights[0].direction);
            float nDotL = dot(normal, L);  // 法线·光源方向
            
            if (nDotL <= 0.f)
            {
                // 黄色：表面背对光源
                radiance = make_float3(0.9f, 0.9f, 0.1f);
            }
            else
            {
                // 阴影光线朝光源方向发射
                float3 shadowOrigin = hitPos + normal * 1e-3f;
                
                // 追踪阴影光线
                uint32_t u0 = __float_as_uint(-999.0f);
                uint32_t u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0, u6 = 0, u7 = 0, u8 = 0;
                
                optixTrace(params.traversable,
                           shadowOrigin,
                           L,  // 朝光源方向（不是 -L）
                           1e-4f, 1e20f, 0.0f,
                           OptixVisibilityMask(255),
                           OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                           0, 1, 0,
                           u0, u1, u2, u3, u4, u5, u6, u7, u8);
                
                float result = __uint_as_float(u0);
                
                if (result < -900.0f)
                {
                    // 青色：payload 没有被修改
                    radiance = make_float3(0.0f, 1.0f, 1.0f);
                }
                else if (result > 0.9f && result < 1.1f)
                {
                    // 绿色：miss（光源可见）
                    radiance = make_float3(0.1f, 0.9f, 0.1f);
                }
                else if (result > 0.2f && result < 0.3f)
                {
                    // 红色：closesthit（遮挡）
                    radiance = make_float3(0.9f, 0.1f, 0.1f);
                }
                else
                {
                    // 蓝色：其他未知值
                    float v = clampf(fabsf(result) * 0.001f, 0.0f, 1.0f);
                    radiance = make_float3(v * 0.2f, v * 0.2f, 0.9f);
                }
            }
            break;
        }
        
        // Next Event Estimation (NEE): 每次弹跳都计算直接光照
        float3 directLight = make_float3(0.f, 0.f, 0.f);
        
        // 环境光只在第一次弹跳添加
        if (bounce == 0) {
            directLight = directLight + directAlbedo * params.ambientIntensity;
        }
        
        // 点光源贡献（每次弹跳都计算）
        // 光照强度缩放 - 平衡PBRT场景和非PBRT场景
        const float lightIntensityScale = 0.1f;  // 中间值，可根据场景调整
        
        for (int i = 0; i < params.numLights; ++i) {
            if (params.lights[i].type == LightTypeData::Point) {
                float3 toLightVec = params.lights[i].position - hitPos;
                float distToLight = length(toLightVec);
                float3 toLightDir = toLightVec / distToLight;
                
                float cosTheta = dot(normal, toLightDir);
                if (cosTheta > 0.0f) {
                    float3 shadowOrigin = hitPos + normal * 1e-4f;
                    bool visible = traceShadowRay(shadowOrigin, toLightDir, 1e-4f, distToLight - 1e-3f);
                    
                    if (visible) {
                        float distSq = distToLight * distToLight;
                        float attenuation = 1.f / (1.f + distSq * 0.01f);
                        float3 scaledIntensity = params.lights[i].intensity * lightIntensityScale;
                        float3 pointContrib = albedo * scaledIntensity * cosTheta * attenuation * (1.f / 3.14159265f);
                        directLight = directLight + pointContrib;
                    }
                }
            } else if (params.lights[i].type == LightTypeData::Spot) {
                float3 toLightVec = params.lights[i].position - hitPos;
                float distToLight = length(toLightVec);
                float3 toLightDir = toLightVec / distToLight;

                float cosTheta = dot(normal, toLightDir);
                if (cosTheta > 0.0f) {
                    float3 shadowOrigin = hitPos + normal * 1e-4f;
                    bool visible = traceShadowRay(shadowOrigin, toLightDir, 1e-4f, distToLight - 1e-3f);

                    if (visible) {
                        float3 lightDir = normalize(params.lights[i].direction);
                        float3 fromLight = normalize(hitPos - params.lights[i].position);
                        float spotCos = dot(lightDir, fromLight);
                        float inner = params.lights[i].cosInner;
                        float outer = params.lights[i].cosOuter;
                        float spotWeight = 0.f;
                        if (spotCos >= inner) {
                            spotWeight = 1.f;
                        } else if (spotCos > outer) {
                            float t = (spotCos - outer) / fmaxf(inner - outer, 1e-4f);
                            spotWeight = t * t * (3.f - 2.f * t);
                        }

                        if (spotWeight > 0.f) {
                            float distSq = distToLight * distToLight;
                            float attenuation = 1.f / (1.f + distSq * 0.01f);
                            float3 scaledIntensity = params.lights[i].intensity * lightIntensityScale * spotWeight;
                            float3 spotContrib = albedo * scaledIntensity * cosTheta * attenuation * (1.f / 3.14159265f);
                            directLight = directLight + spotContrib;
                        }
                    }
                }
            } else if (params.lights[i].type == LightTypeData::Directional) {
                float3 L = normalize(params.lights[i].direction);
                float cosTheta = dot(normal, L);
                
                if (cosTheta > 0.0f) {
                    // 软阴影：给光线方向添加随机抖动
                    float3 jitteredL = L;
                    float lightRadius = 0.02f;  // 光源角度半径
                    float3 tangent, bitangent;
                    buildTangentBasis(L, tangent, bitangent);
                    float jx = (rng.nextFloat() - 0.5f) * lightRadius;
                    float jy = (rng.nextFloat() - 0.5f) * lightRadius;
                    jitteredL = normalize(L + tangent * jx + bitangent * jy);
                    
                    float3 shadowOrigin = hitPos + normal * 1e-4f;
                    bool visible = traceShadowRay(shadowOrigin, jitteredL, 1e-4f, 1e20f);
                    
                    if (visible) {
                        // 使用缩放后的强度，避免过曝
                        float3 scaledIntensity = params.lights[i].intensity * lightIntensityScale;
                        float3 dirContrib = albedo * scaledIntensity * cosTheta * (1.f / 3.14159265f);
                        directLight = directLight + dirContrib;
                    }
                }
            }
        }
        
        // 方向光贡献（每次弹跳都计算，带软阴影）
        for (int i = 0; i < params.numDirectionalLights; ++i) {
            float3 L = normalize(params.directionalLights[i].direction);
            float cosTheta = dot(normal, L);
            
            if (cosTheta > 0.0f) {
                // 软阴影：给光线方向添加随机抖动
                float3 jitteredL = L;
                float lightRadius = 0.02f;  // 光源角度半径，模拟太阳大小
                float3 tangent, bitangent;
                buildTangentBasis(L, tangent, bitangent);
                float jx = (rng.nextFloat() - 0.5f) * lightRadius;
                float jy = (rng.nextFloat() - 0.5f) * lightRadius;
                jitteredL = normalize(L + tangent * jx + bitangent * jy);
                
                float3 shadowOrigin = hitPos + normal * 1e-4f;
                bool visible = traceShadowRay(shadowOrigin, jitteredL, 1e-4f, 1e20f);
                
                if (visible) {
                    // 使用缩放后的强度，避免过曝
                    float3 scaledIntensity = params.directionalLights[i].intensity * lightIntensityScale;
                    float3 dirContrib = albedo * scaledIntensity * cosTheta * (1.f / 3.14159265f);
                    directLight = directLight + dirContrib;
                }
            }
        }
        
        // 调试：可视化直接光照贡献
        if (params.debugMode == 2)
        {
            float3 directLightViz = directLight * 0.5f;  // 缩放以便可见
            params.output[pixelID] = make_float4(directLightViz.x, directLightViz.y, directLightViz.z, 1.f);
            params.accum[pixelID] = make_float4(directLightViz.x, directLightViz.y, directLightViz.z, 1.f);
            break;  // 跳过到下一像素
        }
        
        // 添加表面自发光（area lights和emissive材质）
        // PBRT的emission可能很大，需要适当缩放
        float emissionScale = 1.0f;  // 可以调整此值来控制发光强度
        float3 scaledEmission = emission * emissionScale;
        // 限制最大emission防止过曝
        float emissionMax = 50.0f;
        scaledEmission.x = fminf(scaledEmission.x, emissionMax);
        scaledEmission.y = fminf(scaledEmission.y, emissionMax);
        scaledEmission.z = fminf(scaledEmission.z, emissionMax);
        radiance = radiance + throughput * scaledEmission;
        
        // 累积直接光照
        radiance = radiance + throughput * directLight;
        
        // 间接光照：使用统一的材质采样接口
        float3 V = normalize(-rayDir);
        MaterialSample ms;
        
        // Sample material BSDF based on material type
        switch (matType) {
            case MaterialTypeData::Diffuse:
                ms = sampleDiffuse(V, normal, albedo, sigma, rng.nextFloat(), rng.nextFloat());
                break;
            
            case MaterialTypeData::Conductor:
                ms = sampleConductor(V, normal, eta, k, uroughness, vroughness, remapRoughness,
                                      rng.nextFloat(), rng.nextFloat());
                break;
            
            case MaterialTypeData::Dielectric:
                ms = sampleDielectric(V, normal, ior, rng.nextFloat());
                break;
            
            case MaterialTypeData::RoughDielectric:
                ms = sampleRoughDielectric(V, normal, ior, uroughness, vroughness, remapRoughness,
                                            rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
                break;
            
            case MaterialTypeData::CoatedDiffuse:
                ms = sampleCoatedDiffuse(V, normal, albedo, ior, uroughness,
                                          rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
                break;
            
            case MaterialTypeData::CoatedConductor:
                // Note: using uroughness/vroughness for both interface and base due to payload constraints
                ms = sampleCoatedConductor(V, normal, eta, k, ior, uroughness, vroughness, remapRoughness,
                                            uroughness, vroughness,
                                            rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
                break;
            
            case MaterialTypeData::BasicPBR:
            default: {
                // Legacy PBR material sampling
                float NoV = fmaxf(dot(normal, V), 0.f);
                // Treat any non-zero transmission as refractive; weight by transmission probability
                float pTrans = clampf(transmission, 0.f, 0.95f);
                bool isTransmissive = pTrans > 0.001f;
                
                float f0d = (ior - 1.f) / (ior + 1.f);
                f0d = f0d * f0d;
                float3 f0 = lerp(make_float3(f0d, f0d, f0d), albedo, metallic);
                float pSpec = clampf((f0.x + f0.y + f0.z) * (1.f / 3.f), 0.05f, 0.95f);
                
                float xi = rng.nextFloat();

                if (isTransmissive && xi < pTrans) {
                    // Refraction
                    float cosThetaI = dot(normal, V);
                    float etaRatio = cosThetaI > 0.0f ? 1.0f / ior : ior;
                    float3 refractNormal = cosThetaI > 0.0f ? normal : normal * -1.0f;
                    
                    float cosThetaI2 = 1.0f - etaRatio * etaRatio * (1.0f - cosThetaI * cosThetaI);
                    
                    if (cosThetaI2 >= 0.0f) {
                        float cosThetaT = sqrtf(cosThetaI2);
                        ms.wi = normalize(V * etaRatio + refractNormal * (etaRatio * fabsf(cosThetaI) - cosThetaT));
                        
                        float3 f0_refr = make_float3(f0d, f0d, f0d);
                        float3 F = fresnelSchlick(fabsf(cosThetaI), f0_refr);
                        ms.f = (make_float3(1.0f, 1.0f, 1.0f) - F) * albedo;
                        ms.pdf = 1.0f;
                        ms.isDelta = true;
                    } else {
                        // Total internal reflection
                        ms.wi = reflectVec(-V, normal);
                        ms.f = make_float3(1.0f, 1.0f, 1.0f);
                        ms.pdf = 1.0f;
                        ms.isDelta = true;
                    }
                } else {
                    // Renormalize xi for the non-transmission branch
                    float xi2 = (isTransmissive ? (xi - pTrans) / fmaxf(1.f - pTrans, 1e-6f) : xi);

                    if (xi2 < pSpec) {
                    // GGX specular
                    float alpha = clampedRoughness * clampedRoughness;
                    float3 H = sampleGGX(normal, alpha, rng.nextFloat(), rng.nextFloat());
                    ms.wi = reflectVec(-V, H);
                    float NoL = fmaxf(dot(normal, ms.wi), 0.f);
                    float NoH = fmaxf(dot(normal, H), 0.f);
                    float VoH = fmaxf(dot(V, H), 0.f);
                    
                    if (NoL > 0.f && NoV > 0.f) {
                        float D = ggxD(NoH, alpha);
                        float G = ggxG(NoV, NoL, alpha);
                        float3 F = fresnelSchlick(VoH, f0);
                        ms.f = F * (D * G / fmaxf(4.f * NoV * NoL, 1e-6f));
                        ms.pdf = (D * NoH) / fmaxf(4.f * VoH, 1e-6f) * pSpec;
                    } else {
                        ms.f = make_float3(0.f, 0.f, 0.f);
                        ms.pdf = 0.f;
                    }
                    ms.isDelta = false;
                    } else {
                    // Cosine-weighted diffuse
                    ms.wi = hemisphereSample(normal, rng.nextFloat(), rng.nextFloat());
                    float NoL = fmaxf(dot(normal, ms.wi), 0.f);
                    
                    float3 Fd = fresnelSchlick(NoV, f0);
                    float3 kd = albedo * (1.f - metallic) * (make_float3(1.f, 1.f, 1.f) - Fd);
                    ms.f = kd * (1.f / 3.14159265f);
                    ms.pdf = NoL * (1.f / 3.14159265f) * (1.f - pSpec);
                    ms.isDelta = false;
                    }
                }
                break;
            }
        }
        
        // Check if sampling succeeded
        if (ms.pdf <= 0.f) break;
        
        float NoL = fmaxf(dot(normal, ms.wi), 0.f);
        if (NoL <= 0.f && !ms.isDelta) break;  // Allow negative for transmission
        
        // Update throughput
        if (ms.isDelta) {
            // Delta components still divide by branch probability for energy balance
            throughput = throughput * (ms.f / fmaxf(ms.pdf, 1e-6f));
        } else {
            // Regular materials: throughput *= f * cos / pdf
            throughput = throughput * ms.f * (fabsf(dot(normal, ms.wi)) / fmaxf(ms.pdf, 1e-6f));
        }
        
        // 检查 throughput 是否有效
        if (isnan(throughput.x) || isnan(throughput.y) || isnan(throughput.z) ||
            isinf(throughput.x) || isinf(throughput.y) || isinf(throughput.z))
            break;
        
        // 俄罗斯轮盘赌终止，最小概率限制为 0.05
        float maxComp = fmaxf(fmaxf(throughput.x, throughput.y), throughput.z);
        float survivalProb = clampf(maxComp, 0.05f, 1.0f);
        if (rng.nextFloat() > survivalProb)
            break;
        throughput = throughput * (1.f / survivalProb);
        
        // 设置下一次反弹
        // 根据光线方向相对法线是否保持同侧来偏移
        float offsetDir = dot(ms.wi, normal) > 0.f ? 1.f : -1.f;
        rayOrg = hitPos + normal * offsetDir * 1e-4f;
        rayDir = ms.wi;
    }
    
    // 渐进累积（调试模式 3, 1, 2 会在上面 break，所以这里只执行调试模式 0）
    if (params.debugMode == 0)
    {
        // 检查 radiance 是否有效，如果无效则丢弃这个样本
        bool validSample = !isnan(radiance.x) && !isnan(radiance.y) && !isnan(radiance.z) &&
                           !isinf(radiance.x) && !isinf(radiance.y) && !isinf(radiance.z) &&
                           radiance.x >= 0.f && radiance.y >= 0.f && radiance.z >= 0.f;
        
        // 限制 radiance 最大值以避免火萄虫（fireflies）
        const float maxRadiance = 100.0f;
        radiance.x = fminf(radiance.x, maxRadiance);
        radiance.y = fminf(radiance.y, maxRadiance);
        radiance.z = fminf(radiance.z, maxRadiance);
        
        float4 accumulated = params.accum[pixelID];
        float sampleCount = accumulated.w;
        float newCount = sampleCount;
        float3 avgRadiance = make_float3(accumulated.x, accumulated.y, accumulated.z);
        
        if (validSample) {
            newCount = sampleCount + 1.f;
            avgRadiance = (avgRadiance * sampleCount + radiance) / newCount;
        }
        
        params.accum[pixelID] = make_float4(avgRadiance.x, avgRadiance.y, avgRadiance.z, newCount);
        params.output[pixelID] = make_float4(avgRadiance.x, avgRadiance.y, avgRadiance.z, 1.f);

        if (params.albedo && params.normal && aovValid)
        {
            float4 albedoAccum = params.albedo[pixelID];
            float4 normalAccum = params.normal[pixelID];
            float aCount = albedoAccum.w;
            float3 avgAlbedo = make_float3(albedoAccum.x, albedoAccum.y, albedoAccum.z);
            float3 avgNormal = make_float3(normalAccum.x, normalAccum.y, normalAccum.z);
            float aNewCount = aCount + 1.f;
            avgAlbedo = (avgAlbedo * aCount + aovAlbedo) / aNewCount;
            avgNormal = (avgNormal * aCount + aovNormal) / aNewCount;
            params.albedo[pixelID] = make_float4(avgAlbedo.x, avgAlbedo.y, avgAlbedo.z, aNewCount);
            params.normal[pixelID] = make_float4(avgNormal.x, avgNormal.y, avgNormal.z, aNewCount);
        }
    }
}

// ---------------------------------------------------------------------------
// __miss__envmap: 天空/环境贴图
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__envmap()
{
    const float3 d = optixGetWorldRayDirection();
    
    // 检查光线标志以确定是否为阴影光线
    unsigned int rayFlags = optixGetRayFlags();
    if (rayFlags & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)
    {
        //  阴影光线未命中 - 光源可见
        payloadSetShadowBlocked(1.0f);
        return;
    }
    
    // 普通光线未命中 - 用很大的t值表示未命中
    optixSetPayload_0(__float_as_uint(1e10f));      // 很大的t值表示未命中
    optixSetPayload_1(__float_as_uint(0.f));        // 未使用
    optixSetPayload_2(__float_as_uint(0.f));        // 未使用
    // 清除材质载荷
    optixSetPayload_3(__float_as_uint(0.f));
    optixSetPayload_4(__float_as_uint(0.f));
    optixSetPayload_5(__float_as_uint(0.f));
    optixSetPayload_6(__float_as_uint(0.f));
    optixSetPayload_7(__float_as_uint(0.f));
    optixSetPayload_8(__float_as_uint(0.f));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 纹理采样辅助函数
// ---------------------------------------------------------------------------
static __device__ float4 sampleTextureRGBA(const uint8_t* pixels, int width, int height, float2 uv)
{
    if (!pixels || width <= 0 || height <= 0)
        return {1.f, 1.f, 1.f, 1.f};
    
    // 包裹坐标
    uv.x = uv.x - floorf(uv.x);
    uv.y = uv.y - floorf(uv.y);
    if (uv.x < 0.f) uv.x += 1.f;
    if (uv.y < 0.f) uv.y += 1.f;
    
    // 使用最近邻采样（如果需要可以是双线性）
    int x = static_cast<int>(uv.x * (width - 1));
    int y = static_cast<int>((1.f - uv.y) * (height - 1));  // 翻转Y以遵循OpenGL约定
    x = x < 0 ? 0 : (x >= width ? width - 1 : x);
    y = y < 0 ? 0 : (y >= height ? height - 1 : y);
    
    // RGBA格式：每像素读取4字节
    const int pixelIdx = (y * width + x) * 4;
    float r = pixels[pixelIdx + 0] * (1.f / 255.f);
    float g = pixels[pixelIdx + 1] * (1.f / 255.f);
    float b = pixels[pixelIdx + 2] * (1.f / 255.f);
    float a = pixels[pixelIdx + 3] * (1.f / 255.f);
    
    return {r, g, b, a};
}

// Backward-compatible helper returning RGB only
static __device__ float3 sampleTexture(const uint8_t* pixels, int width, int height, float2 uv)
{
    float4 v = sampleTextureRGBA(pixels, width, height, uv);
    return make_float3(v.x, v.y, v.z);
}

static __device__ float3 sampleTextureHDR(const float4* pixels, int width, int height, float2 uv)
{
    if (!pixels || width <= 0 || height <= 0)
        return {0.f, 0.f, 0.f};

    uv.x = uv.x - floorf(uv.x);
    uv.y = uv.y - floorf(uv.y);
    if (uv.x < 0.f) uv.x += 1.f;
    if (uv.y < 0.f) uv.y += 1.f;

    int x = static_cast<int>(uv.x * (width - 1));
    int y = static_cast<int>((1.f - uv.y) * (height - 1));
    x = x < 0 ? 0 : (x >= width ? width - 1 : x);
    y = y < 0 ? 0 : (y >= height ? height - 1 : y);

    const int pixelIdx = (y * width + x);
    float4 v = pixels[pixelIdx];
    return {v.x, v.y, v.z};
}

static __device__ float3 sampleTextureFloat(const float4* pixels, int width, int height, float2 uv)
{
    if (!pixels || width <= 0 || height <= 0)
        return {0.f, 0.f, 0.f};

    uv.x = uv.x - floorf(uv.x);
    uv.y = uv.y - floorf(uv.y);
    if (uv.x < 0.f) uv.x += 1.f;
    if (uv.y < 0.f) uv.y += 1.f;

    int x = static_cast<int>(uv.x * (width - 1));
    int y = static_cast<int>((1.f - uv.y) * (height - 1));
    x = x < 0 ? 0 : (x >= width ? width - 1 : x);
    y = y < 0 ? 0 : (y >= height ? height - 1 : y);

    const int pixelIdx = (y * width + x);
    float4 c = pixels[pixelIdx];
    return {c.x, c.y, c.z};
}

// ---------------------------------------------------------------------------
// Curve intersection helpers (custom primitives)
// ---------------------------------------------------------------------------
static __forceinline__ __device__ float3 bezierEvaluate(const float3 cp[4], float u, float3* deriv)
{
    float3 cp1[3] = { lerp(cp[0], cp[1], u), lerp(cp[1], cp[2], u), lerp(cp[2], cp[3], u) };
    float3 cp2[2] = { lerp(cp1[0], cp1[1], u), lerp(cp1[1], cp1[2], u) };
    if (deriv) {
        float3 d = cp2[1] - cp2[0];
        if (dot(d, d) > 1e-10f)
            *deriv = d * 3.f;
        else
            *deriv = cp[3] - cp[0];
    }
    return lerp(cp2[0], cp2[1], u);
}

static __forceinline__ __device__ void subdivideBezier(const float3 cp[4], float3 outCp[7])
{
    outCp[0] = cp[0];
    outCp[1] = (cp[0] + cp[1]) * 0.5f;
    outCp[2] = (cp[0] + cp[1] * 2.f + cp[2]) * 0.25f;
    outCp[3] = (cp[0] + cp[1] * 3.f + cp[2] * 3.f + cp[3]) * 0.125f;
    outCp[4] = (cp[1] + cp[2] * 2.f + cp[3]) * 0.25f;
    outCp[5] = (cp[2] + cp[3]) * 0.5f;
    outCp[6] = cp[3];
}

static __device__ bool curveLeafIntersect(
    const CurveData& c,
    const float3 cp[4],
    float rayLength,
    float tMax,
    float u0,
    float u1,
    float* tHit,
    float* outU,
    float* outV)
{
    float edge = (cp[1].y - cp[0].y) * -cp[0].y + cp[0].x * (cp[0].x - cp[1].x);
    if (edge < 0) return false;
    edge = (cp[2].y - cp[3].y) * -cp[3].y + cp[3].x * (cp[3].x - cp[2].x);
    if (edge < 0) return false;

    float2 segDir = make_float2(cp[3].x - cp[0].x, cp[3].y - cp[0].y);
    float denom = segDir.x * segDir.x + segDir.y * segDir.y;
    if (denom == 0) return false;
    float w = (segDir.x * -cp[0].x + segDir.y * -cp[0].y) / denom;

    float u = clampf(lerpf(u0, u1, w), u0, u1);
    float hitWidth = lerpf(c.width0, c.width1, u);

    float3 dpcdw;
    float3 pc = bezierEvaluate(cp, clampf(w, 0.f, 1.f), &dpcdw);
    float ptCurveDist2 = pc.x * pc.x + pc.y * pc.y;

    if (ptCurveDist2 > (hitWidth * hitWidth * 0.25f)) return false;
    if (pc.z < 0.f || pc.z > rayLength * tMax) return false;

    float t = pc.z / rayLength;
    if (tHit && t > *tHit) return false;

    float ptCurveDist = sqrtf(fmaxf(ptCurveDist2, 0.f));
    float edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y;
    float v = (edgeFunc > 0) ? 0.5f + ptCurveDist / hitWidth : 0.5f - ptCurveDist / hitWidth;

    if (tHit) *tHit = t;
    if (outU) *outU = u;
    if (outV) *outV = v;
    return true;
}

static __device__ bool curveRecursiveIntersect(
    const CurveData& c,
    const float3 cp[4],
    float3 rayDir,
    float tMax,
    const float3 wx, const float3 wy, const float3 wz,
    float u0, float u1,
    int depth,
    float* tHit,
    float* outU,
    float* outV)
{
    (void)wx;
    (void)wy;
    (void)wz;

    float rayLength = length(rayDir);
    bool hit = false;

    struct StackNode
    {
        float3 cp[4];
        float u0;
        float u1;
        int depth;
    };

    const int kMaxStack = 64;
    StackNode stack[kMaxStack];
    int stackSize = 0;

    StackNode root;
    for (int i = 0; i < 4; ++i)
        root.cp[i] = cp[i];
    root.u0 = u0;
    root.u1 = u1;
    root.depth = depth;
    stack[stackSize++] = root;

    while (stackSize > 0)
    {
        StackNode node = stack[--stackSize];

        if (node.depth <= 0)
        {
            if (curveLeafIntersect(c, node.cp, rayLength, tMax, node.u0, node.u1, tHit, outU, outV))
                hit = true;
            continue;
        }

        float3 cpSplit[7];
        subdivideBezier(node.cp, cpSplit);
        float uSeg[3] = {node.u0, 0.5f * (node.u0 + node.u1), node.u1};

        for (int seg = 0; seg < 2; ++seg)
        {
            float maxWidth = fmaxf(lerpf(c.width0, c.width1, uSeg[seg]),
                                   lerpf(c.width0, c.width1, uSeg[seg + 1]));
            float3 cps[4] = {cpSplit[seg * 3 + 0], cpSplit[seg * 3 + 1], cpSplit[seg * 3 + 2], cpSplit[seg * 3 + 3]};
            float minX = fminf(fminf(cps[0].x, cps[1].x), fminf(cps[2].x, cps[3].x));
            float minY = fminf(fminf(cps[0].y, cps[1].y), fminf(cps[2].y, cps[3].y));
            float minZ = fminf(fminf(cps[0].z, cps[1].z), fminf(cps[2].z, cps[3].z));
            float maxX = fmaxf(fmaxf(cps[0].x, cps[1].x), fmaxf(cps[2].x, cps[3].x));
            float maxY = fmaxf(fmaxf(cps[0].y, cps[1].y), fmaxf(cps[2].y, cps[3].y));
            float maxZ = fmaxf(fmaxf(cps[0].z, cps[1].z), fmaxf(cps[2].z, cps[3].z));
            float w = 0.5f * maxWidth;

            if (minX > w || maxX < -w || minY > w || maxY < -w || maxZ < 0.f || minZ > rayLength * tMax)
                continue;

            if (stackSize >= kMaxStack - 1)
            {
                if (curveLeafIntersect(c, cps, rayLength, tMax, uSeg[seg], uSeg[seg + 1], tHit, outU, outV))
                    hit = true;
                continue;
            }

            StackNode child;
            for (int i = 0; i < 4; ++i)
                child.cp[i] = cps[i];
            child.u0 = uSeg[seg];
            child.u1 = uSeg[seg + 1];
            child.depth = node.depth - 1;
            stack[stackSize++] = child;
        }
    }

    return hit;
}

extern "C" __global__ void __intersection__curve()
{
    const unsigned int primIdx = optixGetPrimitiveIndex();
    if (!params.curves || primIdx >= static_cast<unsigned int>(params.numCurves))
        return;

    const CurveData& c = params.curves[primIdx];
    float3 rayOrig = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float tMax = optixGetRayTmax();

    float3 dx = cross(rayDir, c.cp[3] - c.cp[0]);
    if (length(dx) < 1e-6f)
    {
        float3 tmp = fabsf(rayDir.x) > 0.9f ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f);
        dx = cross(rayDir, tmp);
    }
    float3 wz = normalize(rayDir);
    float3 wx = normalize(dx);
    float3 wy = normalize(cross(wz, wx));

    float3 cpRay[4];
    for (int i = 0; i < 4; ++i) {
        float3 op = c.cp[i] - rayOrig;
        cpRay[i] = make_float3(dot(op, wx), dot(op, wy), dot(op, wz));
    }

    float maxWidth = fmaxf(c.width0, c.width1);
    float minX = fminf(fminf(cpRay[0].x, cpRay[1].x), fminf(cpRay[2].x, cpRay[3].x));
    float minY = fminf(fminf(cpRay[0].y, cpRay[1].y), fminf(cpRay[2].y, cpRay[3].y));
    float minZ = fminf(fminf(cpRay[0].z, cpRay[1].z), fminf(cpRay[2].z, cpRay[3].z));
    float maxX = fmaxf(fmaxf(cpRay[0].x, cpRay[1].x), fmaxf(cpRay[2].x, cpRay[3].x));
    float maxY = fmaxf(fmaxf(cpRay[0].y, cpRay[1].y), fmaxf(cpRay[2].y, cpRay[3].y));
    float maxZ = fmaxf(fmaxf(cpRay[0].z, cpRay[1].z), fmaxf(cpRay[2].z, cpRay[3].z));
    float w = 0.5f * maxWidth;

    if (minX > w || maxX < -w || minY > w || maxY < -w || maxZ < 0.f)
        return;

    float L0 = 0.f;
    for (int i = 0; i < 2; ++i) {
        float3 a = cpRay[i] - cpRay[i + 1] * 2.f + cpRay[i + 2];
        L0 = fmaxf(L0, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fabsf(a.z)));
    }

    int maxDepth = 0;
    if (L0 > 0.f) {
        float eps = maxWidth * 0.05f;
        float r0 = log2f(1.41421356237f * 6.f * L0 / fmaxf(8.f * eps, 1e-6f)) * 0.5f;
        maxDepth = (int)clampf(r0, 0.f, 10.f);
    }

    float tHit = 1e30f;
    float uHit = 0.f;
    float vHit = 0.f;
    bool hit = curveRecursiveIntersect(c, cpRay, rayDir, tMax, wx, wy, wz, 0.f, 1.f, maxDepth, &tHit, &uHit, &vHit);
    if (!hit) return;

    optixReportIntersection(tHit, 0, __float_as_uint(uHit), __float_as_uint(vHit));
}

// __closesthit__pbr: 命中表面着色器 - 计算真实三角形法线
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__pbr()
{
    // 检查是否为阴影光线：使用光线标志而不是 payload
    unsigned int rayFlags = optixGetRayFlags();
    if (rayFlags & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)
    {
        // 阴影光线命中物体 - 设置为遮挡（使用特殊值 0.25f 以便调试）
        payloadSetShadowBlocked(0.25f);
        return;
    }

    const HitGroupData* data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    
    // 获取光线信息
    float t = optixGetRayTmax();  // 命中距离
    
    float3 normal;
    float2 uv = make_float2(0.5f, 0.5f);

    if (data->isCurve)
    {
        const unsigned int primIdx = optixGetPrimitiveIndex();
        if (params.curves && primIdx < static_cast<unsigned int>(params.numCurves))
        {
            const CurveData& c = params.curves[primIdx];
            float u = __uint_as_float(optixGetAttribute_0());
            float v = __uint_as_float(optixGetAttribute_1());
            float3 dpdu;
            bezierEvaluate(c.cp, u, &dpdu);
            float width = lerpf(c.width0, c.width1, u);

            if (c.type == CurveTypeData::Ribbon && c.hasNormals)
            {
                float3 nHit = normalize(lerp(c.n0, c.n1, u));
                float3 dpdv = normalize(cross(nHit, dpdu)) * width;
                normal = normalize(cross(dpdu, dpdv));
            }
            else
            {
                float3 t = normalize(dpdu);
                float3 b = normalize(cross(optixGetWorldRayDirection(), t));
                if (length(b) < 1e-6f)
                    b = normalize(cross(make_float3(0.f, 1.f, 0.f), t));
                float3 n = normalize(cross(t, b));

                if (c.type == CurveTypeData::Cylinder)
                {
                    float theta = (v - 0.5f) * 3.14159265f;
                    float3 dpdv = normalize(b * cosf(theta) + n * sinf(theta)) * width;
                    normal = normalize(cross(t, dpdv));
                }
                else
                {
                    normal = n;
                }
            }

            const float3 rayDir = optixGetWorldRayDirection();
            if (dot(normal, rayDir) > 0.f)
                normal = normal * -1.f;

            uv = make_float2(u, v);
        }
        else
        {
            float3 rayDir = optixGetWorldRayDirection();
            normal = normalize(-rayDir);
        }
    }
    else if (data->vertices && data->indices)
    {
        // 获取三角形索引和重心坐标
        const unsigned int primIdx = optixGetPrimitiveIndex();
        const float2 bary = optixGetTriangleBarycentrics();
        const float b0 = 1.f - bary.x - bary.y;
        const float b1 = bary.x;
        const float b2 = bary.y;
        
        if (data->hasNormals && data->normals)
        {
            // 使用重心坐标插值顶点法线
            const unsigned int i0 = data->indices[primIdx * 3 + 0];
            const unsigned int i1 = data->indices[primIdx * 3 + 1];
            const unsigned int i2 = data->indices[primIdx * 3 + 2];
            
            const float3 n0 = data->normals[i0];
            const float3 n1 = data->normals[i1];
            const float3 n2 = data->normals[i2];
            
            normal = normalize(n0 * b0 + n1 * b1 + n2 * b2);
        }
        else
        {
            // 从三角形边计算平坦几何法线
            const unsigned int i0 = data->indices[primIdx * 3 + 0];
            const unsigned int i1 = data->indices[primIdx * 3 + 1];
            const unsigned int i2 = data->indices[primIdx * 3 + 2];
            
            const float3 v0 = data->vertices[i0];
            const float3 v1 = data->vertices[i1];
            const float3 v2 = data->vertices[i2];
            
            const float3 e1 = v1 - v0;
            const float3 e2 = v2 - v0;
            normal = normalize(cross(e1, e2));
        }
        
        // 翻转法线朝向相机
        const float3 rayDir = optixGetWorldRayDirection();
        if (dot(normal, rayDir) > 0.f)
            normal = normal * -1.f;
    }
    else
    {
        // 后备方案：使用取反的光线方向
        float3 rayDir = optixGetWorldRayDirection();
        normal.x = -rayDir.x;
        normal.y = -rayDir.y;
        normal.z = -rayDir.z;
        normal = normalize(normal);
    }
    
    // 计算材质baseColor：如果可用则采样纹理，否则使用baseColor
    float3 matBase = data->baseColor;
    float matMetallic = data->metallic;
    float matRoughness = data->roughness;
    float3 matEmission = data->emission;
    float matTransmission = data->transmission;
    float matIor = data->ior;
    
    // 获取纹理坐标
    if (!data->isCurve && data->hasTexCoords && data->texCoords)
    {
        // 插值纹理坐标
        const unsigned int primIdx = optixGetPrimitiveIndex();
        const float2 bary = optixGetTriangleBarycentrics();
        const float b0 = 1.f - bary.x - bary.y;
        const float b1 = bary.x;
        const float b2 = bary.y;
        
        const unsigned int i0 = data->indices[primIdx * 3 + 0];
        const unsigned int i1 = data->indices[primIdx * 3 + 1];
        const unsigned int i2 = data->indices[primIdx * 3 + 2];
        
        const float2 uv0 = data->texCoords[i0];
        const float2 uv1 = data->texCoords[i1];
        const float2 uv2 = data->texCoords[i2];
        
        uv = make_float2(uv0.x * b0 + uv1.x * b1 + uv2.x * b2,
                         uv0.y * b0 + uv1.y * b1 + uv2.y * b2);
    }
    
    float alpha = 1.f;

    // 采样Base Color纹理并读取alpha（用于cutout/半透明）
    if (data->hasBaseColorTex && data->baseColorTexWidth > 0 && data->baseColorTexHeight > 0)
    {
        float4 baseSample = sampleTextureRGBA(data->baseColorTexPixels, data->baseColorTexWidth, data->baseColorTexHeight, uv);
        matBase = make_float3(baseSample.x, baseSample.y, baseSample.z);
        alpha = baseSample.w;
    }
    
    // 采样Metallic + Roughness纹理（通常R通道=金属度，G通道=粗糙度）
    if (data->hasMetallicRoughnessTex && data->metallicRoughnessTexWidth > 0 && data->metallicRoughnessTexHeight > 0)
    {
        float3 mrSample = sampleTexture(data->metallicRoughnessTexPixels, data->metallicRoughnessTexWidth, data->metallicRoughnessTexHeight, uv);
        // glTF标准：R通道=metallic, G通道=roughness
        matMetallic = mrSample.x;
        matRoughness = mrSample.y;
    }
    
    // 采样法线贴图
    if (data->hasNormalTex && data->normalTexWidth > 0 && data->normalTexHeight > 0)
    {
        float3 normalMapSample = sampleTexture(data->normalTexPixels, data->normalTexWidth, data->normalTexHeight, uv);
        // 法线贴图通常存储在[0,1]范围，需要转换到[-1,1]
        float3 tangentNormal = normalMapSample * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
        
        // 构建TBN矩阵（Tangent-Bitangent-Normal）
        float3 tangent, bitangent;
        buildTangentBasis(normal, tangent, bitangent);
        
        // 将切线空间法线转换到世界空间
        normal = normalize(tangent * tangentNormal.x + bitangent * tangentNormal.y + normal * tangentNormal.z);
        
        // 确保法线朝向相机
        const float3 rayDir = optixGetWorldRayDirection();
        if (dot(normal, rayDir) > 0.f)
            normal = normal * -1.f;
    }
    
    // 采样发光纹理
    if (data->hasEmissionTex && data->emissionTexWidth > 0 && data->emissionTexHeight > 0)
    {
        float3 emissionSample = sampleTexture(data->emissionTexPixels, data->emissionTexWidth, data->emissionTexHeight, uv);
        matEmission = matEmission * emissionSample;  // 与材质emission相乘
    }
    
    // 采样透射率纹理（用于玻璃、透明材质）
    if (data->hasTransmissionTex && data->transmissionTexWidth > 0 && data->transmissionTexHeight > 0)
    {
        float3 transmissionSample = sampleTexture(data->transmissionTexPixels, data->transmissionTexWidth, data->transmissionTexHeight, uv);
        matTransmission = matTransmission * transmissionSample.x;  // 使用R通道
    }

    // Alpha cutout：如果alpha很低则视为未命中；否则将alpha转换为透射度补充
    if (alpha < 0.05f)
    {
        // 标记为未命中，回退到背景/下一次追踪
        optixSetPayload_0(__float_as_uint(1e10f));
        optixSetPayload_1(__float_as_uint(0.f));
        optixSetPayload_2(__float_as_uint(0.f));
        optixSetPayload_3(__float_as_uint(0.f));
        optixSetPayload_4(__float_as_uint(0.f));
        optixSetPayload_5(__float_as_uint(0.f));
        optixSetPayload_6(__float_as_uint(0.f));
        optixSetPayload_7(__float_as_uint(0.f));
        optixSetPayload_8(__float_as_uint(0.f));
        return;
    }

    // 将不透明度映射到透射度（适用于半透明贴图，和 PBR 的 transmission 叠加）
    matTransmission = fminf(1.f, matTransmission + (1.f - alpha));
    
    // Encode material type and parameters to payload
    // p0=t, p1-3=normal, p4=materialType (as uint), p5-7=baseColor/albedo
    // p8-10=additional params (depends on material type)
    // p11-13=emission, transmission, ior OR eta/k/roughness
    
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(normal.x));
    optixSetPayload_2(__float_as_uint(normal.y));
    optixSetPayload_3(__float_as_uint(normal.z));
    
    // Material type as uint
    optixSetPayload_4(static_cast<uint32_t>(data->materialType));
    
    // Based on material type, encode different parameters
    switch (data->materialType) {
        case MaterialTypeData::Diffuse:
            // p5-7: reflectance (albedo), p8: sigma
            optixSetPayload_5(__float_as_uint(matBase.x));
            optixSetPayload_6(__float_as_uint(matBase.y));
            optixSetPayload_7(__float_as_uint(matBase.z));
            optixSetPayload_8(__float_as_uint(data->sigma));
            optixSetPayload_9(__float_as_uint(matEmission.x));
            optixSetPayload_10(__float_as_uint(matEmission.y));
            optixSetPayload_11(__float_as_uint(matEmission.z));
            break;
            
        case MaterialTypeData::Conductor:
            // p5-7: eta (complex IOR real), p8-10: k (extinction), p11-12: roughness
            optixSetPayload_5(__float_as_uint(data->eta.x));
            optixSetPayload_6(__float_as_uint(data->eta.y));
            optixSetPayload_7(__float_as_uint(data->eta.z));
            optixSetPayload_8(__float_as_uint(data->k.x));
            optixSetPayload_9(__float_as_uint(data->k.y));
            optixSetPayload_10(__float_as_uint(data->k.z));
            optixSetPayload_11(__float_as_uint(data->uroughness));
            optixSetPayload_12(__float_as_uint(data->vroughness));
            optixSetPayload_13(__float_as_uint(data->remapRoughness ? 1.f : 0.f));
            break;
            
        case MaterialTypeData::Dielectric:
        case MaterialTypeData::RoughDielectric:
            // p5: eta, p6-7: roughness (for RoughDielectric), p8: remapRoughness
            optixSetPayload_5(__float_as_uint(matIor));
            optixSetPayload_6(__float_as_uint(data->uroughness));
            optixSetPayload_7(__float_as_uint(data->vroughness));
            optixSetPayload_8(__float_as_uint(data->remapRoughness ? 1.f : 0.f));
            optixSetPayload_9(__float_as_uint(matEmission.x));
            optixSetPayload_10(__float_as_uint(matEmission.y));
            optixSetPayload_11(__float_as_uint(matEmission.z));
            break;
            
        case MaterialTypeData::CoatedDiffuse:
            // p5-7: base reflectance, p8: interface eta, p9-10: interface roughness
            optixSetPayload_5(__float_as_uint(matBase.x));
            optixSetPayload_6(__float_as_uint(matBase.y));
            optixSetPayload_7(__float_as_uint(matBase.z));
            optixSetPayload_8(__float_as_uint(matIor));  // interface eta
            optixSetPayload_9(__float_as_uint(data->uroughness));  // interface uroughness
            optixSetPayload_10(__float_as_uint(data->vroughness));
            optixSetPayload_11(__float_as_uint(matEmission.x));
            optixSetPayload_12(__float_as_uint(matEmission.y));
            optixSetPayload_13(__float_as_uint(matEmission.z));
            break;
            
        case MaterialTypeData::CoatedConductor:
            // p5-7: conductor eta, p8-10: conductor k, p11: interface IOR, p12-13: interface roughness
            // Note: Can't fit all parameters in 14 payload slots, using simplified encoding
            // Store: eta (3), k (3), interface properties (2), base roughness (2)
            optixSetPayload_5(__float_as_uint(data->eta.x));
            optixSetPayload_6(__float_as_uint(data->eta.y));
            optixSetPayload_7(__float_as_uint(data->eta.z));
            optixSetPayload_8(__float_as_uint(data->k.x));
            optixSetPayload_9(__float_as_uint(data->k.y));
            optixSetPayload_10(__float_as_uint(data->k.z));
            optixSetPayload_11(__float_as_uint(matIor));  // interface IOR
            optixSetPayload_12(__float_as_uint(data->uroughness));  // interface uroughness  
            optixSetPayload_13(__float_as_uint(data->vroughness));  // interface vroughness
            // Note: base conductor roughness not transmitted, would need more payload slots
            // Using uroughness/vroughness for both interface and base as approximation
            break;
            
        case MaterialTypeData::BasicPBR:
        default:
            // Legacy: baseColor, metallic, roughness, emission, transmission, ior
            optixSetPayload_5(__float_as_uint(matBase.x));
            optixSetPayload_6(__float_as_uint(matBase.y));
            optixSetPayload_7(__float_as_uint(matBase.z));
            optixSetPayload_8(__float_as_uint(matMetallic));
            optixSetPayload_9(__float_as_uint(matRoughness));
            optixSetPayload_10(__float_as_uint(matEmission.x));
            optixSetPayload_11(__float_as_uint(matEmission.y));
            optixSetPayload_12(__float_as_uint(matEmission.z));
            optixSetPayload_13(__float_as_uint(matTransmission));
            break;
    }
}

