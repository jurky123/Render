#pragma once

#include "Spectrum.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <memory>
#include <string>
#include <vector>

// Complete pbrt-v4 light system
// Supports: Point, Distant, Spot, Goniometric, Projection, Infinite (with importance sampling)

namespace pbrt {

// Forward declarations
struct LightSampleContext;
struct LightSample;
class Distribution2D;

// Light types
enum class LightType {
    Point,
    Distant,
    Spot,
    Infinite,
    Goniometric,
    Projection,
    DiffuseArea
};

// Light interface
class Light {
public:
    virtual ~Light() = default;
    virtual LightType GetType() const = 0;
    virtual Spectrum Le(const glm::vec3& dir) const = 0;
    virtual std::string ToString() const = 0;
};

// Distribution2D for importance sampling (used by infinite lights)
class Distribution2D {
public:
    Distribution2D(const std::vector<float>& data, int width, int height);
    
    // Sample a 2D position based on distribution
    glm::vec2 Sample(const glm::vec2& u, float* pdf = nullptr) const;
    float Pdf(const glm::vec2& uv) const;
    
private:
    struct Distribution1D {
        std::vector<float> func;
        std::vector<float> cdf;
        float funcInt;
        
        Distribution1D(const std::vector<float>& f);
        int Sample(float u, float* pdf = nullptr) const;
        float Pdf(int index) const;
    };
    
    std::vector<Distribution1D> conditionalV;
    Distribution1D marginalU;
    int width, height;
};

// PointLight
class PointLight : public Light {
public:
    glm::vec3 position;
    Spectrum I;  // Intensity
    float scale;
    
    PointLight(const glm::vec3& pos, const Spectrum& I, float scale = 1.0f)
        : position(pos), I(I), scale(scale) {}
    
    LightType GetType() const override { return LightType::Point; }
    Spectrum Le(const glm::vec3& dir) const override { return Spectrum::CreateConstant(0); }
    std::string ToString() const override { return "PointLight"; }
};

// DistantLight (directional)
class DistantLight : public Light {
public:
    glm::vec3 direction;
    Spectrum L;  // Radiance
    float scale;
    
    DistantLight(const glm::vec3& dir, const Spectrum& L, float scale = 1.0f)
        : direction(glm::normalize(dir)), L(L), scale(scale) {}
    
    LightType GetType() const override { return LightType::Distant; }
    Spectrum Le(const glm::vec3& dir) const override { return L; }
    std::string ToString() const override { return "DistantLight"; }
};

// SpotLight
class SpotLight : public Light {
public:
    glm::vec3 position;
    glm::vec3 direction;
    Spectrum I;
    float cosTotalWidth;
    float cosFalloffStart;
    float scale;
    
    SpotLight(const glm::vec3& pos, const glm::vec3& dir, const Spectrum& I,
              float totalWidth, float falloffStart, float scale = 1.0f)
        : position(pos), direction(glm::normalize(dir)), I(I), scale(scale) {
        cosTotalWidth = std::cos(glm::radians(totalWidth));
        cosFalloffStart = std::cos(glm::radians(falloffStart));
    }
    
    LightType GetType() const override { return LightType::Spot; }
    Spectrum Le(const glm::vec3& dir) const override { return Spectrum::CreateConstant(0); }
    std::string ToString() const override { return "SpotLight"; }
};

// InfiniteLight (environment map with importance sampling)
class InfiniteLight : public Light {
public:
    Spectrum Lemit;  // Constant emission (if no map)
    std::shared_ptr<Distribution2D> distribution;
    std::vector<RGB> envMap;
    int width, height;
    glm::mat4 worldToLight;
    float scale;
    const RGBColorSpace* colorSpace;
    
    InfiniteLight(const Spectrum& L, float scale = 1.0f)
        : Lemit(L), width(0), height(0), scale(scale), colorSpace(RGBColorSpace::sRGB) {
        worldToLight = glm::mat4(1.0f);
    }
    
    InfiniteLight(const std::string& texturePath, float scale = 1.0f,
                  const RGBColorSpace* cs = RGBColorSpace::sRGB);
    
    LightType GetType() const override { return LightType::Infinite; }
    
    Spectrum Le(const glm::vec3& dir) const override {
        if (envMap.empty()) return Lemit;
        
        // Direction to spherical coordinates
        glm::vec3 d = glm::normalize(glm::vec3(worldToLight * glm::vec4(dir, 0.0f)));
        float theta = std::acos(std::clamp(d.z, -1.0f, 1.0f));
        float phi = std::atan2(d.y, d.x);
        if (phi < 0) phi += 2 * glm::pi<float>();
        
        float u = phi / (2 * glm::pi<float>());
        float v = theta / glm::pi<float>();
        
        int x = std::clamp(int(u * width), 0, width - 1);
        int y = std::clamp(int(v * height), 0, height - 1);
        
        RGB rgb = envMap[y * width + x];
        return Spectrum::CreateRGBUnbounded(colorSpace, rgb * scale);
    }
    
    // Importance sample direction based on environment map
    glm::vec3 SampleLi(const glm::vec2& u, float* pdf) const;
    float Pdf(const glm::vec3& dir) const;
    
    std::string ToString() const override { return "InfiniteLight"; }
};

// GoniometricLight (measured light distribution)
class GoniometricLight : public Light {
public:
    glm::vec3 position;
    Spectrum I;
    float scale;
    // Simplified: no actual goniometric data yet
    
    GoniometricLight(const glm::vec3& pos, const Spectrum& I, float scale = 1.0f)
        : position(pos), I(I), scale(scale) {}
    
    LightType GetType() const override { return LightType::Goniometric; }
    Spectrum Le(const glm::vec3& dir) const override { return Spectrum::CreateConstant(0); }
    std::string ToString() const override { return "GoniometricLight"; }
};

// ProjectionLight (projector)
class ProjectionLight : public Light {
public:
    glm::vec3 position;
    glm::vec3 direction;
    Spectrum I;
    float fov;
    float scale;
    // Simplified: no projection texture yet
    
    ProjectionLight(const glm::vec3& pos, const glm::vec3& dir, const Spectrum& I,
                    float fov = 45.0f, float scale = 1.0f)
        : position(pos), direction(glm::normalize(dir)), I(I), fov(fov), scale(scale) {}
    
    LightType GetType() const override { return LightType::Projection; }
    Spectrum Le(const glm::vec3& dir) const override { return Spectrum::CreateConstant(0); }
    std::string ToString() const override { return "ProjectionLight"; }
};

// DiffuseAreaLight (area light on geometry)
class DiffuseAreaLight : public Light {
public:
    Spectrum Lemit;
    float scale;
    bool twoSided;
    
    // Associated with geometry (shape index)
    int shapeIndex;
    
    DiffuseAreaLight(const Spectrum& L, float scale = 1.0f, bool twoSided = false)
        : Lemit(L), scale(scale), twoSided(twoSided), shapeIndex(-1) {}
    
    LightType GetType() const override { return LightType::DiffuseArea; }
    Spectrum Le(const glm::vec3& dir) const override { return Lemit; }
    std::string ToString() const override { return "DiffuseAreaLight"; }
};

// Light factories
std::shared_ptr<Light> CreatePointLight(const glm::vec3& position, const Spectrum& I, float scale = 1.0f);
std::shared_ptr<Light> CreateDistantLight(const glm::vec3& direction, const Spectrum& L, float scale = 1.0f);
std::shared_ptr<Light> CreateSpotLight(const glm::vec3& position, const glm::vec3& direction,
                                        const Spectrum& I, float coneAngle, float coneDeltaAngle, float scale = 1.0f);
std::shared_ptr<Light> CreateInfiniteLight(const Spectrum& L, float scale = 1.0f);
std::shared_ptr<Light> CreateInfiniteLightFromTexture(const std::string& texturePath, float scale = 1.0f);
std::shared_ptr<Light> CreateGoniometricLight(const glm::vec3& position, const Spectrum& I, float scale = 1.0f);
std::shared_ptr<Light> CreateProjectionLight(const glm::vec3& position, const glm::vec3& direction,
                                              const Spectrum& I, float fov = 45.0f, float scale = 1.0f);
std::shared_ptr<Light> CreateDiffuseAreaLight(const Spectrum& L, float scale = 1.0f, bool twoSided = false);

} // namespace pbrt
