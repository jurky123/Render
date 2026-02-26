#include "PBRTv4Lights.h"
#include <stb_image.h>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace pbrt {

// Distribution1D implementation
Distribution2D::Distribution1D::Distribution1D(const std::vector<float>& f) : func(f) {
    int n = f.size();
    cdf.resize(n + 1);
    cdf[0] = 0;
    
    for (int i = 1; i < n + 1; ++i) {
        cdf[i] = cdf[i - 1] + func[i - 1] / n;
    }
    
    funcInt = cdf[n];
    if (funcInt == 0) {
        for (int i = 1; i < n + 1; ++i)
            cdf[i] = float(i) / float(n);
    } else {
        for (int i = 1; i < n + 1; ++i)
            cdf[i] /= funcInt;
    }
}

int Distribution2D::Distribution1D::Sample(float u, float* pdf) const {
    int offset = std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin() - 1;
    offset = std::clamp(offset, 0, int(func.size()) - 1);
    if (pdf) *pdf = Pdf(offset);
    return offset;
}

float Distribution2D::Distribution1D::Pdf(int index) const {
    return (funcInt > 0) ? func[index] / (funcInt * func.size()) : 0;
}

// Distribution2D implementation
Distribution2D::Distribution2D(const std::vector<float>& data, int w, int h)
    : width(w), height(h), marginalU({}) {
    
    // Build conditional distributions for each row
    conditionalV.reserve(height);
    for (int v = 0; v < height; ++v) {
        std::vector<float> row(width);
        for (int u = 0; u < width; ++u) {
            row[u] = data[v * width + u];
        }
        conditionalV.emplace_back(row);
    }
    
    // Build marginal distribution
    std::vector<float> marginalFunc(height);
    for (int v = 0; v < height; ++v) {
        marginalFunc[v] = conditionalV[v].funcInt;
    }
    marginalU = Distribution1D(marginalFunc);
}

glm::vec2 Distribution2D::Sample(const glm::vec2& u, float* pdf) const {
    float pdfs[2];
    int v = marginalU.Sample(u.y, &pdfs[1]);
    int uu = conditionalV[v].Sample(u.x, &pdfs[0]);
    
    if (pdf) *pdf = pdfs[0] * pdfs[1];
    
    return glm::vec2((uu + u.x) / width, (v + u.y) / height);
}

float Distribution2D::Pdf(const glm::vec2& uv) const {
    int iu = std::clamp(int(uv.x * width), 0, width - 1);
    int iv = std::clamp(int(uv.y * height), 0, height - 1);
    return conditionalV[iv].func[iu] / marginalU.funcInt;
}

// InfiniteLight with texture
InfiniteLight::InfiniteLight(const std::string& texturePath, float scale, const RGBColorSpace* cs)
    : Lemit(Spectrum::CreateConstant(0)), scale(scale), colorSpace(cs) {
    
    worldToLight = glm::mat4(1.0f);
    
    // Load HDR/EXR image
    int channels;
    float* data = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "Failed to load infinite light texture: " << texturePath << std::endl;
        width = height = 1;
        envMap.resize(1);
        envMap[0] = RGB(1, 1, 1);
        return;
    }
    
    envMap.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        envMap[i] = RGB(data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]);
    }
    stbi_image_free(data);
    
    // Build importance sampling distribution
    std::vector<float> luminance(width * height);
    for (int y = 0; y < height; ++y) {
        float sinTheta = std::sin(glm::pi<float>() * float(y + 0.5f) / float(height));
        for (int x = 0; x < width; ++x) {
            RGB rgb = envMap[y * width + x];
            float lum = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
            luminance[y * width + x] = lum * sinTheta;  // Weight by solid angle
        }
    }
    
    distribution = std::make_shared<Distribution2D>(luminance, width, height);
}

glm::vec3 InfiniteLight::SampleLi(const glm::vec2& u, float* pdf) const {
    if (!distribution) {
        // Uniform sampling
        float phi = u.x * 2 * glm::pi<float>();
        float cosTheta = 1 - 2 * u.y;
        float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        glm::vec3 dir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
        if (pdf) *pdf = 1.0f / (4 * glm::pi<float>());
        return dir;
    }
    
    float mapPdf;
    glm::vec2 uv = distribution->Sample(u, &mapPdf);
    if (mapPdf == 0) return glm::vec3(0, 0, 1);
    
    // UV to direction
    float theta = uv.y * glm::pi<float>();
    float phi = uv.x * 2 * glm::pi<float>();
    float cosTheta = std::cos(theta);
    float sinTheta = std::sin(theta);
    glm::vec3 dir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
    
    // Transform to world space
    dir = glm::normalize(glm::vec3(glm::inverse(worldToLight) * glm::vec4(dir, 0.0f)));
    
    // PDF in solid angle
    if (pdf) {
        *pdf = mapPdf / (2 * glm::pi<float>() * glm::pi<float>() * sinTheta);
        if (sinTheta == 0) *pdf = 0;
    }
    
    return dir;
}

float InfiniteLight::Pdf(const glm::vec3& dir) const {
    if (!distribution) return 1.0f / (4 * glm::pi<float>());
    
    glm::vec3 d = glm::normalize(glm::vec3(worldToLight * glm::vec4(dir, 0.0f)));
    float theta = std::acos(std::clamp(d.z, -1.0f, 1.0f));
    float phi = std::atan2(d.y, d.x);
    if (phi < 0) phi += 2 * glm::pi<float>();
    
    glm::vec2 uv(phi / (2 * glm::pi<float>()), theta / glm::pi<float>());
    float sinTheta = std::sin(theta);
    if (sinTheta == 0) return 0;
    
    return distribution->Pdf(uv) / (2 * glm::pi<float>() * glm::pi<float>() * sinTheta);
}

// Light factories
std::shared_ptr<Light> CreatePointLight(const glm::vec3& position, const Spectrum& I, float scale) {
    return std::make_shared<PointLight>(position, I, scale);
}

std::shared_ptr<Light> CreateDistantLight(const glm::vec3& direction, const Spectrum& L, float scale) {
    return std::make_shared<DistantLight>(direction, L, scale);
}

std::shared_ptr<Light> CreateSpotLight(const glm::vec3& position, const glm::vec3& direction,
                                        const Spectrum& I, float coneAngle, float coneDeltaAngle, float scale) {
    return std::make_shared<SpotLight>(position, direction, I, coneAngle, coneDeltaAngle, scale);
}

std::shared_ptr<Light> CreateInfiniteLight(const Spectrum& L, float scale) {
    return std::make_shared<InfiniteLight>(L, scale);
}

std::shared_ptr<Light> CreateInfiniteLightFromTexture(const std::string& texturePath, float scale) {
    return std::make_shared<InfiniteLight>(texturePath, scale);
}

std::shared_ptr<Light> CreateGoniometricLight(const glm::vec3& position, const Spectrum& I, float scale) {
    return std::make_shared<GoniometricLight>(position, I, scale);
}

std::shared_ptr<Light> CreateProjectionLight(const glm::vec3& position, const glm::vec3& direction,
                                              const Spectrum& I, float fov, float scale) {
    return std::make_shared<ProjectionLight>(position, direction, I, fov, scale);
}

std::shared_ptr<Light> CreateDiffuseAreaLight(const Spectrum& L, float scale, bool twoSided) {
    return std::make_shared<DiffuseAreaLight>(L, scale, twoSided);
}

} // namespace pbrt
