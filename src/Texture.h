#pragma once

#include "Spectrum.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <memory>
#include <string>
#include <functional>

// pbrt-v4 style texture system
// Supports: Float & Spectrum textures with various mappings and operations

namespace pbrt {

// Forward declarations
class TextureMapping2D;
class TextureMapping3D;
struct TextureEvalContext;

// TextureEvalContext (shading point info)
struct TextureEvalContext {
    glm::vec3 p;        // Position
    glm::vec3 dpdx;     // Position derivatives
    glm::vec3 dpdy;
    glm::vec2 uv;       // Texture coordinates
    glm::vec2 dudx, dvdy; // UV derivatives
    glm::vec3 n;        // Normal
    
    TextureEvalContext() = default;
    TextureEvalContext(const glm::vec3& p, const glm::vec2& uv = glm::vec2(0))
        : p(p), uv(uv), dpdx(0), dpdy(0), dudx(0), dvdy(0), n(0, 0, 1) {}
};

// Base texture interfaces
class FloatTexture {
public:
    virtual ~FloatTexture() = default;
    virtual float Evaluate(const TextureEvalContext& ctx) const = 0;
};

class SpectrumTexture {
public:
    virtual ~SpectrumTexture() = default;
    virtual Spectrum Evaluate(const TextureEvalContext& ctx) const = 0;
};

// Texture Mappings
enum class TextureMappingType {
    UV,
    Spherical,
    Cylindrical,
    Planar,
    Point3D
};

class TextureMapping2D {
public:
    virtual ~TextureMapping2D() = default;
    virtual glm::vec2 Map(const TextureEvalContext& ctx, glm::vec2* dstdx = nullptr, glm::vec2* dstdy = nullptr) const = 0;
};

class UVMapping : public TextureMapping2D {
public:
    float su, sv, du, dv;
    
    UVMapping(float su = 1, float sv = 1, float du = 0, float dv = 0)
        : su(su), sv(sv), du(du), dv(dv) {}
    
    glm::vec2 Map(const TextureEvalContext& ctx, glm::vec2* dstdx = nullptr, glm::vec2* dstdy = nullptr) const override {
        if (dstdx) *dstdx = glm::vec2(su, 0) * ctx.dudx;
        if (dstdy) *dstdy = glm::vec2(0, sv) * ctx.dvdy;
        return glm::vec2(su * ctx.uv.x + du, sv * ctx.uv.y + dv);
    }
};

class SphericalMapping : public TextureMapping2D {
public:
    glm::mat4 worldToTexture;
    
    SphericalMapping(const glm::mat4& worldToTexture = glm::mat4(1))
        : worldToTexture(worldToTexture) {}
    
    glm::vec2 Map(const TextureEvalContext& ctx, glm::vec2* dstdx = nullptr, glm::vec2* dstdy = nullptr) const override {
        glm::vec3 dir = glm::normalize(glm::vec3(worldToTexture * glm::vec4(ctx.p, 1.0f)));
        float theta = std::acos(std::clamp(dir.z, -1.0f, 1.0f));
        float phi = std::atan2(dir.y, dir.x);
        if (phi < 0) phi += 2 * glm::pi<float>();
        return glm::vec2(phi / (2 * glm::pi<float>()), theta / glm::pi<float>());
    }
};

class CylindricalMapping : public TextureMapping2D {
public:
    glm::mat4 worldToTexture;
    
    CylindricalMapping(const glm::mat4& worldToTexture = glm::mat4(1))
        : worldToTexture(worldToTexture) {}
    
    glm::vec2 Map(const TextureEvalContext& ctx, glm::vec2* dstdx = nullptr, glm::vec2* dstdy = nullptr) const override {
        glm::vec3 p = glm::vec3(worldToTexture * glm::vec4(ctx.p, 1.0f));
        float phi = std::atan2(p.y, p.x);
        if (phi < 0) phi += 2 * glm::pi<float>();
        return glm::vec2(phi / (2 * glm::pi<float>()), p.z);
    }
};

class PlanarMapping : public TextureMapping2D {
public:
    glm::vec3 vs, vt;
    float ds, dt;
    
    PlanarMapping(const glm::vec3& vs = glm::vec3(1, 0, 0), const glm::vec3& vt = glm::vec3(0, 1, 0),
                  float ds = 0, float dt = 0)
        : vs(vs), vt(vt), ds(ds), dt(dt) {}
    
    glm::vec2 Map(const TextureEvalContext& ctx, glm::vec2* dstdx = nullptr, glm::vec2* dstdy = nullptr) const override {
        return glm::vec2(glm::dot(ctx.p, vs) + ds, glm::dot(ctx.p, vt) + dt);
    }
};

class TextureMapping3D {
public:
    virtual ~TextureMapping3D() = default;
    virtual glm::vec3 Map(const TextureEvalContext& ctx) const = 0;
};

class IdentityMapping3D : public TextureMapping3D {
public:
    glm::mat4 worldToTexture;
    
    IdentityMapping3D(const glm::mat4& worldToTexture = glm::mat4(1))
        : worldToTexture(worldToTexture) {}
    
    glm::vec3 Map(const TextureEvalContext& ctx) const override {
        return glm::vec3(worldToTexture * glm::vec4(ctx.p, 1.0f));
    }
};

// --- Float Textures ---

class FloatConstantTexture : public FloatTexture {
public:
    float value;
    FloatConstantTexture(float v) : value(v) {}
    float Evaluate(const TextureEvalContext& ctx) const override { return value; }
};

class FloatScaledTexture : public FloatTexture {
public:
    std::shared_ptr<FloatTexture> tex;
    float scale;
    
    FloatScaledTexture(std::shared_ptr<FloatTexture> tex, float scale)
        : tex(tex), scale(scale) {}
    
    float Evaluate(const TextureEvalContext& ctx) const override {
        return scale * tex->Evaluate(ctx);
    }
};

class FloatMixTexture : public FloatTexture {
public:
    std::shared_ptr<FloatTexture> tex1, tex2;
    std::shared_ptr<FloatTexture> amount; // Mix factor [0,1]
    
    FloatMixTexture(std::shared_ptr<FloatTexture> t1, std::shared_ptr<FloatTexture> t2,
                    std::shared_ptr<FloatTexture> amt)
        : tex1(t1), tex2(t2), amount(amt) {}
    
    float Evaluate(const TextureEvalContext& ctx) const override {
        float t = std::clamp(amount->Evaluate(ctx), 0.0f, 1.0f);
        return (1 - t) * tex1->Evaluate(ctx) + t * tex2->Evaluate(ctx);
    }
};

class FloatBilerpTexture : public FloatTexture {
public:
    std::shared_ptr<TextureMapping2D> mapping;
    float v00, v01, v10, v11;
    
    FloatBilerpTexture(std::shared_ptr<TextureMapping2D> mapping,
                       float v00, float v01, float v10, float v11)
        : mapping(mapping), v00(v00), v01(v01), v10(v10), v11(v11) {}
    
    float Evaluate(const TextureEvalContext& ctx) const override {
        glm::vec2 st = mapping->Map(ctx);
        float u = st.x - std::floor(st.x);
        float v = st.y - std::floor(st.y);
        return (1 - u) * (1 - v) * v00 + (1 - u) * v * v01 + u * (1 - v) * v10 + u * v * v11;
    }
};

class FloatCheckerboardTexture : public FloatTexture {
public:
    std::shared_ptr<TextureMapping2D> mapping;
    std::shared_ptr<FloatTexture> tex1, tex2;
    
    FloatCheckerboardTexture(std::shared_ptr<TextureMapping2D> mapping,
                             std::shared_ptr<FloatTexture> t1,
                             std::shared_ptr<FloatTexture> t2)
        : mapping(mapping), tex1(t1), tex2(t2) {}
    
    float Evaluate(const TextureEvalContext& ctx) const override {
        glm::vec2 st = mapping->Map(ctx);
        bool check = (int(std::floor(st.x)) + int(std::floor(st.y))) % 2 == 0;
        return check ? tex1->Evaluate(ctx) : tex2->Evaluate(ctx);
    }
};

class FloatImageTexture : public FloatTexture {
public:
    std::shared_ptr<TextureMapping2D> mapping;
    std::vector<float> pixels;
    int width, height;
    bool invert;
    
    FloatImageTexture(std::shared_ptr<TextureMapping2D> mapping,
                      const std::string& filename, bool invert = false);
    
    float Evaluate(const TextureEvalContext& ctx) const override {
        if (pixels.empty()) return 0;
        glm::vec2 st = mapping->Map(ctx);
        st.x = st.x - std::floor(st.x);
        st.y = st.y - std::floor(st.y);
        int x = std::clamp(int(st.x * width), 0, width - 1);
        int y = std::clamp(int(st.y * height), 0, height - 1);
        float v = pixels[y * width + x];
        return invert ? (1 - v) : v;
    }
};

// --- Spectrum Textures ---

class SpectrumConstantTexture : public SpectrumTexture {
public:
    Spectrum value;
    SpectrumConstantTexture(const Spectrum& v) : value(v) {}
    Spectrum Evaluate(const TextureEvalContext& ctx) const override { return value; }
};

class SpectrumScaledTexture : public SpectrumTexture {
public:
    std::shared_ptr<SpectrumTexture> tex;
    std::shared_ptr<FloatTexture> scale;
    
    SpectrumScaledTexture(std::shared_ptr<SpectrumTexture> tex, std::shared_ptr<FloatTexture> scale)
        : tex(tex), scale(scale) {}
    
    Spectrum Evaluate(const TextureEvalContext& ctx) const override {
        float s = scale->Evaluate(ctx);
        Spectrum sp = tex->Evaluate(ctx);
        // Scale spectrum (simplified)
        auto scaled = std::make_shared<ConstantSpectrum>(sp.MaxValue() * s);
        return Spectrum(scaled);
    }
};

class SpectrumMixTexture : public SpectrumTexture {
public:
    std::shared_ptr<SpectrumTexture> tex1, tex2;
    std::shared_ptr<FloatTexture> amount;
    
    SpectrumMixTexture(std::shared_ptr<SpectrumTexture> t1, std::shared_ptr<SpectrumTexture> t2,
                       std::shared_ptr<FloatTexture> amt)
        : tex1(t1), tex2(t2), amount(amt) {}
    
    Spectrum Evaluate(const TextureEvalContext& ctx) const override {
        float t = std::clamp(amount->Evaluate(ctx), 0.0f, 1.0f);
        Spectrum s1 = tex1->Evaluate(ctx);
        Spectrum s2 = tex2->Evaluate(ctx);
        // Simplified mix
        float v = (1 - t) * s1.MaxValue() + t * s2.MaxValue();
        return Spectrum::CreateConstant(v);
    }
};

class SpectrumCheckerboardTexture : public SpectrumTexture {
public:
    std::shared_ptr<TextureMapping2D> mapping;
    std::shared_ptr<SpectrumTexture> tex1, tex2;
    
    SpectrumCheckerboardTexture(std::shared_ptr<TextureMapping2D> mapping,
                                std::shared_ptr<SpectrumTexture> t1,
                                std::shared_ptr<SpectrumTexture> t2)
        : mapping(mapping), tex1(t1), tex2(t2) {}
    
    Spectrum Evaluate(const TextureEvalContext& ctx) const override {
        glm::vec2 st = mapping->Map(ctx);
        bool check = (int(std::floor(st.x)) + int(std::floor(st.y))) % 2 == 0;
        return check ? tex1->Evaluate(ctx) : tex2->Evaluate(ctx);
    }
};

class SpectrumImageTexture : public SpectrumTexture {
public:
    std::shared_ptr<TextureMapping2D> mapping;
    std::vector<RGB> pixels;
    int width, height;
    const RGBColorSpace* colorspace;
    
    SpectrumImageTexture(std::shared_ptr<TextureMapping2D> mapping,
                         const std::string& filename,
                         const RGBColorSpace* cs = RGBColorSpace::sRGB);
    
    Spectrum Evaluate(const TextureEvalContext& ctx) const override {
        if (pixels.empty()) return Spectrum::CreateConstant(0);
        glm::vec2 st = mapping->Map(ctx);
        st.x = st.x - std::floor(st.x);
        st.y = st.y - std::floor(st.y);
        int x = std::clamp(int(st.x * width), 0, width - 1);
        int y = std::clamp(int(st.y * height), 0, height - 1);
        RGB rgb = pixels[y * width + x];
        return Spectrum::CreateRGBAlbedo(colorspace, rgb);
    }
};

// Factories
std::shared_ptr<FloatTexture> CreateFloatConstantTexture(float value);
std::shared_ptr<FloatTexture> CreateFloatImageTexture(const std::string& filename,
                                                       std::shared_ptr<TextureMapping2D> mapping = nullptr);
std::shared_ptr<SpectrumTexture> CreateSpectrumConstantTexture(const Spectrum& value);
std::shared_ptr<SpectrumTexture> CreateSpectrumImageTexture(const std::string& filename,
                                                             std::shared_ptr<TextureMapping2D> mapping = nullptr,
                                                             const RGBColorSpace* cs = RGBColorSpace::sRGB);

} // namespace pbrt
