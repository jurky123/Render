#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <algorithm>
#include <cmath>

// Simplified spectrum system aligned with pbrt-v4
// Supports: Constant, RGB (Albedo/Unbounded/Illuminant), Blackbody, PiecewiseLinear, DenselySampled

namespace pbrt {

// Constants
constexpr float Lambda_min = 360.0f;
constexpr float Lambda_max = 830.0f;
constexpr int NSpectrumSamples = 4;
constexpr float CIE_Y_integral = 106.856895f;

// Forward declarations
class RGBColorSpace;
class SampledWavelengths;
class SampledSpectrum;

// RGB class
struct RGB {
    float r = 0, g = 0, b = 0;
    
    RGB() = default;
    RGB(float r, float g, float b) : r(r), g(g), b(b) {}
    
    RGB operator+(const RGB& o) const { return RGB(r + o.r, g + o.g, b + o.b); }
    RGB operator-(const RGB& o) const { return RGB(r - o.r, g - o.g, b - o.b); }
    RGB operator*(const RGB& o) const { return RGB(r * o.r, g * o.g, b * o.b); }
    RGB operator*(float s) const { return RGB(r * s, g * s, b * s); }
    RGB operator/(float s) const { return RGB(r / s, g / s, b / s); }
    
    RGB& operator+=(const RGB& o) { r += o.r; g += o.g; b += o.b; return *this; }
    RGB& operator*=(float s) { r *= s; g *= s; b *= s; return *this; }
    
    bool operator==(const RGB& o) const { return r == o.r && g == o.g && b == o.b; }
    
    float MaxComponent() const { return std::max({r, g, b}); }
    float Average() const { return (r + g + b) / 3.0f; }
};

inline RGB operator*(float s, const RGB& rgb) { return rgb * s; }

// XYZ class
struct XYZ {
    float X = 0, Y = 0, Z = 0;
    XYZ() = default;
    XYZ(float x, float y, float z) : X(x), Y(y), Z(z) {}
};

// RGBColorSpace  
class RGBColorSpace {
public:
    // sRGB primaries & D65 white point
    static const RGBColorSpace* sRGB;
    
    RGBColorSpace();
    
    RGB ToRGB(const XYZ& xyz) const;
    XYZ ToXYZ(const RGB& rgb) const;
    
    // Conversion matrix (simplified, identity for now)
    glm::mat3 XYZFromRGB = glm::mat3(1.0f);
    glm::mat3 RGBFromXYZ = glm::mat3(1.0f);
};

// SampledWavelengths
class SampledWavelengths {
public:
    std::array<float, NSpectrumSamples> lambda;
    std::array<float, NSpectrumSamples> pdf;
    
    SampledWavelengths() {
        lambda.fill(550.0f);
        pdf.fill(1.0f / (Lambda_max - Lambda_min));
    }
    
    static SampledWavelengths SampleUniform(float u) {
        SampledWavelengths swl;
        swl.lambda[0] = Lambda_min + u * (Lambda_max - Lambda_min);
        float delta = (Lambda_max - Lambda_min) / NSpectrumSamples;
        for (int i = 1; i < NSpectrumSamples; ++i) {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            if (swl.lambda[i] > Lambda_max)
                swl.lambda[i] = Lambda_min + (swl.lambda[i] - Lambda_max);
        }
        for (int i = 0; i < NSpectrumSamples; ++i)
            swl.pdf[i] = 1.0f / (Lambda_max - Lambda_min);
        return swl;
    }
    
    float operator[](int i) const { return lambda[i]; }
};

// SampledSpectrum
class SampledSpectrum {
public:
    std::array<float, NSpectrumSamples> values;
    
    SampledSpectrum() { values.fill(0.0f); }
    SampledSpectrum(float c) { values.fill(c); }
    
    float& operator[](int i) { return values[i]; }
    float operator[](int i) const { return values[i]; }
    
    SampledSpectrum operator+(const SampledSpectrum& o) const {
        SampledSpectrum r;
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = values[i] + o.values[i];
        return r;
    }
    
    SampledSpectrum operator*(const SampledSpectrum& o) const {
        SampledSpectrum r;
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = values[i] * o.values[i];
        return r;
    }
    
    SampledSpectrum operator*(float s) const {
        SampledSpectrum r;
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = values[i] * s;
        return r;
    }
    
    float Average() const {
        float sum = 0;
        for (int i = 0; i < NSpectrumSamples; ++i) sum += values[i];
        return sum / NSpectrumSamples;
    }
    
    float MaxComponent() const {
        return *std::max_element(values.begin(), values.end());
    }
    
    RGB ToRGB(const SampledWavelengths& lambda, const RGBColorSpace& cs) const;
    XYZ ToXYZ(const SampledWavelengths& lambda) const;
};

inline SampledSpectrum operator*(float s, const SampledSpectrum& sp) { return sp * s; }

// Blackbody helper
inline float Blackbody(float lambda, float T) {
    if (T <= 0) return 0;
    const float c = 299792458.0f;
    const float h = 6.62606957e-34f;
    const float kb = 1.3806488e-23f;
    float l = lambda * 1e-9f;
    float Le = (2 * h * c * c) / (std::pow(l, 5) * (std::exp((h * c) / (l * kb * T)) - 1));
    return Le;
}

// Spectrum base type (variant-like)
enum class SpectrumType {
    Constant,
    DenselySampled,
    PiecewiseLinear,
    RGBAlbedo,
    RGBUnbounded,
    RGBIlluminant,
    Blackbody
};

// Base spectrum handles
class SpectrumHandle {
public:
    SpectrumType type;
    virtual ~SpectrumHandle() = default;
    virtual float Evaluate(float lambda) const = 0;
    virtual SampledSpectrum Sample(const SampledWavelengths& lambda) const = 0;
    virtual float MaxValue() const = 0;
};

class ConstantSpectrum : public SpectrumHandle {
public:
    float c;
    ConstantSpectrum(float c = 0) : c(c) { type = SpectrumType::Constant; }
    float Evaluate(float lambda) const override { return c; }
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override { return SampledSpectrum(c); }
    float MaxValue() const override { return c; }
};

class DenselySampledSpectrum : public SpectrumHandle {
public:
    int lambda_min, lambda_max;
    std::vector<float> values;
    
    DenselySampledSpectrum(int lmin = (int)Lambda_min, int lmax = (int)Lambda_max)
        : lambda_min(lmin), lambda_max(lmax), values(lmax - lmin + 1, 0.0f) {
        type = SpectrumType::DenselySampled;
    }
    
    float Evaluate(float lambda) const override {
        int idx = (int)std::round(lambda) - lambda_min;
        if (idx < 0 || idx >= (int)values.size()) return 0;
        return values[idx];
    }
    
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = Evaluate(lambda[i]);
        return s;
    }
    
    float MaxValue() const override {
        if (values.empty()) return 0;
        return *std::max_element(values.begin(), values.end());
    }
};

class PiecewiseLinearSpectrum : public SpectrumHandle {
public:
    std::vector<float> lambdas, values;
    
    PiecewiseLinearSpectrum() { type = SpectrumType::PiecewiseLinear; }
    
    PiecewiseLinearSpectrum(const std::vector<float>& l, const std::vector<float>& v)
        : lambdas(l), values(v) {
        type = SpectrumType::PiecewiseLinear;
    }
    
    float Evaluate(float lambda) const override {
        if (lambdas.empty() || lambda < lambdas.front() || lambda > lambdas.back())
            return 0;
        // Linear interpolation
        for (size_t i = 0; i < lambdas.size() - 1; ++i) {
            if (lambda >= lambdas[i] && lambda <= lambdas[i + 1]) {
                float t = (lambda - lambdas[i]) / (lambdas[i + 1] - lambdas[i]);
                return values[i] * (1 - t) + values[i + 1] * t;
            }
        }
        return 0;
    }
    
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = Evaluate(lambda[i]);
        return s;
    }
    
    float MaxValue() const override {
        if (values.empty()) return 0;
        return *std::max_element(values.begin(), values.end());
    }
};

class BlackbodySpectrum : public SpectrumHandle {
public:
    float T;
    float normalizationFactor;
    
    BlackbodySpectrum(float T = 6500) : T(T) {
        type = SpectrumType::Blackbody;
        // Normalize to peak = 1
        float lambdaMax = 2.8977721e-3f / T * 1e9f; // Wien's displacement law
        normalizationFactor = 1.0f / Blackbody(lambdaMax, T);
    }
    
    float Evaluate(float lambda) const override {
        return Blackbody(lambda, T) * normalizationFactor;
    }
    
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = Evaluate(lambda[i]);
        return s;
    }
    
    float MaxValue() const override { return 1.0f; }
};

// RGBAlbedoSpectrum (for reflectances, clamped [0,1])
class RGBAlbedoSpectrum : public SpectrumHandle {
public:
    RGB rgb;
    const RGBColorSpace* cs;
    
    RGBAlbedoSpectrum(const RGBColorSpace* cs, const RGB& rgb)
        : rgb(Clamp(rgb)), cs(cs) {
        type = SpectrumType::RGBAlbedo;
    }
    
    static RGB Clamp(const RGB& rgb) {
        return RGB(std::clamp(rgb.r, 0.0f, 1.0f),
                   std::clamp(rgb.g, 0.0f, 1.0f),
                   std::clamp(rgb.b, 0.0f, 1.0f));
    }
    
    float Evaluate(float lambda) const override {
        // Simplified: return average
        return rgb.Average();
    }
    
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override {
        return SampledSpectrum(rgb.Average());
    }
    
    float MaxValue() const override { return rgb.MaxComponent(); }
};

// RGBUnboundedSpectrum (for emitters, can exceed 1)
class RGBUnboundedSpectrum : public SpectrumHandle {
public:
    RGB rgb;
    const RGBColorSpace* cs;
    float scale;
    
    RGBUnboundedSpectrum(const RGBColorSpace* cs, const RGB& rgb)
        : rgb(rgb), cs(cs), scale(1.0f) {
        type = SpectrumType::RGBUnbounded;
    }
    
    float Evaluate(float lambda) const override {
        return scale * rgb.Average();
    }
    
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override {
        return SampledSpectrum(scale * rgb.Average());
    }
    
    float MaxValue() const override { return scale * rgb.MaxComponent(); }
};

// RGBIlluminantSpectrum (RGB modulated by illuminant)
class RGBIlluminantSpectrum : public SpectrumHandle {
public:
    RGB rgb;
    const RGBColorSpace* cs;
    float scale;
    std::shared_ptr<DenselySampledSpectrum> illuminant;
    
    RGBIlluminantSpectrum(const RGBColorSpace* cs, const RGB& rgb)
        : rgb(rgb), cs(cs), scale(1.0f) {
        type = SpectrumType::RGBIlluminant;
        // Default to D65 approximation
        illuminant = std::make_shared<DenselySampledSpectrum>();
        for (int i = 0; i < (int)illuminant->values.size(); ++i)
            illuminant->values[i] = 1.0f; // Flat for now
    }
    
    float Evaluate(float lambda) const override {
        if (!illuminant) return 0;
        return scale * rgb.Average() * illuminant->Evaluate(lambda);
    }
    
    SampledSpectrum Sample(const SampledWavelengths& lambda) const override {
        if (!illuminant) return SampledSpectrum(0);
        SampledSpectrum s(scale * rgb.Average());
        return s * illuminant->Sample(lambda);
    }
    
    float MaxValue() const override {
        if (!illuminant) return 0;
        return scale * rgb.MaxComponent() * illuminant->MaxValue();
    }
};

// Spectrum wrapper (like pbrt's TaggedPointer)
class Spectrum {
public:
    std::shared_ptr<SpectrumHandle> handle;
    
    Spectrum() : handle(std::make_shared<ConstantSpectrum>(0)) {}
    Spectrum(std::shared_ptr<SpectrumHandle> h) : handle(h) {}
    
    // Factory methods
    static Spectrum CreateConstant(float c) {
        return Spectrum(std::make_shared<ConstantSpectrum>(c));
    }
    
    static Spectrum CreateRGBAlbedo(const RGBColorSpace* cs, const RGB& rgb) {
        return Spectrum(std::make_shared<RGBAlbedoSpectrum>(cs, rgb));
    }
    
    static Spectrum CreateRGBUnbounded(const RGBColorSpace* cs, const RGB& rgb) {
        return Spectrum(std::make_shared<RGBUnboundedSpectrum>(cs, rgb));
    }
    
    static Spectrum CreateRGBIlluminant(const RGBColorSpace* cs, const RGB& rgb) {
        return Spectrum(std::make_shared<RGBIlluminantSpectrum>(cs, rgb));
    }
    
    static Spectrum CreateBlackbody(float T) {
        return Spectrum(std::make_shared<BlackbodySpectrum>(T));
    }
    
    static Spectrum CreatePiecewiseLinear(const std::vector<float>& lambdas, const std::vector<float>& values) {
        return Spectrum(std::make_shared<PiecewiseLinearSpectrum>(lambdas, values));
    }
    
    float operator()(float lambda) const { return handle->Evaluate(lambda); }
    SampledSpectrum Sample(const SampledWavelengths& lambda) const { return handle->Sample(lambda); }
    float MaxValue() const { return handle->MaxValue(); }
    
    // Conversion to RGB (simplified)
    RGB ToRGB(const RGBColorSpace* cs = RGBColorSpace::sRGB) const {
        SampledWavelengths lambda = SampledWavelengths::SampleUniform(0.5f);
        SampledSpectrum s = handle->Sample(lambda);
        float avg = s.Average();
        return RGB(avg, avg, avg); // Simplified
    }
};

// Helper: Load spectrum from SPD file
Spectrum LoadSpectrumFile(const std::string& filename);

} // namespace pbrt
