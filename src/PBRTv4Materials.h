#pragma once

#include "Spectrum.h"
#include "Texture.h"
#include <glm/glm.hpp>
#include <memory>
#include <string>

// Complete pbrt-v4 material system aligned with official implementation
// Supports all materials: diffuse, conductor, dielectric, coateddiffuse, coatedconductor, etc.

namespace pbrt {

//前向声明
class MaterialEvalContext;

// Material types matching pbrt-v4
enum class MaterialType {
    Diffuse,
    Conductor,
    Dielectric,
    RoughDielectric,
    CoatedDiffuse,
    CoatedConductor,
    Subsurface,
    Hair,
    Measured,
    Mix,
    Interface
};

// Material interface
class Material {
public:
    virtual ~Material() = default;
    virtual MaterialType GetType() const = 0;
    virtual std::string ToString() const = 0;
};

// DiffuseMaterial ("diffuse")
class DiffuseMaterial : public Material {
public:
    std::shared_ptr<SpectrumTexture> reflectance;
    std::shared_ptr<FloatTexture> sigma;  // roughness (Oren-Nayar)
    std::shared_ptr<SpectrumTexture> displacement;
    std::shared_ptr<FloatTexture> normalmap;
    
    DiffuseMaterial(std::shared_ptr<SpectrumTexture> refl,
                    std::shared_ptr<FloatTexture> sigma = nullptr,
                    std::shared_ptr<SpectrumTexture> displacement = nullptr,
                    std::shared_ptr<FloatTexture> normalmap = nullptr)
        : reflectance(refl), sigma(sigma), displacement(displacement), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::Diffuse; }
    std::string ToString() const override { return "DiffuseMaterial"; }
};

// ConductorMaterial ("conductor")
class ConductorMaterial : public Material {
public:
    std::shared_ptr<SpectrumTexture> eta;        // IOR
    std::shared_ptr<SpectrumTexture> k;          // extinction coefficient
    std::shared_ptr<FloatTexture> uroughness;    // surface roughness u
    std::shared_ptr<FloatTexture> vroughness;    // surface roughness v
    std::shared_ptr<SpectrumTexture> reflectance; // optional artist-friendly
    bool remaproughness;
    std::shared_ptr<FloatTexture> normalmap;
    
    ConductorMaterial(std::shared_ptr<SpectrumTexture> eta,
                      std::shared_ptr<SpectrumTexture> k,
                      std::shared_ptr<FloatTexture> urough,
                      std::shared_ptr<FloatTexture> vrough,
                      bool remaproughness = true,
                      std::shared_ptr<FloatTexture> normalmap = nullptr)
        : eta(eta), k(k), uroughness(urough), vroughness(vrough),
          remaproughness(remaproughness), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::Conductor; }
    std::string ToString() const override { return "ConductorMaterial"; }
};

// DielectricMaterial ("dielectric")
class DielectricMaterial : public Material {
public:
    float eta;  // IOR
    std::shared_ptr<SpectrumTexture> reflectance;
    std::shared_ptr<SpectrumTexture> transmittance;
    std::shared_ptr<FloatTexture> normalmap;
    
    DielectricMaterial(float eta = 1.5f,
                       std::shared_ptr<SpectrumTexture> refl = nullptr,
                       std::shared_ptr<SpectrumTexture> trans = nullptr,
                       std::shared_ptr<FloatTexture> normalmap = nullptr)
        : eta(eta), reflectance(refl), transmittance(trans), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::Dielectric; }
    std::string ToString() const override { return "DielectricMaterial"; }
};

// RoughDielectricMaterial ("roughdielectric" / "thindielectric")
class RoughDielectricMaterial : public Material {
public:
    float eta;
    std::shared_ptr<FloatTexture> uroughness, vroughness;
    std::shared_ptr<SpectrumTexture> reflectance, transmittance;
    bool remaproughness;
    std::shared_ptr<FloatTexture> normalmap;
    
    RoughDielectricMaterial(float eta = 1.5f,
                            std::shared_ptr<FloatTexture> urough = nullptr,
                            std::shared_ptr<FloatTexture> vrough = nullptr,
                            bool remaproughness = true,
                            std::shared_ptr<FloatTexture> normalmap = nullptr)
        : eta(eta), uroughness(urough), vroughness(vrough),
          remaproughness(remaproughness), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::RoughDielectric; }
    std::string ToString() const override { return "RoughDielectricMaterial"; }
};

// CoatedDiffuseMaterial ("coateddiffuse")
class CoatedDiffuseMaterial : public Material {
public:
    std::shared_ptr<SpectrumTexture> reflectance;  // base diffuse
    std::shared_ptr<FloatTexture> uroughness, vroughness;  // coating roughness
    float eta;  // coating IOR
    float thickness;
    std::shared_ptr<SpectrumTexture> albedo;  // coating absorption
    std::shared_ptr<FloatTexture> g;  // anisotropy
    bool remaproughness;
    std::shared_ptr<FloatTexture> normalmap;
    
    CoatedDiffuseMaterial(std::shared_ptr<SpectrumTexture> refl,
                          float eta = 1.5f,
                          std::shared_ptr<FloatTexture> urough = nullptr,
                          std::shared_ptr<FloatTexture> vrough = nullptr,
                          bool remaproughness = true,
                          std::shared_ptr<FloatTexture> normalmap = nullptr)
        : reflectance(refl), eta(eta), uroughness(urough), vroughness(vrough),
          thickness(0.01f), g(nullptr), remaproughness(remaproughness), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::CoatedDiffuse; }
    std::string ToString() const override { return "CoatedDiffuseMaterial"; }
};

// CoatedConductorMaterial ("coatedconductor")
class CoatedConductorMaterial : public Material {
public:
    std::shared_ptr<SpectrumTexture> conductor_eta, conductor_k;
    std::shared_ptr<FloatTexture> conductor_uroughness, conductor_vroughness;
    std::shared_ptr<FloatTexture> interface_uroughness, interface_vroughness;
    float interface_eta;
    float thickness;
    std::shared_ptr<SpectrumTexture> albedo;
    std::shared_ptr<FloatTexture> g;
    bool remaproughness;
    std::shared_ptr<FloatTexture> normalmap;
    
    CoatedConductorMaterial(std::shared_ptr<SpectrumTexture> eta,
                            std::shared_ptr<SpectrumTexture> k,
                            float interface_eta = 1.5f,
                            std::shared_ptr<FloatTexture> conductor_urough = nullptr,
                            std::shared_ptr<FloatTexture> conductor_vrough = nullptr,
                            std::shared_ptr<FloatTexture> interface_urough = nullptr,
                            std::shared_ptr<FloatTexture> interface_vrough = nullptr,
                            bool remaproughness = true,
                            std::shared_ptr<FloatTexture> normalmap = nullptr)
        : conductor_eta(eta), conductor_k(k), interface_eta(interface_eta),
          conductor_uroughness(conductor_urough), conductor_vroughness(conductor_vrough),
          interface_uroughness(interface_urough), interface_vroughness(interface_vrough),
          thickness(0.01f), g(nullptr), remaproughness(remaproughness), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::CoatedConductor; }
    std::string ToString() const override { return "CoatedConductorMaterial"; }
};

// SubsurfaceMaterial ("subsurface")
class SubsurfaceMaterial : public Material {
public:
    Spectrum sigma_a, sigma_s;  // absorption, scattering
    float scale, eta;
    std::shared_ptr<SpectrumTexture> reflectance;
    std::shared_ptr<FloatTexture> uroughness, vroughness;
    bool remaproughness;
    std::shared_ptr<FloatTexture> normalmap;
    
    SubsurfaceMaterial(const Spectrum& sigma_a, const Spectrum& sigma_s,
                       float scale = 1.0f, float eta = 1.3f,
                       std::shared_ptr<FloatTexture> urough = nullptr,
                       std::shared_ptr<FloatTexture> vrough = nullptr,
                       bool remaproughness = true,
                       std::shared_ptr<FloatTexture> normalmap = nullptr)
        : sigma_a(sigma_a), sigma_s(sigma_s), scale(scale), eta(eta),
          uroughness(urough), vroughness(vrough),
          remaproughness(remaproughness), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::Subsurface; }
    std::string ToString() const override { return "SubsurfaceMaterial"; }
};

// HairMaterial ("hair")
class HairMaterial : public Material {
public:
    Spectrum sigma_a;
    RGB color;
    float eumelanin, pheomelanin;
    float eta, beta_m, beta_n, alpha;
    std::shared_ptr<FloatTexture> normalmap;
    
    HairMaterial(const RGB& color = RGB(0.5, 0.5, 0.5),
                 float eumelanin = 1.3f, float pheomelanin = 0.0f,
                 float eta = 1.55f, float beta_m = 0.3f, float beta_n = 0.3f, float alpha = 2.0f,
                 std::shared_ptr<FloatTexture> normalmap = nullptr)
        : color(color), eumelanin(eumelanin), pheomelanin(pheomelanin),
          eta(eta), beta_m(beta_m), beta_n(beta_n), alpha(alpha), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::Hair; }
    std::string ToString() const override { return "HairMaterial"; }
};

// MixMaterial ("mix")
class MixMaterial : public Material {
public:
    std::shared_ptr<Material> mat1, mat2;
    std::shared_ptr<FloatTexture> amount;
    
    MixMaterial(std::shared_ptr<Material> m1, std::shared_ptr<Material> m2,
                std::shared_ptr<FloatTexture> amt)
        : mat1(m1), mat2(m2), amount(amt) {}
    
    MaterialType GetType() const override { return MaterialType::Mix; }
    std::string ToString() const override { return "MixMaterial"; }
};

// InterfaceMaterial ("interface") - for layered materials
class InterfaceMaterial : public Material {
public:
    std::shared_ptr<Material> material;
    std::shared_ptr<SpectrumTexture> displacement;
    std::shared_ptr<FloatTexture> normalmap;
    
    InterfaceMaterial(std::shared_ptr<Material> mat,
                      std::shared_ptr<SpectrumTexture> displacement = nullptr,
                      std::shared_ptr<FloatTexture> normalmap = nullptr)
        : material(mat), displacement(displacement), normalmap(normalmap) {}
    
    MaterialType GetType() const override { return MaterialType::Interface; }
    std::string ToString() const override { return "InterfaceMaterial"; }
};

// Material factories (will be called by PBRTLoader)
std::shared_ptr<Material> CreateDiffuseMaterial(
    std::shared_ptr<SpectrumTexture> reflectance,
    std::shared_ptr<FloatTexture> sigma = nullptr,
    std::shared_ptr<SpectrumTexture> displacement = nullptr,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateConductorMaterial(
    std::shared_ptr<SpectrumTexture> eta,
    std::shared_ptr<SpectrumTexture> k,
    std::shared_ptr<FloatTexture> uroughness,
    std::shared_ptr<FloatTexture> vroughness,
    bool remaproughness = true,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateDielectricMaterial(
    float eta = 1.5f,
    std::shared_ptr<SpectrumTexture> reflectance = nullptr,
    std::shared_ptr<SpectrumTexture> transmittance = nullptr,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateRoughDielectricMaterial(
    float eta,
    std::shared_ptr<FloatTexture> uroughness,
    std::shared_ptr<FloatTexture> vroughness,
    bool remaproughness = true,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateCoatedDiffuseMaterial(
    std::shared_ptr<SpectrumTexture> reflectance,
    float interface_eta,
    std::shared_ptr<FloatTexture> interface_uroughness,
    std::shared_ptr<FloatTexture> interface_vroughness,
    bool remaproughness = true,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateCoatedConductorMaterial(
    std::shared_ptr<SpectrumTexture> conductor_eta,
    std::shared_ptr<SpectrumTexture> conductor_k,
    float interface_eta,
    std::shared_ptr<FloatTexture> conductor_uroughness,
    std::shared_ptr<FloatTexture> conductor_vroughness,
    std::shared_ptr<FloatTexture> interface_uroughness,
    std::shared_ptr<FloatTexture> interface_vroughness,
    bool remaproughness = true,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateSubsurfaceMaterial(
    const Spectrum& sigma_a,
    const Spectrum& sigma_s,
    float scale,
    float eta,
    std::shared_ptr<FloatTexture> uroughness,
    std::shared_ptr<FloatTexture> vroughness,
    bool remaproughness = true,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateHairMaterial(
    const RGB& color,
    float eumelanin,
    float pheomelanin,
    float eta,
    float beta_m,
    float beta_n,
    float alpha,
    std::shared_ptr<FloatTexture> normalmap = nullptr);

std::shared_ptr<Material> CreateMixMaterial(
    std::shared_ptr<Material> mat1,
    std::shared_ptr<Material> mat2,
    std::shared_ptr<FloatTexture> amount);

} // namespace pbrt
