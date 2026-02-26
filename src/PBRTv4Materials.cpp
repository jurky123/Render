#include "PBRTv4Materials.h"

namespace pbrt {

// Material factory implementations

std::shared_ptr<Material> CreateDiffuseMaterial(
    std::shared_ptr<SpectrumTexture> reflectance,
    std::shared_ptr<FloatTexture> sigma,
    std::shared_ptr<SpectrumTexture> displacement,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<DiffuseMaterial>(reflectance, sigma, displacement, normalmap);
}

std::shared_ptr<Material> CreateConductorMaterial(
    std::shared_ptr<SpectrumTexture> eta,
    std::shared_ptr<SpectrumTexture> k,
    std::shared_ptr<FloatTexture> uroughness,
    std::shared_ptr<FloatTexture> vroughness,
    bool remaproughness,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<ConductorMaterial>(eta, k, uroughness, vroughness, remaproughness, normalmap);
}

std::shared_ptr<Material> CreateDielectricMaterial(
    float eta,
    std::shared_ptr<SpectrumTexture> reflectance,
    std::shared_ptr<SpectrumTexture> transmittance,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<DielectricMaterial>(eta, reflectance, transmittance, normalmap);
}

std::shared_ptr<Material> CreateRoughDielectricMaterial(
    float eta,
    std::shared_ptr<FloatTexture> uroughness,
    std::shared_ptr<FloatTexture> vroughness,
    bool remaproughness,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<RoughDielectricMaterial>(eta, uroughness, vroughness, remaproughness, normalmap);
}

std::shared_ptr<Material> CreateCoatedDiffuseMaterial(
    std::shared_ptr<SpectrumTexture> reflectance,
    float interface_eta,
    std::shared_ptr<FloatTexture> interface_uroughness,
    std::shared_ptr<FloatTexture> interface_vroughness,
    bool remaproughness,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<CoatedDiffuseMaterial>(reflectance, interface_eta,
                                                     interface_uroughness, interface_vroughness,
                                                     remaproughness, normalmap);
}

std::shared_ptr<Material> CreateCoatedConductorMaterial(
    std::shared_ptr<SpectrumTexture> conductor_eta,
    std::shared_ptr<SpectrumTexture> conductor_k,
    float interface_eta,
    std::shared_ptr<FloatTexture> conductor_uroughness,
    std::shared_ptr<FloatTexture> conductor_vroughness,
    std::shared_ptr<FloatTexture> interface_uroughness,
    std::shared_ptr<FloatTexture> interface_vroughness,
    bool remaproughness,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<CoatedConductorMaterial>(conductor_eta, conductor_k, interface_eta,
                                                       conductor_uroughness, conductor_vroughness,
                                                       interface_uroughness, interface_vroughness,
                                                       remaproughness, normalmap);
}

std::shared_ptr<Material> CreateSubsurfaceMaterial(
    const Spectrum& sigma_a,
    const Spectrum& sigma_s,
    float scale,
    float eta,
    std::shared_ptr<FloatTexture> uroughness,
    std::shared_ptr<FloatTexture> vroughness,
    bool remaproughness,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<SubsurfaceMaterial>(sigma_a, sigma_s, scale, eta,
                                                 uroughness, vroughness, remaproughness, normalmap);
}

std::shared_ptr<Material> CreateHairMaterial(
    const RGB& color,
    float eumelanin,
    float pheomelanin,
    float eta,
    float beta_m,
    float beta_n,
    float alpha,
    std::shared_ptr<FloatTexture> normalmap) {
    return std::make_shared<HairMaterial>(color, eumelanin, pheomelanin, eta, beta_m, beta_n, alpha, normalmap);
}

std::shared_ptr<Material> CreateMixMaterial(
    std::shared_ptr<Material> mat1,
    std::shared_ptr<Material> mat2,
    std::shared_ptr<FloatTexture> amount) {
    return std::make_shared<MixMaterial>(mat1, mat2, amount);
}

} // namespace pbrt
