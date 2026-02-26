#include "Texture.h"
#include <stb_image.h>
#include <iostream>

namespace pbrt {

// FloatImageTexture
FloatImageTexture::FloatImageTexture(std::shared_ptr<TextureMapping2D> mapping,
                                     const std::string& filename, bool invert)
    : mapping(mapping), width(0), height(0), invert(invert) {
    
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Failed to load float image texture: " << filename << std::endl;
        return;
    }
    
    pixels.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = data[i] / 255.0f;
    }
    
    stbi_image_free(data);
}

// SpectrumImageTexture
SpectrumImageTexture::SpectrumImageTexture(std::shared_ptr<TextureMapping2D> mapping,
                                           const std::string& filename,
                                           const RGBColorSpace* cs)
    : mapping(mapping), width(0), height(0), colorspace(cs) {
    
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "Failed to load spectrum image texture: " << filename << std::endl;
        return;
    }
    
    pixels.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = RGB(
            data[i * 3 + 0] / 255.0f,
            data[i * 3 + 1] / 255.0f,
            data[i * 3 + 2] / 255.0f
        );
    }
    
    stbi_image_free(data);
}

// Factories
std::shared_ptr<FloatTexture> CreateFloatConstantTexture(float value) {
    return std::make_shared<FloatConstantTexture>(value);
}

std::shared_ptr<FloatTexture> CreateFloatImageTexture(const std::string& filename,
                                                       std::shared_ptr<TextureMapping2D> mapping) {
    if (!mapping)
        mapping = std::make_shared<UVMapping>();
    return std::make_shared<FloatImageTexture>(mapping, filename);
}

std::shared_ptr<SpectrumTexture> CreateSpectrumConstantTexture(const Spectrum& value) {
    return std::make_shared<SpectrumConstantTexture>(value);
}

std::shared_ptr<SpectrumTexture> CreateSpectrumImageTexture(const std::string& filename,
                                                             std::shared_ptr<TextureMapping2D> mapping,
                                                             const RGBColorSpace* cs) {
    if (!mapping)
        mapping = std::make_shared<UVMapping>();
    return std::make_shared<SpectrumImageTexture>(mapping, filename, cs);
}

} // namespace pbrt
