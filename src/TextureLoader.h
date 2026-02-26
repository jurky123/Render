#pragma once

#include <string>
#include <cstdint>
#include <cstring>
#include <vector>

/**
 * @brief Simple texture data container
 */
struct TextureData
{
    uint8_t* pixels = nullptr;
    int      width  = 0;
    int      height = 0;
    int      channels = 0;  // 1=gray, 2=gray+alpha, 3=RGB, 4=RGBA
    
    TextureData() = default;
    ~TextureData() { cleanup(); }
    
    void cleanup()
    {
        if (pixels)
        {
            delete[] pixels;
            pixels = nullptr;
        }
        width = height = channels = 0;
    }
    
    // Copy semantics
    TextureData(const TextureData& other)
    {
        if (other.pixels && other.width > 0 && other.height > 0)
        {
            const size_t size = static_cast<size_t>(other.width) * static_cast<size_t>(other.height) * static_cast<size_t>(other.channels);
            pixels = new uint8_t[size];
            std::memcpy(pixels, other.pixels, size);
            width = other.width;
            height = other.height;
            channels = other.channels;
        }
    }
    
    TextureData& operator=(const TextureData& other)
    {
        if (this != &other)
        {
            cleanup();
            if (other.pixels && other.width > 0 && other.height > 0)
            {
                const size_t size = static_cast<size_t>(other.width) * static_cast<size_t>(other.height) * static_cast<size_t>(other.channels);
                pixels = new uint8_t[size];
                std::memcpy(pixels, other.pixels, size);
                width = other.width;
                height = other.height;
                channels = other.channels;
            }
        }
        return *this;
    }
    
    // Move semantics
    TextureData(TextureData&& other) noexcept
        : pixels(other.pixels), width(other.width), height(other.height), channels(other.channels)
    {
        other.pixels = nullptr;
    }
    
    TextureData& operator=(TextureData&& other) noexcept
    {
        cleanup();
        pixels = other.pixels;
        width = other.width;
        height = other.height;
        channels = other.channels;
        other.pixels = nullptr;
        return *this;
    }
};

/**
 * @brief HDR texture container for EXR environment maps (linear float data)
 */
struct HDRTextureData
{
    std::vector<float> pixels; // RGBA floats, linear space
    int width = 0;
    int height = 0;
    int channels = 0; // 4 = RGBA

    bool valid() const
    {
        return !pixels.empty() && width > 0 && height > 0 && channels == 4;
    }
};

/**
 * @brief Load PNG or JPG texture from file
 * Returns TextureData with pixel data (RGBA format, 4 bytes per pixel)
 */
TextureData loadTexture(const std::string& filePath);

/**
 * @brief Load EXR image as linear float RGBA data (no tone mapping)
 */
HDRTextureData loadEXRFloat(const std::string& filePath);
