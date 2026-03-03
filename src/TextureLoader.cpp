#include "TextureLoader.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// OpenEXR support (native .exr loading)
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <Imath/ImathBox.h>
#include <Imath/ImathVec.h>
#include <Imath/half.h>

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;


// Simple PPM reader for fallback
static TextureData loadPPM(const std::string& filepath)
{
    TextureData result;
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return result;
    
    std::string magic;
    int w, h, maxval;
    file >> magic >> w >> h >> maxval;
    
    if (magic != "P6" || w <= 0 || h <= 0 || maxval != 255)
        return result;
    
    // Skip whitespace
    file.get();
    
    result.width = w;
    result.height = h;
    result.channels = 4;
    result.pixels = new uint8_t[w * h * 4];
    
    uint8_t* rgb = new uint8_t[w * h * 3];
    file.read(reinterpret_cast<char*>(rgb), w * h * 3);
    
    // Convert RGB to RGBA
    for (int i = 0; i < w * h; ++i)
    {
        result.pixels[i * 4 + 0] = rgb[i * 3 + 0];
        result.pixels[i * 4 + 1] = rgb[i * 3 + 1];
        result.pixels[i * 4 + 2] = rgb[i * 3 + 2];
        result.pixels[i * 4 + 3] = 255;
    }
    
    delete[] rgb;
    return result;
}

// Minimal PNG reader using raw byte analysis (very basic, for common PNGs)
static bool isPNG(const uint8_t* data, size_t size)
{
    return size >= 8 && 
           data[0] == 0x89 && data[1] == 0x50 && 
           data[2] == 0x4E && data[3] == 0x47;
}

// Minimal JPG checker
static bool isJPEG(const uint8_t* data, size_t size)
{
    return size >= 2 && data[0] == 0xFF && data[1] == 0xD8;
}

// Create a placeholder texture (1x1 white or colored based on filename)
static TextureData createPlaceholder()
{
    TextureData result;
    result.width = 1;
    result.height = 1;
    result.channels = 4;
    result.pixels = new uint8_t[4];
    // Light gray placeholder
    result.pixels[0] = 200;
    result.pixels[1] = 200;
    result.pixels[2] = 200;
    result.pixels[3] = 255;
    return result;
}

/**
 * @brief Load HDR image using stb_image (Radiance .hdr format)
 * Note: For .exr files, pre-convert using: magick input.exr output.hdr
 */
static TextureData loadHDR(const std::string& filepath)
{
    TextureData result;
    
    int width, height, channels;
    // Request 3 channels (RGB), stb_image will convert
    float* hdr_data = stbi_loadf(filepath.c_str(), &width, &height, &channels, 3);
    
    if (!hdr_data) {
        std::cerr << "[TextureLoader] Failed to load HDR: " << filepath << "\n";
        std::cerr << "[TextureLoader] stb_image error: " << stbi_failure_reason() << "\n";
        return createPlaceholder();
    }
    
    std::cout << "[TextureLoader] Loading HDR: " << filepath 
              << " (" << width << "x" << height << ")\n";
    
    // Convert float HDR to 8-bit sRGB
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.pixels = new uint8_t[width * height * 4];
    
    // Tone map and convert to sRGB
    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < 3; ++c) {
            float val = hdr_data[i * 3 + c];
            
            // Apply tone mapping: Reinhard operator
            val = val / (1.0f + val);
            
            // Clamp to [0,1]
            val = std::max(0.0f, std::min(1.0f, val));
            
            // sRGB gamma correction
            if (val <= 0.0031308f)
                val = val * 12.92f;
            else
                val = 1.055f * std::pow(val, 1.0f / 2.4f) - 0.055f;
            
            result.pixels[i * 4 + c] = static_cast<uint8_t>(val * 255.0f + 0.5f);
        }
        // Set alpha to fully opaque
        result.pixels[i * 4 + 3] = 255;
    }
    
    // Free stb_image memory
    stbi_image_free(hdr_data);
    
    std::cout << "[TextureLoader] Successfully loaded HDR with tone mapping: " 
              << result.width << "x" << result.height << "\n";
    
    return result;
}

/**
 * @brief Load EXR image as linear float RGBA data (no tone mapping)
 */
HDRTextureData loadEXRFloat(const std::string& filepath)
{
    HDRTextureData result;
    try {
        InputFile file(filepath.c_str());
        Box2i dw = file.header().dataWindow();

        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        std::cout << "[TextureLoader] Loading EXR (float): " << filepath
                  << " (" << width << "x" << height << ")\n";

        const ChannelList& channels = file.header().channels();
        std::vector<std::string> channelNames;
        if (channels.findChannel("R") && channels.findChannel("G") && channels.findChannel("B")) {
            channelNames = {"R", "G", "B"};
        } else {
            for (auto iter = channels.begin(); iter != channels.end() && channelNames.size() < 3; ++iter) {
                channelNames.push_back(iter.name());
            }
        }

        if (channelNames.size() < 3) {
            std::cerr << "[TextureLoader] EXR file has fewer than 3 channels: " << filepath << "\n";
            return result;
        }

        result.width = width;
        result.height = height;
        result.channels = 4;
        result.pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

        FrameBuffer fb;
        fb.insert(channelNames[0].c_str(), Slice(FLOAT,
                             (char*)(&result.pixels[0]),
                             sizeof(float) * 4,
                             sizeof(float) * 4 * width,
                             1, 1, 0.0f));
        fb.insert(channelNames[1].c_str(), Slice(FLOAT,
                             (char*)(&result.pixels[1]),
                             sizeof(float) * 4,
                             sizeof(float) * 4 * width,
                             1, 1, 0.0f));
        fb.insert(channelNames[2].c_str(), Slice(FLOAT,
                             (char*)(&result.pixels[2]),
                             sizeof(float) * 4,
                             sizeof(float) * 4 * width,
                             1, 1, 0.0f));

        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);

        // Set alpha channel to 1.0 after reading pixel data
        for (size_t i = 0; i < result.pixels.size(); i += 4) {
            result.pixels[i + 3] = 1.0f;
        }

        std::cout << "[TextureLoader] Successfully loaded EXR (float): "
                  << result.width << "x" << result.height << "\n";
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[TextureLoader] Failed to load EXR (float): " << filepath << "\n";
        std::cerr << "[TextureLoader] OpenEXR error: " << e.what() << "\n";
        return result;
    }
}

/**
 * @brief Load OpenEXR image using the OpenEXR library
 * Converts float HDR data to 8-bit sRGB for consistency
 */
static TextureData loadEXR(const std::string& filepath)
{
    TextureData result;
    try {
        InputFile file(filepath.c_str());
        Box2i dw = file.header().dataWindow();

        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        std::cout << "[TextureLoader] Loading EXR: " << filepath
                  << " (" << width << "x" << height << ")\n";

        std::vector<float> floatPixels(width * height * 4, 0.0f);

        const ChannelList& channels = file.header().channels();
        std::vector<std::string> channelNames;
        if (channels.findChannel("R") && channels.findChannel("G") && channels.findChannel("B")) {
            channelNames = {"R", "G", "B"};
        } else {
            for (auto iter = channels.begin(); iter != channels.end() && channelNames.size() < 3; ++iter) {
                channelNames.push_back(iter.name());
            }
        }

        if (channelNames.size() < 3) {
            std::cerr << "[TextureLoader] EXR file has fewer than 3 channels: " << filepath << "\n";
            return createPlaceholder();
        }

        FrameBuffer fb;
        fb.insert(channelNames[0].c_str(), Slice(FLOAT,
                             (char*)(&floatPixels[0]),
                             sizeof(float) * 4,
                             sizeof(float) * 4 * width,
                             1, 1, 0.0f));
        fb.insert(channelNames[1].c_str(), Slice(FLOAT,
                             (char*)(&floatPixels[1]),
                             sizeof(float) * 4,
                             sizeof(float) * 4 * width,
                             1, 1, 0.0f));
        fb.insert(channelNames[2].c_str(), Slice(FLOAT,
                             (char*)(&floatPixels[2]),
                             sizeof(float) * 4,
                             sizeof(float) * 4 * width,
                             1, 1, 0.0f));

        for (size_t i = 0; i < floatPixels.size(); i += 4) {
            floatPixels[i + 3] = 1.0f;
        }

        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);

        result.width = width;
        result.height = height;
        result.channels = 4;
        result.pixels = new uint8_t[width * height * 4];

        for (int i = 0; i < width * height; ++i) {
            for (int c = 0; c < 4; ++c) {
                float val = floatPixels[i * 4 + c];
                val = val / (1.0f + val);
                val = std::max(0.0f, std::min(1.0f, val));
                if (val <= 0.0031308f)
                    val = val * 12.92f;
                else
                    val = 1.055f * std::pow(val, 1.0f / 2.4f) - 0.055f;
                result.pixels[i * 4 + c] = static_cast<uint8_t>(val * 255.0f + 0.5f);
            }
        }

        std::cout << "[TextureLoader] Successfully loaded EXR with tone mapping: "
                  << result.width << "x" << result.height << "\n";
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[TextureLoader] Failed to load EXR file: " << filepath << "\n";
        std::cerr << "[TextureLoader] OpenEXR error: " << e.what() << "\n";
        return createPlaceholder();
    }
}

/**
 * @brief Load image from file (supports PNG, JPG, HDR, and EXR)
 */
TextureData loadTexture(const std::string& filePath)
{
    std::cout << "[TextureLoader] Attempting to load: " << filePath << "\n";
    
    // Check for HDR extension (Radiance format supported by stb_image)
    std::string lower_path = filePath;
    for (auto &c : lower_path) c = std::tolower(c);
    
    // Check file extension
    if (lower_path.size() >= 4) {
        std::string ext = lower_path.substr(lower_path.size() - 4);
        
        if (ext == ".hdr") {
            return loadHDR(filePath);
        } else if (ext == ".exr") {
            return loadEXR(filePath);
        }
    }
    
    TextureData result;
    
    // Load as RGBA (4 = RGBA format) via stb_image
    int width, height, channels;
    unsigned char* pixels = stbi_load(filePath.c_str(), &width, &height, &channels, 4);
    
    if (!pixels)
    {
        std::cerr << "[TextureLoader] Failed to load image: " << filePath << "\n";
        std::cerr << "[TextureLoader] stb_image error: " << stbi_failure_reason() << "\n";
        return createPlaceholder();
    }
    
    result.width = width;
    result.height = height;
    result.channels = 4; // stb_image loaded as RGBA
    result.pixels = new uint8_t[width * height * 4];
    
    // Copy pixel data
    std::memcpy(result.pixels, pixels, width * height * 4);
    
    // Free stb_image memory
    stbi_image_free(pixels);
    
    std::cout << "[TextureLoader] Successfully loaded: " << result.width << "x" << result.height 
              << " (" << result.channels << " channels)\n";
    
    return result;
}
