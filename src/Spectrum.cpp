#include "Spectrum.h"
#include <glm/gtc/constants.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

namespace pbrt {

// RGBColorSpace static member
const RGBColorSpace* RGBColorSpace::sRGB = nullptr;

RGBColorSpace::RGBColorSpace() {
    // Default to identity (simplified)
    XYZFromRGB = glm::mat3(1.0f);
    RGBFromXYZ = glm::mat3(1.0f);
}

RGB RGBColorSpace::ToRGB(const XYZ& xyz) const {
    // Simplified conversion (should use RGBFromXYZ matrix)
    return RGB(xyz.X, xyz.Y, xyz.Z);
}

XYZ RGBColorSpace::ToXYZ(const RGB& rgb) const {
    // Simplified conversion
    return XYZ(rgb.r, rgb.g, rgb.b);
}

// SampledSpectrum conversions
RGB SampledSpectrum::ToRGB(const SampledWavelengths& lambda, const RGBColorSpace& cs) const {
    // Simplified: return average as grayscale
    float avg = Average();
    return RGB(avg, avg, avg);
}

XYZ SampledSpectrum::ToXYZ(const SampledWavelengths& lambda) const {
    float y = Average();
    return XYZ(y, y, y);
}

// Load spectrum from SPD file (wavelength, value pairs)
Spectrum LoadSpectrumFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open spectrum file: " << filename << std::endl;
        return Spectrum::CreateConstant(0);
    }
    
    std::vector<float> lambdas, values;
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        float lambda, value;
        if (iss >> lambda >> value) {
            lambdas.push_back(lambda);
            values.push_back(value);
        }
    }
    
    if (lambdas.empty()) {
        std::cerr << "No data in spectrum file: " << filename << std::endl;
        return Spectrum::CreateConstant(0);
    }
    
    return Spectrum::CreatePiecewiseLinear(lambdas, values);
}

} // namespace pbrt
