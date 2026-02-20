#pragma once

// Material.h is intentionally thin – the Material struct lives in Scene.h
// so that Assimp conversion code and GPU data structures all share the same
// definition.  This header simply re-exports that struct and provides a few
// standalone helpers.

#include "Scene.h" // Material struct defined here

#include <cmath>
#include <glm/glm.hpp>

/**
 * @brief Evaluate the Schlick Fresnel approximation on the CPU.
 *
 * Used in software fall-back rendering and unit tests.
 *
 * @param cosTheta  cosine of the angle between the view direction and the
 *                  half-vector (or surface normal for the base case).
 * @param f0        reflectance at normal incidence.
 */
inline glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& f0)
{
    const float t = 1.f - cosTheta;
    const float t2 = t * t;
    const float t5 = t2 * t2 * t;
    return f0 + (glm::vec3(1.f) - f0) * t5;
}

/**
 * @brief Convert a linear colour to sRGB.
 */
inline glm::vec3 linearToSRGB(const glm::vec3& c)
{
    auto ch = [](float x) {
        return (x <= 0.0031308f)
               ? 12.92f * x
               : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
    };
    return {ch(c.r), ch(c.g), ch(c.b)};
}
