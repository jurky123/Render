#include "CpuFallbackBackend.h"

#include "Camera.h"
#include "RenderSettings.h"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstring>
#include <random>
#include <vector>

CpuFallbackBackend::CpuFallbackBackend(int width, int height)
    : m_width(width), m_height(height)
{
}

void CpuFallbackBackend::setScene(const Scene* scene)
{
    m_scene = scene;
    resetAccumulation();
}

void CpuFallbackBackend::resize(int width, int height)
{
    m_width = width;
    m_height = height;
    resetAccumulation();
}

void CpuFallbackBackend::resetAccumulation()
{
    m_accumulatedSamples = 0;
}

uint32_t CpuFallbackBackend::accumulatedSamples() const
{
    return m_accumulatedSamples;
}

void CpuFallbackBackend::render(const Camera& camera, unsigned int glPBO, const RenderSettings& settings)
{
    const float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    const glm::vec3 llc = camera.lowerLeftCorner(aspect);
    const glm::vec3 hor = camera.horizontal(aspect);
    const glm::vec3 ver = camera.vertical();
    const glm::vec3 ori = camera.position();

    std::vector<float> pixels(static_cast<size_t>(m_width * m_height) * 4);
    std::mt19937 rng(m_accumulatedSamples);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            glm::vec3 color(0.f);
            for (int s = 0; s < settings.samplesPerPixel; ++s)
            {
                const float u = (x + dist(rng)) / static_cast<float>(m_width);
                const float v = (y + dist(rng)) / static_cast<float>(m_height);
                const glm::vec3 dir = glm::normalize(llc + u * hor + v * ver - ori);
                const float t = 0.5f * (dir.y + 1.f);
                color += glm::mix(glm::vec3(1.f), glm::vec3(0.5f, 0.7f, 1.f), t);
            }

            color /= static_cast<float>(settings.samplesPerPixel);
            const size_t idx = static_cast<size_t>((y * m_width + x)) * 4;
            pixels[idx + 0] = color.r;
            pixels[idx + 1] = color.g;
            pixels[idx + 2] = color.b;
            pixels[idx + 3] = 1.f;
        }
    }

    m_accumulatedSamples += static_cast<uint32_t>(settings.samplesPerPixel);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);
    void* ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (ptr)
    {
        std::memcpy(ptr, pixels.data(), pixels.size() * sizeof(float));
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    (void)m_scene;
}
