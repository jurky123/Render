#include "OptixBackend.h"

#include "CpuFallbackBackend.h"
#include "OptixRenderer.h"
#include "RenderSettings.h"

OptixBackend::OptixBackend(int width, int height)
    : m_optix(std::make_unique<OptixRenderer>(width, height))
    , m_fallback(std::make_unique<CpuFallbackBackend>(width, height))
{
}

OptixBackend::~OptixBackend() = default;

void OptixBackend::setScene(const Scene* scene)
{
    m_optix->setScene(scene);
    m_fallback->setScene(scene);
}

void OptixBackend::resize(int width, int height)
{
    m_optix->resize(width, height);
    m_fallback->resize(width, height);
}

void OptixBackend::resetAccumulation()
{
    m_optix->resetAccumulation();
    m_fallback->resetAccumulation();
}

void OptixBackend::render(const Camera& camera, unsigned int glPBO, const RenderSettings& settings)
{
    m_optix->setDebugDirectional(settings.debugDirectional);
    m_optix->setDebugMode(settings.debugMode);
    m_optix->render(camera, glPBO, settings.samplesPerPixel, settings.maxBounces);
}

uint32_t OptixBackend::accumulatedSamples() const
{
    return m_optix->accumulatedSamples();
}
