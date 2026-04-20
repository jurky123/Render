#pragma once

#include "IRenderBackend.h"

#include <memory>

class CpuFallbackBackend;
class OptixRenderer;

class OptixBackend final : public IRenderBackend
{
public:
    OptixBackend(int width, int height);
    ~OptixBackend() override;

    void setScene(const Scene* scene) override;
    void resize(int width, int height) override;
    void resetAccumulation() override;
    void render(const Camera& camera, unsigned int glPBO, const RenderSettings& settings) override;
    uint32_t accumulatedSamples() const override;

private:
    std::unique_ptr<OptixRenderer> m_optix;
    std::unique_ptr<CpuFallbackBackend> m_fallback;
};
