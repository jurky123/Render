#pragma once

#include "IRenderBackend.h"

class CpuFallbackBackend final : public IRenderBackend
{
public:
    CpuFallbackBackend(int width, int height);
    ~CpuFallbackBackend() override = default;

    void setScene(const Scene* scene) override;
    void resize(int width, int height) override;
    void resetAccumulation() override;
    void render(const Camera& camera, unsigned int glPBO, const RenderSettings& settings) override;
    uint32_t accumulatedSamples() const override;

private:
    int m_width;
    int m_height;
    const Scene* m_scene = nullptr;
    uint32_t m_accumulatedSamples = 0;
};
