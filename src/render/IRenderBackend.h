#pragma once

#include <cstdint>

class Camera;
class Scene;
struct RenderSettings;

class IRenderBackend
{
public:
    virtual ~IRenderBackend() = default;

    virtual void setScene(const Scene* scene) = 0;
    virtual void resize(int width, int height) = 0;
    virtual void resetAccumulation() = 0;
    virtual void render(const Camera& camera, unsigned int glPBO, const RenderSettings& settings) = 0;
    virtual uint32_t accumulatedSamples() const = 0;
};
