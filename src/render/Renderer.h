#pragma once

#include <cstdint>
#include <memory>

class Camera;
class Scene;
class IRenderBackend;
struct RenderSettings;

class Renderer
{
public:
    Renderer(int width, int height);
    ~Renderer();

    void setScene(const Scene* scene);
    void resize(int width, int height);
    void resetAccumulation();
    void render(const Camera& camera, const RenderSettings& settings);
    uint32_t accumulatedSamples() const;

private:
    void initGL();
    void destroyGL();

    int m_width;
    int m_height;
    unsigned int m_quadVAO = 0;
    unsigned int m_displayTex = 0;
    unsigned int m_displayShader = 0;
    unsigned int m_pbo = 0;
    std::unique_ptr<IRenderBackend> m_backend;
};
