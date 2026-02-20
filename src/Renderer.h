#pragma once

#include <cstdint>
#include <memory>

// Forward declarations
class Camera;
class Scene;
class OptixRenderer;

/**
 * @brief High-level renderer facade.
 *
 * Owns an OptixRenderer for GPU path tracing and an OpenGL full-screen quad
 * that blits the result to the screen.  On platforms without an OptiX-capable
 * GPU the OptixRenderer gracefully degrades to a CPU fallback that renders a
 * simple gradient so the window still opens.
 */
class Renderer
{
public:
    Renderer(int width, int height);
    ~Renderer();

    /** Render one frame into the output texture, then blit to the screen. */
    void render(const Camera& camera);

    /** Resize the output buffer. */
    void resize(int width, int height);

    /** Invalidate the accumulation buffer so progressive refinement restarts. */
    void resetAccumulation();

    /* ---- per-frame render settings ---- */
    void setSamplesPerPixel(int spp);
    void setMaxBounces     (int bounces);
    void setExposure       (float exposure);
    void setScene          (const Scene* scene);

    /** Number of accumulated samples in the current frame. */
    uint32_t accumulatedSamples() const;

private:
    void initGL();
    void destroyGL();

    int   m_width;
    int   m_height;
    float m_exposure = 1.f;
    int   m_spp      = 4;
    int   m_maxBounces = 8;

    /* OpenGL objects for full-screen blit */
    unsigned int m_quadVAO      = 0;
    unsigned int m_quadVBO      = 0;
    unsigned int m_displayTex   = 0;
    unsigned int m_displayShader= 0;

    /* CUDA interop pixel buffer object */
    unsigned int m_pbo          = 0;

    std::unique_ptr<OptixRenderer> m_optix;
};
