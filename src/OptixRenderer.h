#pragma once

#include <cstdint>
#include <vector>
#include <memory>

// Forward declarations
class Camera;
class Scene;

/**
 * @brief Wraps the NVIDIA OptiX 7 path-tracing pipeline.
 *
 * When OptiX or CUDA is unavailable, this class falls back to a minimal
 * CPU-based renderer that writes a simple gradient so the application
 * remains usable for UI/workflow development without a GPU.
 *
 * The rendered pixels are written into the OpenGL PBO supplied by @ref render.
 */
class OptixRenderer
{
public:
    OptixRenderer(int width, int height);
    ~OptixRenderer();

    /** Render one sample into @p glPBO. */
    void render(const Camera& camera,
                unsigned int  glPBO,
                int           samplesPerPixel,
                int           maxBounces);

    void resize         (int width, int height);
    void resetAccumulation();
    void setScene       (const Scene* scene);
    void setDebugDirectional(bool enabled) { m_debugDirectional = enabled; }
    void setDebugMode   (int mode) { m_debugMode = mode; }

    uint32_t accumulatedSamples() const { return m_accumulatedSamples; }

private:
    /* ---- OptiX pipeline state (compiled away when OptiX not available) ---- */
    void initOptix   ();
    void buildModule ();
    void buildPipeline();
    void buildSBT    ();
    void buildAccel  ();
    void buildLights (); // 上传光源和环境光到GPU
    void destroyOptix();

    /* ---- CPU fallback ---- */
    void renderCPU(const Camera& camera, unsigned int glPBO,
                   int spp, int maxBounces);

    int  m_width;
    int  m_height;
    bool m_optixReady = false;

    const Scene* m_scene = nullptr;

    uint32_t m_accumulatedSamples = 0;
    bool m_debugDirectional = false;
    int m_debugMode = 0;  // 0=完整追踪, 1=法线, 2=直接光, 3=材质颜色

    /* ---- opaque pimpl for OptiX handles (avoids polluting non-CUDA TUs) ---- */
    struct OptixState;
    std::unique_ptr<OptixState> m_state;
};
