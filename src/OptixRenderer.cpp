#include "OptixRenderer.h"
#include "Camera.h"
#include "Scene.h"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <cstdlib>

// ---------------------------------------------------------------------------
// OptiX/CUDA includes – conditionally compiled
// ---------------------------------------------------------------------------
#ifdef PATHTRACER_OPTIX_ENABLED
#  include <cuda_runtime.h>
#  include <cuda_gl_interop.h>
#  include <optix.h>
#  include <optix_host.h>
#  include <optix_function_table_definition.h>
#  include <optix_stubs.h>

static void logOptixLine(const std::string& message)
{
    static std::string logPath = []() {
        const char* envPath = std::getenv("PATHTRACER_LOG");
        return envPath ? std::string(envPath) : std::string("PathTracer_optix.log");
    }();
    static std::ofstream logFile(logPath, std::ios::app);
    if (logFile)
    {
        logFile << message << "\n";
        logFile.flush();
    }
    std::cerr << message << "\n";
}

static void optixLogCallback(unsigned int level, const char* tag,
                             const char* message, void*)
{
    if (level <= 3)
    {
        std::ostringstream oss;
        oss << "[OptiX][" << tag << "] " << message;
        logOptixLine(oss.str());
    }
}

#  define CUDA_CHECK(call)                                                  \
    do {                                                                    \
        cudaError_t rc = (call);                                            \
        if (rc != cudaSuccess) {                                            \
            std::ostringstream oss;                                         \
            oss << "[CUDA] " << cudaGetErrorString(rc)                     \
                << " at " << __FILE__ << ":" << __LINE__;                  \
            logOptixLine(oss.str());                                        \
            m_optixReady = false;                                           \
        }                                                                   \
    } while(0)

#  define OPTIX_CHECK(call)                                                 \
    do {                                                                    \
        OptixResult rc = (call);                                            \
        if (rc != OPTIX_SUCCESS) {                                          \
            std::ostringstream oss;                                         \
            oss << "[OptiX] " << optixGetErrorString(rc)                   \
                << " at " << __FILE__ << ":" << __LINE__;                  \
            logOptixLine(oss.str());                                        \
            m_optixReady = false;                                           \
        }                                                                   \
    } while(0)
#endif // PATHTRACER_OPTIX_ENABLED

// ---------------------------------------------------------------------------
// Shared launch parameters struct (must match shaders/path_tracer.cu)
// ---------------------------------------------------------------------------
#include "launch_params.h"

// ---------------------------------------------------------------------------
// OptixState – hides all OptiX handles from non-CUDA translation units
// ---------------------------------------------------------------------------
struct OptixRenderer::OptixState
{
#ifdef PATHTRACER_OPTIX_ENABLED
    OptixDeviceContext    context        = nullptr;
    OptixPipeline         pipeline       = nullptr;
    OptixModule           module         = nullptr;
    OptixProgramGroup     raygenPG       = nullptr;
    OptixProgramGroup     missPG         = nullptr;
    OptixProgramGroup     hitgroupPG     = nullptr;
    OptixProgramGroup     hitgroupCurvePG = nullptr;
    OptixShaderBindingTable sbt          = {};
    OptixTraversableHandle gasHandle     = 0;

    CUstream              stream         = nullptr;
    CUdeviceptr           d_sbt          = 0;
    CUdeviceptr           d_gas          = 0;
    CUdeviceptr           d_launchParams = 0;
    CUdeviceptr           d_accum        = 0;
    cudaGraphicsResource* pboResource    = nullptr;

    // Denoiser + AOV buffers
    CUdeviceptr           d_beauty       = 0;
    CUdeviceptr           d_albedo       = 0;
    CUdeviceptr           d_normal       = 0;
    CUdeviceptr           d_denoised     = 0;
    CUdeviceptr           d_intensity    = 0;
    OptixDenoiser         denoiser       = nullptr;
    CUdeviceptr           d_denoiserState   = 0;
    CUdeviceptr           d_denoiserScratch = 0;
    size_t                denoiserStateSize = 0;
    size_t                denoiserScratchSize = 0;
    bool                  denoiserReady = false;

    // Persistent per-mesh geometry buffers for closesthit access
    std::vector<CUdeviceptr> d_meshVerts;
    std::vector<CUdeviceptr> d_meshNormals;
    std::vector<CUdeviceptr> d_meshIndices;
    std::vector<CUdeviceptr> d_meshTexCoords;    // texture coordinates per mesh

    // Per-material texture buffers
    std::vector<CUdeviceptr> d_baseColorTexPixels;  // RGBA texture pixels per material
    std::vector<int>         baseColorTexWidths;    // texture dimensions per material
    std::vector<int>         baseColorTexHeights;

    std::vector<CUdeviceptr> d_normalTexPixels;
    std::vector<int>         normalTexWidths;
    std::vector<int>         normalTexHeights;

    std::vector<CUdeviceptr> d_metallicRoughnessTexPixels;
    std::vector<int>         metallicRoughnessTexWidths;
    std::vector<int>         metallicRoughnessTexHeights;

    std::vector<CUdeviceptr> d_emissionTexPixels;
    std::vector<int>         emissionTexWidths;
    std::vector<int>         emissionTexHeights;

    std::vector<CUdeviceptr> d_transmissionTexPixels;
    std::vector<int>         transmissionTexWidths;
    std::vector<int>         transmissionTexHeights;

    CUdeviceptr           d_envmapPixels = 0;
    CUdeviceptr           d_envmapPixelsF = 0;
    int                   envmapWidth = 0;
    int                   envmapHeight = 0;

    CUdeviceptr           d_curveData = 0;
    CUdeviceptr           d_curveAabbs = 0;
    CUdeviceptr           d_curveSbtIndices = 0;
    int                   curveCount = 0;
#endif
    LaunchParams launchParams = {};
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

OptixRenderer::OptixRenderer(int width, int height)
    : m_width(width), m_height(height)
    , m_state(std::make_unique<OptixState>())
{
    initOptix();
}

OptixRenderer::~OptixRenderer()
{
    destroyOptix();
}

// ---------------------------------------------------------------------------
// initOptix
// ---------------------------------------------------------------------------

void OptixRenderer::initOptix()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    // Initialise CUDA
    CUDA_CHECK(cudaFree(nullptr)); // forces CUDA context creation

    // Initialise OptiX function table
    OptixResult res = optixInit();
    if (res != OPTIX_SUCCESS)
    {
        {
            std::ostringstream oss;
            oss << "[OptiX] optixInit failed: " << optixGetErrorString(res)
                << " – falling back to CPU renderer.";
            logOptixLine(oss.str());
        }
        return;
    }

    // Create OptiX device context
    CUcontext cuCtx = nullptr; // use current context
    OptixDeviceContextOptions opts = {};
    opts.logCallbackFunction = optixLogCallback;
    opts.logCallbackLevel = 4;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &opts, &m_state->context));
    if (!m_state->context)
    {
        logOptixLine("[OptiX] Failed to create device context");
        return;
    }

    CUDA_CHECK(cudaStreamCreate(&m_state->stream));

    // Allocate accumulation buffer
    const size_t bufSize = static_cast<size_t>(m_width * m_height) * 4 * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_accum), bufSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_beauty), bufSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_albedo), bufSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_normal), bufSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_denoised), bufSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_intensity), sizeof(float)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_accum), 0, bufSize));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_beauty), 0, bufSize));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_albedo), 0, bufSize));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_normal), 0, bufSize));

    // Create OptiX denoiser (HDR + albedo/normal guides)
    OptixDenoiserOptions denoiseOpts = {};
    denoiseOpts.guideAlbedo = 1;
    denoiseOpts.guideNormal = 1;

    if (OPTIX_SUCCESS == optixDenoiserCreate(m_state->context,
                                             OPTIX_DENOISER_MODEL_KIND_HDR,
                                             &denoiseOpts,
                                             &m_state->denoiser))
    {
        OptixDenoiserSizes denoiseSizes = {};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            m_state->denoiser, m_width, m_height, &denoiseSizes));

        m_state->denoiserStateSize = denoiseSizes.stateSizeInBytes;
        m_state->denoiserScratchSize = denoiseSizes.withoutOverlapScratchSizeInBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_denoiserState),
                              m_state->denoiserStateSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_denoiserScratch),
                              m_state->denoiserScratchSize));

        OPTIX_CHECK(optixDenoiserSetup(
            m_state->denoiser, m_state->stream,
            m_width, m_height,
            m_state->d_denoiserState, m_state->denoiserStateSize,
            m_state->d_denoiserScratch, m_state->denoiserScratchSize));

        m_state->denoiserReady = true;
    }
    else
    {
        logOptixLine("[OptiX] Failed to create denoiser. Denoising disabled.");
        m_state->denoiserReady = false;
    }

    buildPipeline();
    
    // After pipeline succeeds, mark as ready
    if (m_state->sbt.raygenRecord != 0)
    {
        m_optixReady = true;
        logOptixLine("[OptiX] Initialised successfully.");
    }
    else
    {
        logOptixLine("[OptiX] Failed to initialize: SBT not ready");
    }
#else
    logOptixLine("[OptiX] Not compiled – using CPU fallback renderer.");
#endif
}

void OptixRenderer::destroyOptix()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (!m_optixReady) return;

    if (m_state->pboResource)
        cudaGraphicsUnregisterResource(m_state->pboResource);
    if (m_state->d_accum)
        cudaFree(reinterpret_cast<void*>(m_state->d_accum));
    if (m_state->d_beauty)
        cudaFree(reinterpret_cast<void*>(m_state->d_beauty));
    if (m_state->d_albedo)
        cudaFree(reinterpret_cast<void*>(m_state->d_albedo));
    if (m_state->d_normal)
        cudaFree(reinterpret_cast<void*>(m_state->d_normal));
    if (m_state->d_denoised)
        cudaFree(reinterpret_cast<void*>(m_state->d_denoised));
    if (m_state->d_intensity)
        cudaFree(reinterpret_cast<void*>(m_state->d_intensity));
    if (m_state->d_denoiserState)
        cudaFree(reinterpret_cast<void*>(m_state->d_denoiserState));
    if (m_state->d_denoiserScratch)
        cudaFree(reinterpret_cast<void*>(m_state->d_denoiserScratch));
    if (m_state->denoiser)
        optixDenoiserDestroy(m_state->denoiser);
    if (m_state->d_launchParams)
        cudaFree(reinterpret_cast<void*>(m_state->d_launchParams));
    if (m_state->d_gas)
        cudaFree(reinterpret_cast<void*>(m_state->d_gas));
    if (m_state->d_sbt)
        cudaFree(reinterpret_cast<void*>(m_state->d_sbt));
    if (m_state->d_envmapPixels)
        cudaFree(reinterpret_cast<void*>(m_state->d_envmapPixels));
    if (m_state->d_curveData)
        cudaFree(reinterpret_cast<void*>(m_state->d_curveData));
    if (m_state->d_curveAabbs)
        cudaFree(reinterpret_cast<void*>(m_state->d_curveAabbs));
    if (m_state->d_curveSbtIndices)
        cudaFree(reinterpret_cast<void*>(m_state->d_curveSbtIndices));
    if (m_state->hitgroupPG)
        optixProgramGroupDestroy(m_state->hitgroupPG);
    if (m_state->hitgroupCurvePG)
        optixProgramGroupDestroy(m_state->hitgroupCurvePG);
    if (m_state->missPG)
        optixProgramGroupDestroy(m_state->missPG);
    if (m_state->raygenPG)
        optixProgramGroupDestroy(m_state->raygenPG);
    if (m_state->pipeline)
        optixPipelineDestroy(m_state->pipeline);
    if (m_state->module)
        optixModuleDestroy(m_state->module);
    if (m_state->context)
        optixDeviceContextDestroy(m_state->context);
    if (m_state->stream)
        cudaStreamDestroy(m_state->stream);
#endif
}

// ---------------------------------------------------------------------------
// buildModule – loads PTX and creates OptiX module
// ---------------------------------------------------------------------------

void OptixRenderer::buildModule()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (!m_state->context) return;

    OptixModuleCompileOptions modOpts = {};
    modOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    modOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    modOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur            = 0;
    pipeOpts.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues          = 14; // expanded to carry t, normal.xyz, baseColor.xyz, metallic, roughness, emission.xyz, transmission, ior
    pipeOpts.numAttributeValues        = 2;
    pipeOpts.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "params";

    // Load PTX file
#ifndef PTX_DIR
#  define PTX_DIR "."
#endif
    const std::string ptxPath = std::string(PTX_DIR) + "/path_tracer.ptx";
    FILE* f = fopen(ptxPath.c_str(), "rb");
    if (!f)
    {
        {
            std::ostringstream oss;
            oss << "[OptiX] Cannot open PTX: " << ptxPath;
            logOptixLine(oss.str());
        }
        m_optixReady = false;
        return;
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> ptx(static_cast<size_t>(sz) + 1, '\0');
    fread(ptx.data(), 1, static_cast<size_t>(sz), f);
    fclose(f);

    char log[2048]; size_t logSz = sizeof(log);
    log[0] = '\0';
    OptixResult modRes = optixModuleCreate(
        m_state->context,
        &modOpts,
        &pipeOpts,
        ptx.data(),
        ptx.size(),
        log,
        &logSz,
        &m_state->module);
    if (logSz > 1)
    {
        std::ostringstream oss;
        oss << "[OptiX] Module log: " << log;
        logOptixLine(oss.str());
    }
    if (modRes != OPTIX_SUCCESS)
    {
        {
            std::ostringstream oss;
            oss << "[OptiX] Module creation failed: " << optixGetErrorString(modRes);
            logOptixLine(oss.str());
        }
        {
            std::ostringstream oss;
            oss << "[OptiX] PTX path: " << ptxPath << " (" << ptx.size() << " bytes)";
            logOptixLine(oss.str());
        }
        m_state->module = nullptr;
        m_optixReady = false;
        return;
    }
#endif
}
// ---------------------------------------------------------------------------
// buildPipeline – creates ray gen, miss, and hit group programs
// ---------------------------------------------------------------------------

void OptixRenderer::buildPipeline()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    buildModule();
    if (!m_state->module) return;

    // Apply same pipeline compile options as buildModule()
    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur            = 0;
    pipeOpts.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues          = 14; // expanded to carry t, normal.xyz, baseColor.xyz, metallic, roughness, emission.xyz, transmission, ior
    pipeOpts.numAttributeValues        = 2;
    pipeOpts.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "params";

    char log[2048]; size_t logSz = sizeof(log);
    OptixProgramGroupOptions pgOpts = {};
    
    // Raygen program
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = m_state->module;
        desc.raygen.entryFunctionName = "__raygen__pathTrace";
        
        logSz = sizeof(log);
        log[0] = '\0';
        OptixResult pgRes = optixProgramGroupCreate(
            m_state->context, &desc, 1, &pgOpts,
            log, &logSz,
            &m_state->raygenPG);
        if (logSz > 1)
            std::cerr << "[OptiX] Raygen log: " << log << "\n";
        if (pgRes != OPTIX_SUCCESS)
        {
            std::cerr << "[OptiX] Raygen program group failed: " << optixGetErrorString(pgRes) << "\n";
            m_state->raygenPG = nullptr;
            return;
        }
    }

    // Miss program
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = m_state->module;
        desc.miss.entryFunctionName = "__miss__envmap";
        
        logSz = sizeof(log);
        log[0] = '\0';
        OptixResult pgRes = optixProgramGroupCreate(
            m_state->context, &desc, 1, &pgOpts,
            log, &logSz,
            &m_state->missPG);
        if (logSz > 1)
            std::cerr << "[OptiX] Miss log: " << log << "\n";
        if (pgRes != OPTIX_SUCCESS)
        {
            std::cerr << "[OptiX] Miss program group failed: " << optixGetErrorString(pgRes) << "\n";
            m_state->missPG = nullptr;
            return;
        }
    }

    // Hitgroup program (triangles)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = m_state->module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__pbr";
        
        logSz = sizeof(log);
        log[0] = '\0';
        OptixResult pgRes = optixProgramGroupCreate(
            m_state->context, &desc, 1, &pgOpts,
            log, &logSz,
            &m_state->hitgroupPG);
        if (logSz > 1)
            std::cerr << "[OptiX] Hitgroup log: " << log << "\n";
        if (pgRes != OPTIX_SUCCESS)
        {
            std::cerr << "[OptiX] Hitgroup program group failed: " << optixGetErrorString(pgRes) << "\n";
            m_state->hitgroupPG = nullptr;
            return;
        }
    }

    // Hitgroup program (curves with custom intersection)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = m_state->module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__pbr";
        desc.hitgroup.moduleIS = m_state->module;
        desc.hitgroup.entryFunctionNameIS = "__intersection__curve";

        logSz = sizeof(log);
        log[0] = '\0';
        OptixResult pgRes = optixProgramGroupCreate(
            m_state->context, &desc, 1, &pgOpts,
            log, &logSz,
            &m_state->hitgroupCurvePG);
        if (pgRes != OPTIX_SUCCESS)
        {
            std::cerr << "[OptiX] Failed to create curve hitgroup: " << optixGetErrorString(pgRes) << "\n";
            if (logSz > 1)
                std::cerr << "[OptiX] Curve hitgroup log: " << log << "\n";
            m_state->hitgroupCurvePG = nullptr;
        }
    }

    if (!m_state->raygenPG || !m_state->missPG || !m_state->hitgroupPG)
        return;

    // Link pipeline
    OptixProgramGroup pgsAll[] = { m_state->raygenPG, m_state->missPG, m_state->hitgroupPG, m_state->hitgroupCurvePG };
    OptixProgramGroup pgsTri[] = { m_state->raygenPG, m_state->missPG, m_state->hitgroupPG };
    
    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 8;

    if (m_state->hitgroupCurvePG)
    {
        logSz = sizeof(log);
        log[0] = '\0';
        OptixResult pipeRes = optixPipelineCreate(
            m_state->context,
            &pipeOpts,
            &linkOpts,
            pgsAll, 4,
            log, &logSz,
            &m_state->pipeline);
        if (logSz > 1)
        {
            std::ostringstream oss;
            oss << "[OptiX] Pipeline log: " << log;
            logOptixLine(oss.str());
        }
        if (pipeRes != OPTIX_SUCCESS)
        {
            {
                std::ostringstream oss;
                oss << "[OptiX] Pipeline creation failed: " << optixGetErrorString(pipeRes);
                logOptixLine(oss.str());
            }
            m_state->pipeline = nullptr;
            return;
        }
    }
    else
    {
        logSz = sizeof(log);
        log[0] = '\0';
        OptixResult pipeRes = optixPipelineCreate(
            m_state->context,
            &pipeOpts,
            &linkOpts,
            pgsTri, 3,
            log, &logSz,
            &m_state->pipeline);
        if (logSz > 1)
            std::cerr << "[OptiX] Pipeline log: " << log << "\n";
        if (pipeRes != OPTIX_SUCCESS)
        {
            std::cerr << "[OptiX] Pipeline creation failed: " << optixGetErrorString(pipeRes) << "\n";
            m_state->pipeline = nullptr;
            return;
        }
    }

    // Set stack size
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_state->pipeline, 2048, 2048, 2048, 1));

    buildSBT();
#endif
}

void OptixRenderer::buildSBT()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (!m_state->module || !m_state->raygenPG || !m_state->missPG || !m_state->hitgroupPG)
    {
        logOptixLine("[OptiX] Cannot build SBT: program groups not created");
        return;
    }

    // Upload texture data if materials have textures
    if (m_scene && !m_scene->empty())
    {
        const auto& materials = m_scene->materials();
        
        // Initialize all texture vectors
        m_state->d_baseColorTexPixels.resize(materials.size(), 0);
        m_state->baseColorTexWidths.resize(materials.size(), 0);
        m_state->baseColorTexHeights.resize(materials.size(), 0);

        m_state->d_normalTexPixels.resize(materials.size(), 0);
        m_state->normalTexWidths.resize(materials.size(), 0);
        m_state->normalTexHeights.resize(materials.size(), 0);

        m_state->d_metallicRoughnessTexPixels.resize(materials.size(), 0);
        m_state->metallicRoughnessTexWidths.resize(materials.size(), 0);
        m_state->metallicRoughnessTexHeights.resize(materials.size(), 0);

        m_state->d_emissionTexPixels.resize(materials.size(), 0);
        m_state->emissionTexWidths.resize(materials.size(), 0);
        m_state->emissionTexHeights.resize(materials.size(), 0);

        m_state->d_transmissionTexPixels.resize(materials.size(), 0);
        m_state->transmissionTexWidths.resize(materials.size(), 0);
        m_state->transmissionTexHeights.resize(materials.size(), 0);

        for (size_t i = 0; i < materials.size(); ++i)
        {
            const auto& mat = materials[i];
            
            // Upload baseColor texture
            if (mat.baseColorTexData.pixels && mat.baseColorTexData.width > 0 && mat.baseColorTexData.height > 0)
            {
                size_t texSize = static_cast<size_t>(mat.baseColorTexData.width) * 
                                static_cast<size_t>(mat.baseColorTexData.height) * 
                                static_cast<size_t>(mat.baseColorTexData.channels);
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Uploading baseColor texture for material " << i
                        << " (" << mat.baseColorTexData.width << "x" << mat.baseColorTexData.height
                        << ", " << texSize << " bytes)";
                    logOptixLine(oss.str());
                }
                
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_baseColorTexPixels[i]), texSize));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_baseColorTexPixels[i]),
                                      mat.baseColorTexData.pixels, texSize, cudaMemcpyHostToDevice));
                
                m_state->baseColorTexWidths[i] = mat.baseColorTexData.width;
                m_state->baseColorTexHeights[i] = mat.baseColorTexData.height;
            }

            // Upload normal texture
            if (mat.normalTexData.pixels && mat.normalTexData.width > 0 && mat.normalTexData.height > 0)
            {
                size_t texSize = static_cast<size_t>(mat.normalTexData.width) * 
                                static_cast<size_t>(mat.normalTexData.height) * 
                                static_cast<size_t>(mat.normalTexData.channels);
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Uploading normal texture for material " << i
                        << " (" << mat.normalTexData.width << "x" << mat.normalTexData.height
                        << ", " << texSize << " bytes)";
                    logOptixLine(oss.str());
                }
                
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_normalTexPixels[i]), texSize));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_normalTexPixels[i]),
                                      mat.normalTexData.pixels, texSize, cudaMemcpyHostToDevice));
                
                m_state->normalTexWidths[i] = mat.normalTexData.width;
                m_state->normalTexHeights[i] = mat.normalTexData.height;
            }

            // Upload metallicRoughness texture
            if (mat.metallicRoughnessTexData.pixels && mat.metallicRoughnessTexData.width > 0 && mat.metallicRoughnessTexData.height > 0)
            {
                size_t texSize = static_cast<size_t>(mat.metallicRoughnessTexData.width) * 
                                static_cast<size_t>(mat.metallicRoughnessTexData.height) * 
                                static_cast<size_t>(mat.metallicRoughnessTexData.channels);
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Uploading metallicRoughness texture for material " << i
                        << " (" << mat.metallicRoughnessTexData.width << "x" << mat.metallicRoughnessTexData.height
                        << ", " << texSize << " bytes)";
                    logOptixLine(oss.str());
                }
                
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_metallicRoughnessTexPixels[i]), texSize));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_metallicRoughnessTexPixels[i]),
                                      mat.metallicRoughnessTexData.pixels, texSize, cudaMemcpyHostToDevice));
                
                m_state->metallicRoughnessTexWidths[i] = mat.metallicRoughnessTexData.width;
                m_state->metallicRoughnessTexHeights[i] = mat.metallicRoughnessTexData.height;
            }

            // Upload emission texture
            if (mat.emissionTexData.pixels && mat.emissionTexData.width > 0 && mat.emissionTexData.height > 0)
            {
                size_t texSize = static_cast<size_t>(mat.emissionTexData.width) * 
                                static_cast<size_t>(mat.emissionTexData.height) * 
                                static_cast<size_t>(mat.emissionTexData.channels);
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Uploading emission texture for material " << i
                        << " (" << mat.emissionTexData.width << "x" << mat.emissionTexData.height
                        << ", " << texSize << " bytes)";
                    logOptixLine(oss.str());
                }
                
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_emissionTexPixels[i]), texSize));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_emissionTexPixels[i]),
                                      mat.emissionTexData.pixels, texSize, cudaMemcpyHostToDevice));
                
                m_state->emissionTexWidths[i] = mat.emissionTexData.width;
                m_state->emissionTexHeights[i] = mat.emissionTexData.height;
            }

            // Upload transmission texture
            if (mat.transmissionTexData.pixels && mat.transmissionTexData.width > 0 && mat.transmissionTexData.height > 0)
            {
                size_t texSize = static_cast<size_t>(mat.transmissionTexData.width) * 
                                static_cast<size_t>(mat.transmissionTexData.height) * 
                                static_cast<size_t>(mat.transmissionTexData.channels);
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Uploading transmission texture for material " << i
                        << " (" << mat.transmissionTexData.width << "x" << mat.transmissionTexData.height
                        << ", " << texSize << " bytes)";
                    logOptixLine(oss.str());
                }
                
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_transmissionTexPixels[i]), texSize));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_transmissionTexPixels[i]),
                                      mat.transmissionTexData.pixels, texSize, cudaMemcpyHostToDevice));
                
                m_state->transmissionTexWidths[i] = mat.transmissionTexData.width;
                m_state->transmissionTexHeights[i] = mat.transmissionTexData.height;
            }
        }
    }

    // Simple SBT with one record per program group
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord
    {
        char      header[OPTIX_SBT_RECORD_HEADER_SIZE];
        HitGroupData data;
    };

    RaygenRecord rg = {};
    OptixResult res = optixSbtRecordPackHeader(m_state->raygenPG, &rg);
    if (res != OPTIX_SUCCESS)
    {
        {
            std::ostringstream oss;
            oss << "[OptiX] Failed to pack raygen record header: " << optixGetErrorString(res);
            logOptixLine(oss.str());
        }
        return;
    }

    MissRecord ms = {};
    res = optixSbtRecordPackHeader(m_state->missPG, &ms);
    if (res != OPTIX_SUCCESS)
    {
        {
            std::ostringstream oss;
            oss << "[OptiX] Failed to pack miss record header: " << optixGetErrorString(res);
            logOptixLine(oss.str());
        }
        return;
    }

    // Build one hit record per mesh + curve
    std::vector<HitRecord> hitRecords;
    if (m_scene && !m_scene->empty())
    {
        const auto& meshes    = m_scene->meshes();
        const auto& curves    = m_scene->curves();
        const auto& materials = m_scene->materials();

        const size_t curveCount = (m_state->hitgroupCurvePG != nullptr) ? curves.size() : 0;
        hitRecords.resize(meshes.size() + curveCount);
        for (size_t i = 0; i < meshes.size(); ++i)
        {
            res = optixSbtRecordPackHeader(m_state->hitgroupPG, &hitRecords[i]);
            if (res != OPTIX_SUCCESS)
            {
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Failed to pack hit record header " << i << ": " << optixGetErrorString(res);
                    logOptixLine(oss.str());
                }
                return;
            }

            const auto& mat = materials[meshes[i].materialIndex];

            // Set material type
            switch (mat.type) {
                case MaterialType::Diffuse:
                    hitRecords[i].data.materialType = MaterialTypeData::Diffuse;
                    break;
                case MaterialType::Conductor:
                    hitRecords[i].data.materialType = MaterialTypeData::Conductor;
                    break;
                case MaterialType::Dielectric:
                    hitRecords[i].data.materialType = MaterialTypeData::Dielectric;
                    break;
                case MaterialType::RoughDielectric:
                    hitRecords[i].data.materialType = MaterialTypeData::RoughDielectric;
                    break;
                case MaterialType::CoatedDiffuse:
                    hitRecords[i].data.materialType = MaterialTypeData::CoatedDiffuse;
                    break;
                case MaterialType::CoatedConductor:
                    hitRecords[i].data.materialType = MaterialTypeData::CoatedConductor;
                    break;
                case MaterialType::Subsurface:
                    hitRecords[i].data.materialType = MaterialTypeData::Subsurface;
                    break;
                case MaterialType::BasicPBR:
                default:
                    hitRecords[i].data.materialType = MaterialTypeData::BasicPBR;
                    break;
            }

            // Set legacy PBR parameters
            hitRecords[i].data.baseColor    = make_float3(mat.baseColor.r,
                                                          mat.baseColor.g,
                                                          mat.baseColor.b);
            {
                std::ostringstream oss;
                oss << "[OptiX] Mesh " << i << " material '" << mat.name
                    << "' type=" << static_cast<int>(mat.type)
                    << " baseColor: (" << mat.baseColor.r << ", "
                    << mat.baseColor.g << ", " << mat.baseColor.b << ")";
                logOptixLine(oss.str());
            }
            hitRecords[i].data.metallic     = mat.metallic;
            hitRecords[i].data.roughness    = mat.roughness;
            hitRecords[i].data.emission     = make_float3(mat.emission.r,
                                                          mat.emission.g,
                                                          mat.emission.b);
            hitRecords[i].data.ior          = mat.ior;
            hitRecords[i].data.transmission = mat.transmission;

            // Set pbrt-v4 material parameters
            hitRecords[i].data.sigma        = mat.sigma;
            hitRecords[i].data.eta          = make_float3(mat.eta.x, mat.eta.y, mat.eta.z);
            hitRecords[i].data.k            = make_float3(mat.k.x, mat.k.y, mat.k.z);
            hitRecords[i].data.uroughness   = mat.uroughness;
            hitRecords[i].data.vroughness   = mat.vroughness;
            hitRecords[i].data.remapRoughness = mat.remapRoughness ? 1 : 0;

            // Fill geometry buffer pointers
            hitRecords[i].data.vertices  = reinterpret_cast<float3*>(m_state->d_meshVerts[i]);
            hitRecords[i].data.indices   = reinterpret_cast<unsigned int*>(m_state->d_meshIndices[i]);
            hitRecords[i].data.normals   = reinterpret_cast<float3*>(m_state->d_meshNormals[i]);
            hitRecords[i].data.hasNormals= (m_state->d_meshNormals[i] != 0) ? 1 : 0;
            hitRecords[i].data.texCoords = reinterpret_cast<float2*>(m_state->d_meshTexCoords[i]);
            hitRecords[i].data.hasTexCoords = (m_state->d_meshTexCoords[i] != 0) ? 1 : 0;
            hitRecords[i].data.materialIndex = static_cast<int>(meshes[i].materialIndex);
            hitRecords[i].data.isCurve = 0;

            // Fill texture pointers
            const uint32_t matIdx = meshes[i].materialIndex;
            
            // Base Color纹理
            if (matIdx < m_state->d_baseColorTexPixels.size() && m_state->d_baseColorTexPixels[matIdx] != 0)
            {
                hitRecords[i].data.baseColorTexPixels = reinterpret_cast<uint8_t*>(m_state->d_baseColorTexPixels[matIdx]);
                hitRecords[i].data.baseColorTexWidth = m_state->baseColorTexWidths[matIdx];
                hitRecords[i].data.baseColorTexHeight = m_state->baseColorTexHeights[matIdx];
                hitRecords[i].data.hasBaseColorTex = 1;
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Mesh " << i << " linking to base color texture: material " << matIdx
                        << " (" << hitRecords[i].data.baseColorTexWidth << "x" << hitRecords[i].data.baseColorTexHeight << ")"
                        << ", texCoords valid: " << (m_state->d_meshTexCoords[i] != 0 ? "yes" : "NO");
                    logOptixLine(oss.str());
                }
            }
            else
            {
                hitRecords[i].data.baseColorTexPixels = nullptr;
                hitRecords[i].data.baseColorTexWidth = 0;
                hitRecords[i].data.baseColorTexHeight = 0;
                hitRecords[i].data.hasBaseColorTex = 0;
            }
            
            // Normal纹理
            if (matIdx < m_state->d_normalTexPixels.size() && m_state->d_normalTexPixels[matIdx] != 0)
            {
                hitRecords[i].data.normalTexPixels = reinterpret_cast<uint8_t*>(m_state->d_normalTexPixels[matIdx]);
                hitRecords[i].data.normalTexWidth = m_state->normalTexWidths[matIdx];
                hitRecords[i].data.normalTexHeight = m_state->normalTexHeights[matIdx];
                hitRecords[i].data.hasNormalTex = 1;
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Mesh " << i << " linking to normal texture: material " << matIdx
                        << " (" << hitRecords[i].data.normalTexWidth << "x" << hitRecords[i].data.normalTexHeight << ")";
                    logOptixLine(oss.str());
                }
            }
            else
            {
                hitRecords[i].data.normalTexPixels = nullptr;
                hitRecords[i].data.normalTexWidth = 0;
                hitRecords[i].data.normalTexHeight = 0;
                hitRecords[i].data.hasNormalTex = 0;
            }
            
            // MetallicRoughness纹理
            if (matIdx < m_state->d_metallicRoughnessTexPixels.size() && m_state->d_metallicRoughnessTexPixels[matIdx] != 0)
            {
                hitRecords[i].data.metallicRoughnessTexPixels = reinterpret_cast<uint8_t*>(m_state->d_metallicRoughnessTexPixels[matIdx]);
                hitRecords[i].data.metallicRoughnessTexWidth = m_state->metallicRoughnessTexWidths[matIdx];
                hitRecords[i].data.metallicRoughnessTexHeight = m_state->metallicRoughnessTexHeights[matIdx];
                hitRecords[i].data.hasMetallicRoughnessTex = 1;
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Mesh " << i << " linking to metallicRoughness texture: material " << matIdx
                        << " (" << hitRecords[i].data.metallicRoughnessTexWidth << "x" << hitRecords[i].data.metallicRoughnessTexHeight << ")";
                    logOptixLine(oss.str());
                }
            }
            else
            {
                hitRecords[i].data.metallicRoughnessTexPixels = nullptr;
                hitRecords[i].data.metallicRoughnessTexWidth = 0;
                hitRecords[i].data.metallicRoughnessTexHeight = 0;
                hitRecords[i].data.hasMetallicRoughnessTex = 0;
            }
            
            // Emission纹理
            if (matIdx < m_state->d_emissionTexPixels.size() && m_state->d_emissionTexPixels[matIdx] != 0)
            {
                hitRecords[i].data.emissionTexPixels = reinterpret_cast<uint8_t*>(m_state->d_emissionTexPixels[matIdx]);
                hitRecords[i].data.emissionTexWidth = m_state->emissionTexWidths[matIdx];
                hitRecords[i].data.emissionTexHeight = m_state->emissionTexHeights[matIdx];
                hitRecords[i].data.hasEmissionTex = 1;
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Mesh " << i << " linking to emission texture: material " << matIdx
                        << " (" << hitRecords[i].data.emissionTexWidth << "x" << hitRecords[i].data.emissionTexHeight << ")";
                    logOptixLine(oss.str());
                }
            }
            else
            {
                hitRecords[i].data.emissionTexPixels = nullptr;
                hitRecords[i].data.emissionTexWidth = 0;
                hitRecords[i].data.emissionTexHeight = 0;
                hitRecords[i].data.hasEmissionTex = 0;
            }
            
            // Transmission纹理
            if (matIdx < m_state->d_transmissionTexPixels.size() && m_state->d_transmissionTexPixels[matIdx] != 0)
            {
                hitRecords[i].data.transmissionTexPixels = reinterpret_cast<uint8_t*>(m_state->d_transmissionTexPixels[matIdx]);
                hitRecords[i].data.transmissionTexWidth = m_state->transmissionTexWidths[matIdx];
                hitRecords[i].data.transmissionTexHeight = m_state->transmissionTexHeights[matIdx];
                hitRecords[i].data.hasTransmissionTex = 1;
                
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Mesh " << i << " linking to transmission texture: material " << matIdx
                        << " (" << hitRecords[i].data.transmissionTexWidth << "x" << hitRecords[i].data.transmissionTexHeight << ")";
                    logOptixLine(oss.str());
                }
            }
            else
            {
                hitRecords[i].data.transmissionTexPixels = nullptr;
                hitRecords[i].data.transmissionTexWidth = 0;
                hitRecords[i].data.transmissionTexHeight = 0;
                hitRecords[i].data.hasTransmissionTex = 0;
            }
        }

        for (size_t i = 0; i < curves.size(); ++i)
        {
            size_t idx = meshes.size() + i;
            if (!m_state->hitgroupCurvePG)
            {
                logOptixLine("[OptiX] Curve hitgroup unavailable; skipping curve records.");
                break;
            }

            res = optixSbtRecordPackHeader(m_state->hitgroupCurvePG, &hitRecords[idx]);
            if (res != OPTIX_SUCCESS)
            {
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Failed to pack curve hit record header " << i << ": " << optixGetErrorString(res);
                    logOptixLine(oss.str());
                }
                return;
            }

            const auto& mat = materials[curves[i].materialIndex];

            switch (mat.type) {
                case MaterialType::Diffuse:
                    hitRecords[idx].data.materialType = MaterialTypeData::Diffuse;
                    break;
                case MaterialType::Conductor:
                    hitRecords[idx].data.materialType = MaterialTypeData::Conductor;
                    break;
                case MaterialType::Dielectric:
                    hitRecords[idx].data.materialType = MaterialTypeData::Dielectric;
                    break;
                case MaterialType::RoughDielectric:
                    hitRecords[idx].data.materialType = MaterialTypeData::RoughDielectric;
                    break;
                case MaterialType::CoatedDiffuse:
                    hitRecords[idx].data.materialType = MaterialTypeData::CoatedDiffuse;
                    break;
                case MaterialType::CoatedConductor:
                    hitRecords[idx].data.materialType = MaterialTypeData::CoatedConductor;
                    break;
                case MaterialType::Subsurface:
                    hitRecords[idx].data.materialType = MaterialTypeData::Subsurface;
                    break;
                case MaterialType::BasicPBR:
                default:
                    hitRecords[idx].data.materialType = MaterialTypeData::BasicPBR;
                    break;
            }

            hitRecords[idx].data.baseColor    = make_float3(mat.baseColor.r, mat.baseColor.g, mat.baseColor.b);
            hitRecords[idx].data.metallic     = mat.metallic;
            hitRecords[idx].data.roughness    = mat.roughness;
            hitRecords[idx].data.emission     = make_float3(mat.emission.r, mat.emission.g, mat.emission.b);
            hitRecords[idx].data.ior          = mat.ior;
            hitRecords[idx].data.transmission = mat.transmission;
            hitRecords[idx].data.sigma        = mat.sigma;
            hitRecords[idx].data.eta          = make_float3(mat.eta.x, mat.eta.y, mat.eta.z);
            hitRecords[idx].data.k            = make_float3(mat.k.x, mat.k.y, mat.k.z);
            hitRecords[idx].data.uroughness   = mat.uroughness;
            hitRecords[idx].data.vroughness   = mat.vroughness;
            hitRecords[idx].data.remapRoughness = mat.remapRoughness ? 1 : 0;

            hitRecords[idx].data.vertices  = nullptr;
            hitRecords[idx].data.indices   = nullptr;
            hitRecords[idx].data.normals   = nullptr;
            hitRecords[idx].data.hasNormals= 0;
            hitRecords[idx].data.texCoords = nullptr;
            hitRecords[idx].data.hasTexCoords = 0;
            hitRecords[idx].data.materialIndex = static_cast<int>(curves[i].materialIndex);
            hitRecords[idx].data.isCurve = 1;

            hitRecords[idx].data.baseColorTexPixels = nullptr;
            hitRecords[idx].data.baseColorTexWidth = 0;
            hitRecords[idx].data.baseColorTexHeight = 0;
            hitRecords[idx].data.hasBaseColorTex = 0;
            hitRecords[idx].data.normalTexPixels = nullptr;
            hitRecords[idx].data.normalTexWidth = 0;
            hitRecords[idx].data.normalTexHeight = 0;
            hitRecords[idx].data.hasNormalTex = 0;
            hitRecords[idx].data.metallicRoughnessTexPixels = nullptr;
            hitRecords[idx].data.metallicRoughnessTexWidth = 0;
            hitRecords[idx].data.metallicRoughnessTexHeight = 0;
            hitRecords[idx].data.hasMetallicRoughnessTex = 0;
            hitRecords[idx].data.emissionTexPixels = nullptr;
            hitRecords[idx].data.emissionTexWidth = 0;
            hitRecords[idx].data.emissionTexHeight = 0;
            hitRecords[idx].data.hasEmissionTex = 0;
            hitRecords[idx].data.transmissionTexPixels = nullptr;
            hitRecords[idx].data.transmissionTexWidth = 0;
            hitRecords[idx].data.transmissionTexHeight = 0;
            hitRecords[idx].data.hasTransmissionTex = 0;
        }
    }
    else
    {
        // Default hit record for empty scene
        hitRecords.resize(1);
        res = optixSbtRecordPackHeader(m_state->hitgroupPG, &hitRecords[0]);
        if (res != OPTIX_SUCCESS)
        {
            {
                std::ostringstream oss;
                oss << "[OptiX] Failed to pack default hit record header: " << optixGetErrorString(res);
                logOptixLine(oss.str());
            }
            return;
        }

        hitRecords[0].data.baseColor    = make_float3(0.8f, 0.8f, 0.8f);
        hitRecords[0].data.metallic     = 0.f;
        hitRecords[0].data.roughness    = 0.5f;
        hitRecords[0].data.emission     = make_float3(0.f, 0.f, 0.f);
        hitRecords[0].data.ior          = 1.45f;
        hitRecords[0].data.transmission = 0.f;
        hitRecords[0].data.vertices   = nullptr;
        hitRecords[0].data.indices    = nullptr;
        hitRecords[0].data.normals    = nullptr;
        hitRecords[0].data.hasNormals = 0;
        hitRecords[0].data.isCurve = 0;
    }

    const size_t sbtSize =
        sizeof(RaygenRecord) + sizeof(MissRecord) +
        hitRecords.size() * sizeof(HitRecord);

    if (m_state->d_sbt)
        cudaFree(reinterpret_cast<void*>(m_state->d_sbt));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_sbt), sbtSize));

    CUdeviceptr p = m_state->d_sbt;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(p), &rg,
                          sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    m_state->sbt.raygenRecord = p;
    p += sizeof(RaygenRecord);

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(p), &ms,
                          sizeof(MissRecord), cudaMemcpyHostToDevice));
    m_state->sbt.missRecordBase          = p;
    m_state->sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_state->sbt.missRecordCount         = 1;
    p += sizeof(MissRecord);

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(p),
                          hitRecords.data(),
                          hitRecords.size() * sizeof(HitRecord),
                          cudaMemcpyHostToDevice));
    m_state->sbt.hitgroupRecordBase          = p;
    m_state->sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    m_state->sbt.hitgroupRecordCount         =
        static_cast<unsigned int>(hitRecords.size());
#endif
}

// ---------------------------------------------------------------------------
// buildAccel – constructs a GAS from the loaded scene
// ---------------------------------------------------------------------------

void OptixRenderer::buildAccel()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (!m_optixReady) {
        logOptixLine("[OptiX] buildAccel: EARLY RETURN - optixReady=false");
        return;
    }
    
    // REMOVED the empty check - allow buildAccel to run even if scene currently has no meshes
    // The scene might be populating asynchronously
    
    if (!m_scene) {
        logOptixLine("[OptiX] buildAccel: EARLY RETURN - scene=null");
        return;
    }

    // Upload all meshes as a single GAS with one build input per mesh
    const auto& meshes = m_scene->meshes();
    const auto& curves = m_scene->curves();
    
    {
        std::ostringstream oss;
        oss << "[OptiX] buildAccel START: meshes=" << meshes.size() << " curves=" << curves.size();
        logOptixLine(oss.str());
    }

    std::vector<CUdeviceptr>       d_vertices(meshes.size());
    std::vector<CUdeviceptr>       d_indices (meshes.size());

    const bool hasCurves = !curves.empty();
    const size_t buildInputCount = meshes.size() + (hasCurves ? 1 : 0);
    std::vector<OptixBuildInput>   buildInputs(buildInputCount);
    std::vector<uint32_t>          flags(meshes.size(), OPTIX_GEOMETRY_FLAG_NONE);

    if (buildInputCount == 0)
    {
        logOptixLine("[OptiX] buildAccel: no geometry, skipping GAS build");
        m_state->gasHandle = 0;
        if (m_state->d_gas)
        {
            cudaFree(reinterpret_cast<void*>(m_state->d_gas));
            m_state->d_gas = 0;
        }
        return;
    }

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const auto& m = meshes[i];

        // Vertices (only xyz)
        std::vector<float3> verts(m.vertices.size());
        for (size_t j = 0; j < m.vertices.size(); ++j)
            verts[j] = make_float3(m.vertices[j].position.x,
                                   m.vertices[j].position.y,
                                   m.vertices[j].position.z);

        // DEBUG: Output vertex range for first mesh
        if (i == 0 && !verts.empty())
        {
            float minX = verts[0].x, maxX = verts[0].x;
            float minY = verts[0].y, maxY = verts[0].y;
            float minZ = verts[0].z, maxZ = verts[0].z;
            for (const auto& v : verts)
            {
                minX = fminf(minX, v.x);  maxX = fmaxf(maxX, v.x);
                minY = fminf(minY, v.y);  maxY = fmaxf(maxY, v.y);
                minZ = fminf(minZ, v.z);  maxZ = fmaxf(maxZ, v.z);
            }
            {
                std::ostringstream oss;
                oss << "[OptiX] Mesh 0 vertex extent: X[" << minX << "," << maxX << "] "
                    << "Y[" << minY << "," << maxY << "] Z[" << minZ << "," << maxZ << "]";
                logOptixLine(oss.str());
            }
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices[i]),
                              verts.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices[i]),
                              verts.data(), verts.size() * sizeof(float3),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices[i]),
                              m.indices.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices[i]),
                              m.indices.data(),
                              m.indices.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        auto& bi = buildInputs[i];
        bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        bi.triangleArray.vertexBuffers         = &d_vertices[i];
        bi.triangleArray.numVertices           =
            static_cast<unsigned int>(m.vertices.size());
        bi.triangleArray.vertexFormat          = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.vertexStrideInBytes   = sizeof(float3);
        bi.triangleArray.indexBuffer           = d_indices[i];
        bi.triangleArray.numIndexTriplets      =
            static_cast<unsigned int>(m.indices.size() / 3);
        bi.triangleArray.indexFormat           = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        bi.triangleArray.indexStrideInBytes    = 0;
        bi.triangleArray.flags                 = &flags[i];
        bi.triangleArray.numSbtRecords         = 1;
        bi.triangleArray.sbtIndexOffsetBuffer  = 0;
        bi.triangleArray.sbtIndexOffsetSizeInBytes  = 0;
        bi.triangleArray.sbtIndexOffsetStrideInBytes= 0;
    }

    if (hasCurves && m_state->hitgroupCurvePG)
    {
        std::vector<CurveData> curveData;
        std::vector<OptixAabb> curveAabbs;
        std::vector<uint32_t>  curveSbtIndices;
        std::vector<uint32_t>  curveFlags;

        curveData.reserve(curves.size());
        curveAabbs.reserve(curves.size());
        curveSbtIndices.reserve(curves.size());
        curveFlags.resize(curves.size(), OPTIX_GEOMETRY_FLAG_NONE);

        for (size_t i = 0; i < curves.size(); ++i)
        {
            const auto& c = curves[i];
            CurveData cd = {};
            for (int j = 0; j < 4; ++j)
                cd.cp[j] = make_float3(c.cp[j].x, c.cp[j].y, c.cp[j].z);
            cd.width0 = c.width0;
            cd.width1 = c.width1;
            cd.type = static_cast<CurveTypeData>(c.type == CurveType::Ribbon ? 2 : (c.type == CurveType::Cylinder ? 1 : 0));
            cd.n0 = make_float3(c.n0.x, c.n0.y, c.n0.z);
            cd.n1 = make_float3(c.n1.x, c.n1.y, c.n1.z);
            cd.hasNormals = c.hasNormals ? 1 : 0;
            cd.materialIndex = static_cast<int>(c.materialIndex);
            curveData.push_back(cd);

            float minX = c.cp[0].x, maxX = c.cp[0].x;
            float minY = c.cp[0].y, maxY = c.cp[0].y;
            float minZ = c.cp[0].z, maxZ = c.cp[0].z;
            for (int j = 1; j < 4; ++j)
            {
                minX = fminf(minX, c.cp[j].x);
                maxX = fmaxf(maxX, c.cp[j].x);
                minY = fminf(minY, c.cp[j].y);
                maxY = fmaxf(maxY, c.cp[j].y);
                minZ = fminf(minZ, c.cp[j].z);
                maxZ = fmaxf(maxZ, c.cp[j].z);
            }
            float w = 0.5f * fmaxf(c.width0, c.width1);
            OptixAabb aabb = {minX - w, minY - w, minZ - w, maxX + w, maxY + w, maxZ + w};
            curveAabbs.push_back(aabb);

            curveSbtIndices.push_back(static_cast<uint32_t>(meshes.size() + i));
        }

        if (m_state->d_curveData)
            cudaFree(reinterpret_cast<void*>(m_state->d_curveData));
        if (m_state->d_curveAabbs)
            cudaFree(reinterpret_cast<void*>(m_state->d_curveAabbs));
        if (m_state->d_curveSbtIndices)
            cudaFree(reinterpret_cast<void*>(m_state->d_curveSbtIndices));

        m_state->curveCount = static_cast<int>(curveData.size());
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_curveData),
                              curveData.size() * sizeof(CurveData)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_curveData),
                              curveData.data(), curveData.size() * sizeof(CurveData),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_curveAabbs),
                              curveAabbs.size() * sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_curveAabbs),
                              curveAabbs.data(), curveAabbs.size() * sizeof(OptixAabb),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_curveSbtIndices),
                              curveSbtIndices.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_curveSbtIndices),
                              curveSbtIndices.data(), curveSbtIndices.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        m_state->launchParams.curves = reinterpret_cast<CurveData*>(m_state->d_curveData);
        m_state->launchParams.numCurves = static_cast<int>(curveData.size());

        OptixBuildInput& bi = buildInputs[buildInputs.size() - 1];
        bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers = &m_state->d_curveAabbs;
        bi.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(curveAabbs.size());
        bi.customPrimitiveArray.flags = curveFlags.data();
        bi.customPrimitiveArray.numSbtRecords = static_cast<unsigned int>(meshes.size() + curveAabbs.size());
        bi.customPrimitiveArray.sbtIndexOffsetBuffer = m_state->d_curveSbtIndices;
        bi.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        bi.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    }
    else
    {
        if (m_state->d_curveData)
        {
            cudaFree(reinterpret_cast<void*>(m_state->d_curveData));
            m_state->d_curveData = 0;
        }
        if (m_state->d_curveAabbs)
        {
            cudaFree(reinterpret_cast<void*>(m_state->d_curveAabbs));
            m_state->d_curveAabbs = 0;
        }
        if (m_state->d_curveSbtIndices)
        {
            cudaFree(reinterpret_cast<void*>(m_state->d_curveSbtIndices));
            m_state->d_curveSbtIndices = 0;
        }
        m_state->launchParams.curves = nullptr;
        m_state->launchParams.numCurves = 0;
        m_state->curveCount = 0;
    }

    OptixAccelBuildOptions accelOpts = {};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                           OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_state->context, &accelOpts,
        buildInputs.data(), static_cast<unsigned int>(buildInputs.size()),
        &bufSizes));

    CUdeviceptr d_temp, d_output;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),   bufSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bufSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        m_state->context, m_state->stream,
        &accelOpts,
        buildInputs.data(), static_cast<unsigned int>(buildInputs.size()),
        d_temp,   bufSizes.tempSizeInBytes,
        d_output, bufSizes.outputSizeInBytes,
        &m_state->gasHandle, nullptr, 0));

    CUDA_CHECK(cudaStreamSynchronize(m_state->stream));
    
    {
        std::ostringstream oss;
        oss << "[OptiX] optixAccelBuild completed: gasHandle=" << std::hex << m_state->gasHandle 
            << std::dec << " outputSize=" << bufSizes.outputSizeInBytes << " bytes";
        logOptixLine(oss.str());
    }
    
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));

    // Free old GAS
    if (m_state->d_gas)
        cudaFree(reinterpret_cast<void*>(m_state->d_gas));
    m_state->d_gas = d_output;

    // Keep vertex/index buffers alive for closesthit to access
    // Free old ones first
    for (auto& p : m_state->d_meshVerts)   cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_meshNormals) cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_meshIndices) cudaFree(reinterpret_cast<void*>(p));

    // Free old texture buffers and coordinates
    for (auto& p : m_state->d_meshTexCoords)           cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_baseColorTexPixels)      cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_normalTexPixels)         cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_metallicRoughnessTexPixels) cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_emissionTexPixels)       cudaFree(reinterpret_cast<void*>(p));
    for (auto& p : m_state->d_transmissionTexPixels)   cudaFree(reinterpret_cast<void*>(p));
    m_state->d_meshTexCoords.clear();
    m_state->d_baseColorTexPixels.clear();
    m_state->baseColorTexWidths.clear();
    m_state->baseColorTexHeights.clear();
    m_state->d_normalTexPixels.clear();
    m_state->normalTexWidths.clear();
    m_state->normalTexHeights.clear();
    m_state->d_metallicRoughnessTexPixels.clear();
    m_state->metallicRoughnessTexWidths.clear();
    m_state->metallicRoughnessTexHeights.clear();
    m_state->d_emissionTexPixels.clear();
    m_state->emissionTexWidths.clear();
    m_state->emissionTexHeights.clear();
    m_state->d_transmissionTexPixels.clear();
    m_state->transmissionTexWidths.clear();
    m_state->transmissionTexHeights.clear();

    m_state->d_meshVerts   = d_vertices;
    m_state->d_meshIndices = d_indices;

    // Upload vertex texture coordinates
    m_state->d_meshTexCoords.resize(meshes.size(), 0);
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const auto& m = meshes[i];
        if (!m.vertices.empty())
        {
            std::vector<float2> texCoords(m.vertices.size());
            for (size_t j = 0; j < m.vertices.size(); ++j)
                texCoords[j] = make_float2(m.vertices[j].texCoord.x,
                                           m.vertices[j].texCoord.y);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_meshTexCoords[i]),
                                  texCoords.size() * sizeof(float2)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_meshTexCoords[i]),
                                  texCoords.data(), texCoords.size() * sizeof(float2),
                                  cudaMemcpyHostToDevice));
        }
    }

    // Upload vertex normals
    m_state->d_meshNormals.resize(meshes.size(), 0);
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const auto& m = meshes[i];
        if (!m.vertices.empty() && m.vertices[0].normal != glm::vec3(0.f))
        {
            std::vector<float3> norms(m.vertices.size());
            for (size_t j = 0; j < m.vertices.size(); ++j)
                norms[j] = make_float3(m.vertices[j].normal.x,
                                       m.vertices[j].normal.y,
                                       m.vertices[j].normal.z);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_meshNormals[i]),
                                  norms.size() * sizeof(float3)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_meshNormals[i]),
                                  norms.data(), norms.size() * sizeof(float3),
                                  cudaMemcpyHostToDevice));
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// buildLights – upload lights and ambient to GPU
// ---------------------------------------------------------------------------
void OptixRenderer::buildLights()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    // Free previous light buffer if any
    if (m_state->launchParams.lights)
    {
        cudaFree(m_state->launchParams.lights);
        m_state->launchParams.lights = nullptr;
        m_state->launchParams.numLights = 0;
    }
    
    // Reset directional lights
    m_state->launchParams.numDirectionalLights = 0;
    
    if (!m_scene) return;
    const auto& lights = m_scene->lights();
    m_state->launchParams.numLights = static_cast<int>(lights.size());
    
    if (!lights.empty())
    {
        // Convert to device struct
        std::vector<LightData> lightData;
        lightData.reserve(lights.size());
        
        // Also fill directional lights array
        int dirLightCount = 0;
        for (const auto& l : lights)
        {
            LightData ld;
            if (l.type == LightType::Point)
                ld.type = LightTypeData::Point;
            else if (l.type == LightType::Spot)
                ld.type = LightTypeData::Spot;
            else
                ld.type = LightTypeData::Directional;
            ld.position = make_float3(l.position.x, l.position.y, l.position.z);
            ld.direction = make_float3(l.direction.x, l.direction.y, l.direction.z);
            ld.intensity = make_float3(l.intensity.x, l.intensity.y, l.intensity.z);
            ld.cosInner = l.cosInner;
            ld.cosOuter = l.cosOuter;
            lightData.push_back(ld);
            
            // Fill directional light array for shader
            if (l.type == LightType::Directional && dirLightCount < 8) {
                m_state->launchParams.directionalLights[dirLightCount].direction = ld.direction;
                m_state->launchParams.directionalLights[dirLightCount].intensity = ld.intensity;
                dirLightCount++;
            }
        }
        m_state->launchParams.numDirectionalLights = dirLightCount;
        
        cudaMalloc(reinterpret_cast<void**>(&m_state->launchParams.lights),
                   lightData.size() * sizeof(LightData));
        cudaMemcpy(m_state->launchParams.lights, lightData.data(),
                   lightData.size() * sizeof(LightData), cudaMemcpyHostToDevice);
    }
    // Ambient
    const glm::vec3& amb = m_scene->ambientInt();
    m_state->launchParams.ambientIntensity = make_float3(amb.x, amb.y, amb.z);

    // Environment map
    if (m_state->d_envmapPixels)
    {
        cudaFree(reinterpret_cast<void*>(m_state->d_envmapPixels));
        m_state->d_envmapPixels = 0;
        m_state->envmapWidth = 0;
        m_state->envmapHeight = 0;
    }
    if (m_state->d_envmapPixelsF)
    {
        cudaFree(reinterpret_cast<void*>(m_state->d_envmapPixelsF));
        m_state->d_envmapPixelsF = 0;
        m_state->envmapWidth = 0;
        m_state->envmapHeight = 0;
    }

    m_state->launchParams.envmapPixels = nullptr;
    m_state->launchParams.envmapPixelsF = nullptr;
    m_state->launchParams.envmapWidth = 0;
    m_state->launchParams.envmapHeight = 0;
    m_state->launchParams.hasEnvmap = 0;
    m_state->launchParams.envmapIsHDR = 0;
    m_state->launchParams.environmentScale = make_float3(amb.x, amb.y, amb.z);
    // Initialize light transform to identity matrix (columns)
    m_state->launchParams.envLightRight = make_float3(1.f, 0.f, 0.f);
    m_state->launchParams.envLightUp = make_float3(0.f, 1.f, 0.f);
    m_state->launchParams.envLightFwd = make_float3(0.f, 0.f, 1.f);

    const auto& env = m_scene->environment();
    if (env.valid)
    {
        if (env.isHDR && !env.hdrPixels.empty() && env.hdrWidth > 0 && env.hdrHeight > 0)
        {
            size_t texSize = static_cast<size_t>(env.hdrWidth) *
                             static_cast<size_t>(env.hdrHeight) *
                             static_cast<size_t>(env.hdrChannels) * sizeof(float);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_envmapPixelsF), texSize));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_envmapPixelsF),
                                  env.hdrPixels.data(), texSize, cudaMemcpyHostToDevice));
            m_state->envmapWidth = env.hdrWidth;
            m_state->envmapHeight = env.hdrHeight;

            m_state->launchParams.envmapPixelsF = reinterpret_cast<float4*>(m_state->d_envmapPixelsF);
            m_state->launchParams.envmapWidth = env.hdrWidth;
            m_state->launchParams.envmapHeight = env.hdrHeight;
            m_state->launchParams.hasEnvmap = 1;
            m_state->launchParams.envmapIsHDR = 1;
            // For HDR, don't apply the 8-bit LDR scale factor
            // HDR preserves full linear values, so just use the baseline (L parameter only, no scale multiplier)
            float scaleReduction = 40.0f;  // Cancel out the scale=40 from PBRT file
            glm::vec3 hdrScale = env.scale / scaleReduction;
            m_state->launchParams.environmentScale = make_float3(hdrScale.x, hdrScale.y, hdrScale.z);
            
            // Set environment light transform (rotation matrix columns)
            m_state->launchParams.envLightRight = make_float3(env.lightTransform[0][0], env.lightTransform[1][0], env.lightTransform[2][0]);
            m_state->launchParams.envLightUp = make_float3(env.lightTransform[0][1], env.lightTransform[1][1], env.lightTransform[2][1]);
            m_state->launchParams.envLightFwd = make_float3(env.lightTransform[0][2], env.lightTransform[1][2], env.lightTransform[2][2]);
        }
        else if (env.texData.pixels && env.texData.width > 0 && env.texData.height > 0)
        {
            size_t texSize = static_cast<size_t>(env.texData.width) *
                             static_cast<size_t>(env.texData.height) *
                             static_cast<size_t>(env.texData.channels);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_envmapPixels), texSize));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_state->d_envmapPixels),
                                  env.texData.pixels, texSize, cudaMemcpyHostToDevice));
            m_state->envmapWidth = env.texData.width;
            m_state->envmapHeight = env.texData.height;

            m_state->launchParams.envmapPixels = reinterpret_cast<uint8_t*>(m_state->d_envmapPixels);
            m_state->launchParams.envmapWidth = env.texData.width;
            m_state->launchParams.envmapHeight = env.texData.height;
            m_state->launchParams.hasEnvmap = 1;
            m_state->launchParams.envmapIsHDR = 0;
            m_state->launchParams.environmentScale = make_float3(env.scale.x, env.scale.y, env.scale.z);
            
            // Set environment light transform (rotation matrix columns)
            m_state->launchParams.envLightRight = make_float3(env.lightTransform[0][0], env.lightTransform[1][0], env.lightTransform[2][0]);
            m_state->launchParams.envLightUp = make_float3(env.lightTransform[0][1], env.lightTransform[1][1], env.lightTransform[2][1]);
            m_state->launchParams.envLightFwd = make_float3(env.lightTransform[0][2], env.lightTransform[1][2], env.lightTransform[2][2]);
        }
    }

    {
        std::ostringstream oss;
        oss << "[OptiX] Lights: " << m_state->launchParams.numLights
            << ", directional: " << m_state->launchParams.numDirectionalLights
            << ", ambient: (" << m_state->launchParams.ambientIntensity.x
            << ", " << m_state->launchParams.ambientIntensity.y
            << ", " << m_state->launchParams.ambientIntensity.z << ")"
            << ", envmap: " << (m_state->launchParams.hasEnvmap ? "yes" : "no")
            << " (" << m_state->launchParams.envmapWidth
            << "x" << m_state->launchParams.envmapHeight << ")"
            << ", envScale: (" << m_state->launchParams.environmentScale.x
            << ", " << m_state->launchParams.environmentScale.y
            << ", " << m_state->launchParams.environmentScale.z << ")";
        logOptixLine(oss.str());
    }
#endif
}

// ---------------------------------------------------------------------------
// setScene / resize / resetAccumulation
// ---------------------------------------------------------------------------

void OptixRenderer::setScene(const Scene* scene)
{
    logOptixLine("[OptiX] setScene called");
    m_scene = scene;
    if (!m_optixReady) {
        logOptixLine("[OptiX] setScene: optixReady=false, returning");
        return;
    }

    logOptixLine("[OptiX] setScene: calling buildAccel");
    buildAccel();
    
    logOptixLine("[OptiX] setScene: buildAccel done, checking SBT");
    if (m_state->raygenPG && m_state->missPG && m_state->hitgroupPG) {
        logOptixLine("[OptiX] setScene: calling buildSBT");
        buildSBT();
    } else {
        logOptixLine("[OptiX] setScene: SBT not ready - raygenPG=" + std::to_string(m_state->raygenPG != nullptr) +
                     " missPG=" + std::to_string(m_state->missPG != nullptr) + 
                     " hitgroupPG=" + std::to_string(m_state->hitgroupPG != nullptr));
    }
    
    logOptixLine("[OptiX] setScene: calling buildLights");
    buildLights();
    
    logOptixLine("[OptiX] setScene: calling resetAccumulation");
    resetAccumulation();
    logOptixLine("[OptiX] setScene DONE");
}

void OptixRenderer::resize(int width, int height)
{
    m_width  = width;
    m_height = height;

#ifdef PATHTRACER_OPTIX_ENABLED
    if (m_optixReady)
    {
        // Unregister old PBO
        if (m_state->pboResource)
        {
            cudaGraphicsUnregisterResource(m_state->pboResource);
            m_state->pboResource = nullptr;
        }
        // Resize accumulation buffer
        const size_t bufSize =
            static_cast<size_t>(m_width * m_height) * 4 * sizeof(float);
        cudaFree(reinterpret_cast<void*>(m_state->d_accum));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_accum), bufSize));
        cudaFree(reinterpret_cast<void*>(m_state->d_beauty));
        cudaFree(reinterpret_cast<void*>(m_state->d_albedo));
        cudaFree(reinterpret_cast<void*>(m_state->d_normal));
        cudaFree(reinterpret_cast<void*>(m_state->d_denoised));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_beauty), bufSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_albedo), bufSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_normal), bufSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_denoised), bufSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_accum), 0, bufSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_beauty), 0, bufSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_albedo), 0, bufSize));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_state->d_normal), 0, bufSize));

        if (m_state->denoiser)
        {
            OptixDenoiserSizes denoiseSizes = {};
            OPTIX_CHECK(optixDenoiserComputeMemoryResources(
                m_state->denoiser, m_width, m_height, &denoiseSizes));
            m_state->denoiserStateSize = denoiseSizes.stateSizeInBytes;
            m_state->denoiserScratchSize = denoiseSizes.withoutOverlapScratchSizeInBytes;

            if (m_state->d_denoiserState)
                cudaFree(reinterpret_cast<void*>(m_state->d_denoiserState));
            if (m_state->d_denoiserScratch)
                cudaFree(reinterpret_cast<void*>(m_state->d_denoiserScratch));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_denoiserState),
                                  m_state->denoiserStateSize));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_denoiserScratch),
                                  m_state->denoiserScratchSize));

            OPTIX_CHECK(optixDenoiserSetup(
                m_state->denoiser, m_state->stream,
                m_width, m_height,
                m_state->d_denoiserState, m_state->denoiserStateSize,
                m_state->d_denoiserScratch, m_state->denoiserScratchSize));
            m_state->denoiserReady = true;
        }
    }
#endif
    resetAccumulation();
}

void OptixRenderer::resetAccumulation()
{
    m_accumulatedSamples = 0;

#ifdef PATHTRACER_OPTIX_ENABLED
    if (m_optixReady && m_state->d_accum)
    {
        const size_t bufSize =
            static_cast<size_t>(m_width * m_height) * 4 * sizeof(float);
        cudaMemset(reinterpret_cast<void*>(m_state->d_accum), 0, bufSize);
        if (m_state->d_beauty)
            cudaMemset(reinterpret_cast<void*>(m_state->d_beauty), 0, bufSize);
        if (m_state->d_albedo)
            cudaMemset(reinterpret_cast<void*>(m_state->d_albedo), 0, bufSize);
        if (m_state->d_normal)
            cudaMemset(reinterpret_cast<void*>(m_state->d_normal), 0, bufSize);
    }
#endif
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

void OptixRenderer::render(const Camera& camera,
                           unsigned int  glPBO,
                           int           spp,
                           int           maxBounces)
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (m_optixReady)
    {
        // Register PBO with CUDA on first use / after resize
        if (!m_state->pboResource)
        {
            CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
                &m_state->pboResource, glPBO,
                cudaGraphicsRegisterFlagsWriteDiscard));
        }

        // Map PBO
        CUDA_CHECK(cudaGraphicsMapResources(1, &m_state->pboResource, m_state->stream));
        size_t bufSize;
        float4* d_output;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&d_output), &bufSize, m_state->pboResource));

        // Fill launch params
        const float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);
        auto& lp = m_state->launchParams;

        const glm::vec3 llc = camera.lowerLeftCorner(aspect);
        const glm::vec3 hor = camera.horizontal(aspect);
        const glm::vec3 ver = camera.vertical();
        const glm::vec3 ori = camera.position();

        lp.camera.origin       = make_float3(ori.x, ori.y, ori.z);
        lp.camera.lowerLeft    = make_float3(llc.x, llc.y, llc.z);
        lp.camera.horizontal   = make_float3(hor.x, hor.y, hor.z);
        lp.camera.vertical     = make_float3(ver.x, ver.y, ver.z);
        lp.output              = reinterpret_cast<float4*>(m_state->d_beauty);
        lp.accum               = reinterpret_cast<float4*>(m_state->d_accum);
        lp.albedo              = reinterpret_cast<float4*>(m_state->d_albedo);
        lp.normal              = reinterpret_cast<float4*>(m_state->d_normal);
        lp.width               = m_width;
        lp.height              = m_height;
        lp.sampleIndex         = m_accumulatedSamples;
        lp.maxBounces          = maxBounces;
        lp.traversable         = m_state->gasHandle;
        lp.debugDirectional    = m_debugDirectional ? 1 : 0;
        lp.debugMode           = m_debugMode;

        if (!m_state->d_launchParams)
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_launchParams),
                                  sizeof(LaunchParams)));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_state->d_launchParams),
                                   &lp, sizeof(LaunchParams),
                                   cudaMemcpyHostToDevice, m_state->stream));

        // Launch
        for (int s = 0; s < spp; ++s)
        {
            lp.sampleIndex = m_accumulatedSamples + s;
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(m_state->d_launchParams),
                &lp, sizeof(LaunchParams),
                cudaMemcpyHostToDevice, m_state->stream));
            OPTIX_CHECK(optixLaunch(
                m_state->pipeline, m_state->stream,
                m_state->d_launchParams, sizeof(LaunchParams),
                &m_state->sbt, m_width, m_height, 1));
        }
        m_accumulatedSamples += static_cast<uint32_t>(spp);

        // One-time diagnostic: sanity-check first pixel and traversal readiness.
        {
            static bool loggedOnce = false;
            if (!loggedOnce)
            {
                float4 firstPixel = {0.f, 0.f, 0.f, 0.f};
                CUDA_CHECK(cudaMemcpyAsync(&firstPixel,
                                           reinterpret_cast<void*>(m_state->d_beauty),
                                           sizeof(float4),
                                           cudaMemcpyDeviceToHost,
                                           m_state->stream));
                CUDA_CHECK(cudaStreamSynchronize(m_state->stream));
                {
                    std::ostringstream oss;
                    oss << "[OptiX] First pixel radiance: ("
                        << firstPixel.x << ", " << firstPixel.y << ", "
                        << firstPixel.z << ", " << firstPixel.w << ")";
                    logOptixLine(oss.str());
                }
                {
                    std::ostringstream oss;
                    oss << "[OptiX] Traversable: "
                        << (m_state->gasHandle ? "ok" : "null")
                        << ", denoiser=" << (m_state->denoiserReady ? "on" : "off")
                        << ", meshes=" << (m_scene ? m_scene->meshes().size() : 0);
                    logOptixLine(oss.str());
                }
                loggedOnce = true;
            }
        }

        const size_t imageSize = static_cast<size_t>(m_width * m_height) * sizeof(float4);
        const bool canDenoise = (m_state->denoiserReady && m_debugMode == 0);
        if (canDenoise)
        {
            OptixImage2D input = {};
            input.data = m_state->d_beauty;
            input.width = m_width;
            input.height = m_height;
            input.rowStrideInBytes = m_width * sizeof(float4);
            input.pixelStrideInBytes = sizeof(float4);
            input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

            OptixImage2D output = {};
            output.data = m_state->d_denoised;
            output.width = m_width;
            output.height = m_height;
            output.rowStrideInBytes = m_width * sizeof(float4);
            output.pixelStrideInBytes = sizeof(float4);
            output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

            OptixImage2D albedo = {};
            albedo.data = m_state->d_albedo;
            albedo.width = m_width;
            albedo.height = m_height;
            albedo.rowStrideInBytes = m_width * sizeof(float4);
            albedo.pixelStrideInBytes = sizeof(float4);
            albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;

            OptixImage2D normal = {};
            normal.data = m_state->d_normal;
            normal.width = m_width;
            normal.height = m_height;
            normal.rowStrideInBytes = m_width * sizeof(float4);
            normal.pixelStrideInBytes = sizeof(float4);
            normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;

            OptixDenoiserGuideLayer guideLayer = {};
            guideLayer.albedo = albedo;
            guideLayer.normal = normal;

            OptixDenoiserLayer denoiseLayer = {};
            denoiseLayer.input = input;
            denoiseLayer.output = output;

            OptixDenoiserParams denoiseParams = {};
            denoiseParams.blendFactor = 0.0f;
            denoiseParams.hdrIntensity = m_state->d_intensity;

            OPTIX_CHECK(optixDenoiserComputeIntensity(
                m_state->denoiser, m_state->stream,
                &input, m_state->d_intensity,
                m_state->d_denoiserScratch, m_state->denoiserScratchSize));

            OPTIX_CHECK(optixDenoiserInvoke(
                m_state->denoiser, m_state->stream,
                &denoiseParams,
                m_state->d_denoiserState, m_state->denoiserStateSize,
                &guideLayer, &denoiseLayer, 1,
                0, 0,
                m_state->d_denoiserScratch, m_state->denoiserScratchSize));

            CUDA_CHECK(cudaMemcpyAsync(d_output,
                                       reinterpret_cast<void*>(m_state->d_denoised),
                                       imageSize, cudaMemcpyDeviceToDevice,
                                       m_state->stream));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(d_output,
                                       reinterpret_cast<void*>(m_state->d_beauty),
                                       imageSize, cudaMemcpyDeviceToDevice,
                                       m_state->stream));
        }

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_state->pboResource, m_state->stream));
        CUDA_CHECK(cudaStreamSynchronize(m_state->stream));
        return;
    }
    else
    {
        static bool loggedFallback = false;
        if (!loggedFallback)
        {
            logOptixLine("[OptiX] Fallback to CPU renderer (OptiX not ready).");
            loggedFallback = true;
        }
    }
#endif

    // CPU fallback
    renderCPU(camera, glPBO, spp, maxBounces);
}

// ---------------------------------------------------------------------------
// CPU fallback – simple gradient + sky
// ---------------------------------------------------------------------------

void OptixRenderer::renderCPU(const Camera& camera,
                               unsigned int  glPBO,
                               int           spp,
                               int           /*maxBounces*/)
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
            for (int s = 0; s < spp; ++s)
            {
                const float u = (x + dist(rng)) / static_cast<float>(m_width);
                const float v = (y + dist(rng)) / static_cast<float>(m_height);

                const glm::vec3 dir = glm::normalize(
                    llc + u * hor + v * ver - ori);

                // Simple sky gradient
                const float t = 0.5f * (dir.y + 1.f);
                const glm::vec3 sky = glm::mix(
                    glm::vec3(1.f, 1.f, 1.f),
                    glm::vec3(0.5f, 0.7f, 1.f), t);
                color += sky;
            }
            color /= static_cast<float>(spp);

            const size_t idx = static_cast<size_t>((y * m_width + x)) * 4;
            pixels[idx + 0] = color.r;
            pixels[idx + 1] = color.g;
            pixels[idx + 2] = color.b;
            pixels[idx + 3] = 1.f;
        }
    }

    m_accumulatedSamples += static_cast<uint32_t>(spp);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glPBO);
    void* ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (ptr)
    {
        std::memcpy(ptr, pixels.data(), pixels.size() * sizeof(float));
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
