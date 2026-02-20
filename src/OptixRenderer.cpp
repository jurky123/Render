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

// ---------------------------------------------------------------------------
// OptiX/CUDA includes – conditionally compiled
// ---------------------------------------------------------------------------
#ifdef PATHTRACER_OPTIX_ENABLED
#  include <cuda_runtime.h>
#  include <cuda_gl_interop.h>
#  include <optix.h>
#  include <optix_function_table_definition.h>
#  include <optix_stubs.h>

#  define CUDA_CHECK(call)                                                  \
    do {                                                                    \
        cudaError_t rc = (call);                                            \
        if (rc != cudaSuccess) {                                            \
            std::cerr << "[CUDA] " << cudaGetErrorString(rc)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            m_optixReady = false;                                           \
        }                                                                   \
    } while(0)

#  define OPTIX_CHECK(call)                                                 \
    do {                                                                    \
        OptixResult rc = (call);                                            \
        if (rc != OPTIX_SUCCESS) {                                          \
            std::cerr << "[OptiX] " << optixGetErrorString(rc)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
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
    OptixShaderBindingTable sbt          = {};
    OptixTraversableHandle gasHandle     = 0;

    CUstream              stream         = nullptr;
    CUdeviceptr           d_sbt          = 0;
    CUdeviceptr           d_gas          = 0;
    CUdeviceptr           d_launchParams = 0;
    CUdeviceptr           d_accum        = 0;
    cudaGraphicsResource* pboResource    = nullptr;
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
        std::cerr << "[OptiX] optixInit failed: " << optixGetErrorString(res)
                  << " – falling back to CPU renderer.\n";
        return;
    }

    // Create OptiX device context
    CUcontext cuCtx = nullptr; // use current context
    OptixDeviceContextOptions opts = {};
    opts.logCallbackFunction = [](unsigned int level, const char* tag,
                                  const char* message, void*)
    {
        if (level <= 3)
            std::cerr << "[OptiX][" << tag << "] " << message << "\n";
    };
    opts.logCallbackLevel = 4;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &opts, &m_state->context));
    CUDA_CHECK(cudaStreamCreate(&m_state->stream));

    // Allocate accumulation buffer
    const size_t bufSize = static_cast<size_t>(m_width * m_height) * 4 * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state->d_accum), bufSize));

    buildPipeline();
    m_optixReady = true;
    std::cout << "[OptiX] Initialised successfully.\n";
#else
    std::cout << "[OptiX] Not compiled – using CPU fallback renderer.\n";
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
    if (m_state->d_launchParams)
        cudaFree(reinterpret_cast<void*>(m_state->d_launchParams));
    if (m_state->d_gas)
        cudaFree(reinterpret_cast<void*>(m_state->d_gas));
    if (m_state->d_sbt)
        cudaFree(reinterpret_cast<void*>(m_state->d_sbt));
    if (m_state->hitgroupPG)
        optixProgramGroupDestroy(m_state->hitgroupPG);
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
// buildPipeline – loads PTX and creates OptiX pipeline
// ---------------------------------------------------------------------------

void OptixRenderer::buildPipeline()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (!m_state->context) return;

    // --- Compile module from embedded PTX ---
    OptixModuleCompileOptions modOpts = {};
    modOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    modOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    modOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur            = 0;
    pipeOpts.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues          = 3;
    pipeOpts.numAttributeValues        = 2;
    pipeOpts.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "params";

    // Load PTX file (written by CMake post-build step)
#ifndef PTX_DIR
#  define PTX_DIR "."
#endif
    const std::string ptxPath = std::string(PTX_DIR) + "/path_tracer.ptx";
    FILE* f = fopen(ptxPath.c_str(), "rb");
    if (!f)
    {
        std::cerr << "[OptiX] Cannot open PTX: " << ptxPath
                  << " – falling back to CPU renderer.\n";
        m_optixReady = false;
        return;
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> ptx(static_cast<size_t>(sz) + 1, '\0');
    fread(ptx.data(), 1, static_cast<size_t>(sz), f);
    fclose(f);

    char log[4096]; size_t logSz = sizeof(log);
    OptixResult r = optixModuleCreateFromPTX(
        m_state->context, &modOpts, &pipeOpts,
        ptx.data(), ptx.size(),
        log, &logSz, &m_state->module);
    if (r != OPTIX_SUCCESS)
    {
        std::cerr << "[OptiX] Module create failed: " << log << "\n";
        m_optixReady = false;
        return;
    }

    // --- Program groups ---
    OptixProgramGroupOptions pgOpts = {};

    // Raygen
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = m_state->module;
        desc.raygen.entryFunctionName = "__raygen__pathTrace";
        OPTIX_CHECK(optixProgramGroupCreate(m_state->context, &desc, 1,
                                            &pgOpts, log, &logSz,
                                            &m_state->raygenPG));
    }
    // Miss
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module           = m_state->module;
        desc.miss.entryFunctionName= "__miss__envmap";
        OPTIX_CHECK(optixProgramGroupCreate(m_state->context, &desc, 1,
                                            &pgOpts, log, &logSz,
                                            &m_state->missPG));
    }
    // Hitgroup (closest hit + any hit)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH             = m_state->module;
        desc.hitgroup.entryFunctionNameCH  = "__closesthit__pbr";
        OPTIX_CHECK(optixProgramGroupCreate(m_state->context, &desc, 1,
                                            &pgOpts, log, &logSz,
                                            &m_state->hitgroupPG));
    }

    // --- Pipeline ---
    OptixProgramGroup pgs[] = {
        m_state->raygenPG, m_state->missPG, m_state->hitgroupPG
    };
    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 8;
    linkOpts.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OPTIX_CHECK(optixPipelineCreate(m_state->context, &pipeOpts, &linkOpts,
                                    pgs, 3, log, &logSz, &m_state->pipeline));

    OptixStackSizes ss = {};
    for (auto& pg : pgs)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &ss));
    unsigned int directStackSize, continuationStackSize;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &ss, linkOpts.maxTraceDepth, 0, 0,
        &directStackSize, &continuationStackSize, nullptr));
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_state->pipeline,
        directStackSize, directStackSize,
        continuationStackSize, 1));
#endif
}

// ---------------------------------------------------------------------------
// buildSBT
// ---------------------------------------------------------------------------

void OptixRenderer::buildSBT()
{
#ifdef PATHTRACER_OPTIX_ENABLED
    if (!m_optixReady) return;

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
    OPTIX_CHECK(optixSbtRecordPackHeader(m_state->raygenPG, &rg));

    MissRecord ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_state->missPG, &ms));

    // Build one hit record per mesh
    std::vector<HitRecord> hitRecords;
    if (m_scene)
    {
        const auto& meshes    = m_scene->meshes();
        const auto& materials = m_scene->materials();

        hitRecords.resize(meshes.size());
        for (size_t i = 0; i < meshes.size(); ++i)
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_state->hitgroupPG, &hitRecords[i]));
            const auto& mat = materials[meshes[i].materialIndex];

            hitRecords[i].data.baseColor    = make_float3(mat.baseColor.r,
                                                          mat.baseColor.g,
                                                          mat.baseColor.b);
            hitRecords[i].data.metallic     = mat.metallic;
            hitRecords[i].data.roughness    = mat.roughness;
            hitRecords[i].data.emission     = make_float3(mat.emission.r,
                                                          mat.emission.g,
                                                          mat.emission.b);
            hitRecords[i].data.ior          = mat.ior;
            hitRecords[i].data.transmission = mat.transmission;
        }
    }
    else
    {
        // Default hit record for empty scene
        hitRecords.resize(1);
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state->hitgroupPG, &hitRecords[0]));
        hitRecords[0].data.baseColor    = make_float3(0.8f, 0.8f, 0.8f);
        hitRecords[0].data.metallic     = 0.f;
        hitRecords[0].data.roughness    = 0.5f;
        hitRecords[0].data.emission     = make_float3(0.f, 0.f, 0.f);
        hitRecords[0].data.ior          = 1.45f;
        hitRecords[0].data.transmission = 0.f;
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
    if (!m_optixReady || !m_scene || m_scene->empty()) return;

    // Upload all meshes as a single GAS with one build input per mesh
    const auto& meshes = m_scene->meshes();

    std::vector<CUdeviceptr>       d_vertices(meshes.size());
    std::vector<CUdeviceptr>       d_indices (meshes.size());
    std::vector<OptixBuildInput>   buildInputs(meshes.size());
    std::vector<uint32_t>          flags(meshes.size(),
                                         OPTIX_GEOMETRY_FLAG_NONE);

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const auto& m = meshes[i];

        // Vertices (only xyz)
        std::vector<float3> verts(m.vertices.size());
        for (size_t j = 0; j < m.vertices.size(); ++j)
            verts[j] = {m.vertices[j].position.x,
                        m.vertices[j].position.y,
                        m.vertices[j].position.z};

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
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));

    // Free old GAS
    if (m_state->d_gas)
        cudaFree(reinterpret_cast<void*>(m_state->d_gas));
    m_state->d_gas = d_output;

    // Free temporary vertex/index buffers
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        cudaFree(reinterpret_cast<void*>(d_vertices[i]));
        cudaFree(reinterpret_cast<void*>(d_indices[i]));
    }
#endif
}

// ---------------------------------------------------------------------------
// setScene / resize / resetAccumulation
// ---------------------------------------------------------------------------

void OptixRenderer::setScene(const Scene* scene)
{
    m_scene = scene;
    buildAccel();
    buildSBT();
    resetAccumulation();
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
        lp.output              = d_output;
        lp.accum               = reinterpret_cast<float4*>(m_state->d_accum);
        lp.width               = m_width;
        lp.height              = m_height;
        lp.sampleIndex         = m_accumulatedSamples;
        lp.maxBounces          = maxBounces;
        lp.traversable         = m_state->gasHandle;

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

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_state->pboResource, m_state->stream));
        CUDA_CHECK(cudaStreamSynchronize(m_state->stream));
        return;
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
