#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "stb_image.h"

struct TextureData
{
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    bool ownsMemory = true;

    bool valid() const { return pixels != nullptr && width > 0 && height > 0; }

    void cleanup()
    {
        if (ownsMemory && pixels)
            stbi_image_free(pixels);
        pixels = nullptr;
        width = height = channels = 0;
    }
};

struct HDRTextureData
{
    std::vector<float> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;

    bool valid() const { return !pixels.empty() && width > 0 && height > 0 && channels > 0; }
};

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
};

enum class MaterialType
{
    BasicPBR = 0,
    Diffuse = 1,
    Conductor = 2,
    Dielectric = 3,
    RoughDielectric = 4,
    CoatedDiffuse = 5,
    CoatedConductor = 6,
    Subsurface = 7
};

struct Material
{
    std::string name;
    MaterialType type = MaterialType::BasicPBR;
    glm::vec3 baseColor = {0.8f, 0.8f, 0.8f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float ior = 1.45f;
    float transmission = 0.0f;
    glm::vec3 emission = {0.f, 0.f, 0.f};
    float sigma = 0.0f;
    glm::vec3 eta = {0.f, 0.f, 0.f};
    glm::vec3 k = {0.f, 0.f, 0.f};
    float uroughness = 0.1f;
    float vroughness = 0.1f;
    bool remapRoughness = true;
    std::string baseColorTexPath;
    std::string normalTexPath;
    std::string metallicRoughnessTexPath;
    std::string emissionTexPath;
    std::string transmissionTexPath;
    TextureData baseColorTexData;
    TextureData normalTexData;
    TextureData metallicRoughnessTexData;
    TextureData emissionTexData;
    TextureData transmissionTexData;
};

enum class LightType
{
    Point,
    Directional,
    Spot
};

struct Light
{
    LightType type;
    glm::vec3 position = {0.f, 0.f, 0.f};
    glm::vec3 direction = {0.f, -1.f, 0.f};
    glm::vec3 intensity = {1.f, 1.f, 1.f};
    float cosInner = 0.f;
    float cosOuter = 0.f;
};

enum class CurveType
{
    Flat,
    Cylinder,
    Ribbon
};

struct CurveSegment
{
    glm::vec3 cp[4];
    float width0 = 0.0f;
    float width1 = 0.0f;
    CurveType type = CurveType::Flat;
    glm::vec3 n0 = {0.f, 1.f, 0.f};
    glm::vec3 n1 = {0.f, 1.f, 0.f};
    bool hasNormals = false;
    uint32_t materialIndex = 0;
};

struct EnvironmentMap
{
    TextureData texData;
    std::vector<float> hdrPixels;
    int hdrWidth = 0;
    int hdrHeight = 0;
    int hdrChannels = 0;
    bool isHDR = false;
    glm::vec3 scale = {1.f, 1.f, 1.f};
    glm::mat3 lightTransform = glm::mat3(1.f);
    bool valid = false;
};

struct CameraInfo
{
    glm::vec3 eye = {0.f, 0.f, -5.f};
    glm::vec3 target = {0.f, 0.f, 0.f};
    glm::vec3 up = {0.f, 1.f, 0.f};
    float fovy = 45.0f;
    float zNear = 0.1f;
    float zFar = 1000.0f;
    bool valid = false;
};

struct Mesh
{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t materialIndex = 0;
};

struct SceneStats
{
    uint32_t meshCount = 0;
    uint32_t triangleCount = 0;
    uint32_t materialCount = 0;
};

class Scene
{
public:
    Scene() = default;

    void clear();
    void addMesh(const Mesh& mesh);
    void addMaterial(const Material& material);
    void addLight(const Light& light);
    void addCurve(const CurveSegment& curve);
    void setCamera(const CameraInfo& camera);
    void setAmbientIntensity(const glm::vec3& ambient);
    void setEnvironmentMap(const TextureData& texData, const glm::vec3& scale, const glm::mat3& lightTransform = glm::mat3(1.f));
    void setEnvironmentMapHDR(const HDRTextureData& texData, const glm::vec3& scale, const glm::mat3& lightTransform = glm::mat3(1.f));
    void setLoadedPath(const std::string& path) { m_path = path; }
    void rebuildStats();

    bool empty() const { return m_meshes.empty() && m_curves.empty(); }

    std::vector<Mesh>& meshes() { return m_meshes; }
    const std::vector<Mesh>& meshes() const { return m_meshes; }
    std::vector<Material>& materials() { return m_materials; }
    const std::vector<Material>& materials() const { return m_materials; }
    std::vector<Light>& lights() { return m_lights; }
    const std::vector<Light>& lights() const { return m_lights; }
    std::vector<CurveSegment>& curves() { return m_curves; }
    const std::vector<CurveSegment>& curves() const { return m_curves; }
    const glm::vec3& ambientInt() const { return m_ambientInt; }
    const SceneStats& stats() const { return m_stats; }
    EnvironmentMap& environment() { return m_environment; }
    const EnvironmentMap& environment() const { return m_environment; }
    const std::string& loadedPath() const { return m_path; }
    CameraInfo& camera() { return m_camera; }
    const CameraInfo& camera() const { return m_camera; }

private:
    std::vector<Mesh> m_meshes;
    std::vector<Material> m_materials;
    std::vector<Light> m_lights;
    std::vector<CurveSegment> m_curves;
    glm::vec3 m_ambientInt = {0.05f, 0.05f, 0.05f};
    CameraInfo m_camera;
    EnvironmentMap m_environment;
    SceneStats m_stats;
    std::string m_path;
};
