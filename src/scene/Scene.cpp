#include "Scene.h"

void Scene::clear()
{
    m_environment.texData.cleanup();
    m_meshes.clear();
    m_materials.clear();
    m_lights.clear();
    m_curves.clear();
    m_ambientInt = glm::vec3(0.05f, 0.05f, 0.05f);
    m_camera = {};
    m_environment = {};
    m_stats = {};
    m_path.clear();
}

void Scene::addMesh(const Mesh& mesh)
{
    m_meshes.push_back(mesh);
}

void Scene::addMaterial(const Material& material)
{
    m_materials.push_back(material);
}

void Scene::addLight(const Light& light)
{
    m_lights.push_back(light);
}

void Scene::addCurve(const CurveSegment& curve)
{
    m_curves.push_back(curve);
}

void Scene::setCamera(const CameraInfo& camera)
{
    m_camera = camera;
}

void Scene::setAmbientIntensity(const glm::vec3& ambient)
{
    m_ambientInt = ambient;
}

void Scene::setEnvironmentMap(const TextureData& texData, const glm::vec3& scale, const glm::mat3& lightTransform)
{
    m_environment.texData.cleanup();
    m_environment.texData = texData;
    m_environment.hdrPixels.clear();
    m_environment.hdrWidth = 0;
    m_environment.hdrHeight = 0;
    m_environment.hdrChannels = 0;
    m_environment.isHDR = false;
    m_environment.scale = scale;
    m_environment.lightTransform = lightTransform;
    m_environment.valid = texData.pixels != nullptr && texData.width > 0 && texData.height > 0;
}

void Scene::setEnvironmentMapHDR(const HDRTextureData& texData, const glm::vec3& scale, const glm::mat3& lightTransform)
{
    m_environment.texData.cleanup();
    m_environment.hdrPixels = texData.pixels;
    m_environment.hdrWidth = texData.width;
    m_environment.hdrHeight = texData.height;
    m_environment.hdrChannels = texData.channels;
    m_environment.isHDR = texData.valid();
    m_environment.scale = scale;
    m_environment.lightTransform = lightTransform;
    m_environment.valid = m_environment.isHDR;
}

void Scene::rebuildStats()
{
    m_stats = {};
    m_stats.meshCount = static_cast<uint32_t>(m_meshes.size());
    m_stats.materialCount = static_cast<uint32_t>(m_materials.size());
    for (const auto& mesh : m_meshes)
        m_stats.triangleCount += static_cast<uint32_t>(mesh.indices.size() / 3);
}
