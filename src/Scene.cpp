#include "Scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/material.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <map>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static glm::mat4 toGlm(const aiMatrix4x4& m)
{
    // aiMatrix4x4 is row-major; glm is column-major
    return glm::transpose(glm::make_mat4(&m.a1));
}

static std::string texturePath(const aiMaterial* mat, aiTextureType type,
                                const std::string& dir)
{
    if (mat->GetTextureCount(type) == 0)
        return {};
    aiString p;
    mat->GetTexture(type, 0, &p);
    const fs::path full = fs::path(dir) / p.C_Str();
    return full.string();
}

// ---------------------------------------------------------------------------
// Scene::load
// ---------------------------------------------------------------------------

bool Scene::load(const std::string& path)
{
    clear();
    return loadAppend(path);
}

bool Scene::loadAppend(const std::string& path)
{
    Assimp::Importer importer;
    const unsigned int flags =
        aiProcess_Triangulate          |
        aiProcess_GenSmoothNormals     |
        aiProcess_FlipUVs              |
        aiProcess_CalcTangentSpace     |
        aiProcess_JoinIdenticalVertices|
        aiProcess_OptimizeMeshes;

    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode)
    {
        std::cerr << "[Scene] Assimp error: " << importer.GetErrorString() << "\n";
        return false;
    }

    m_path = path;
    const std::string dir = fs::path(path).parent_path().string();

    loadMaterials(scene, dir);
    processNode(scene, scene->mRootNode, glm::mat4(1.f));

    m_stats.meshCount     = static_cast<uint32_t>(m_meshes.size());
    m_stats.materialCount = static_cast<uint32_t>(m_materials.size());
    for (const auto& mesh : m_meshes)
        m_stats.triangleCount +=
            static_cast<uint32_t>(mesh.indices.size() / 3);

    std::cout << "[Scene] Loaded " << m_stats.meshCount     << " meshes, "
              << m_stats.triangleCount << " triangles, "
              << m_stats.materialCount << " materials.\n";
    return true;
}

void Scene::clear()
{
    m_meshes.clear();
    m_materials.clear();
    m_lights.clear();
    m_curves.clear();
    m_ambientInt = glm::vec3(0.05f, 0.05f, 0.05f);
    m_camera = {};
    m_environment = {};
    m_stats  = {};
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
    m_environment.texData = texData;
    m_environment.hdrPixels.clear();
    m_environment.hdrWidth = 0;
    m_environment.hdrHeight = 0;
    m_environment.hdrChannels = 0;
    m_environment.isHDR = false;
    m_environment.scale = scale;
    m_environment.lightTransform = lightTransform;
    m_environment.valid = (texData.pixels != nullptr && texData.width > 0 && texData.height > 0);
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

// ---------------------------------------------------------------------------
// Scene::processNode
// ---------------------------------------------------------------------------

void Scene::processNode(const aiScene* scene, const aiNode* node,
                        const glm::mat4& parentTransform)
{
    const glm::mat4 transform = parentTransform * toGlm(node->mTransformation);

    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        m_meshes.push_back(processMesh(scene, mesh, transform));
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        processNode(scene, node->mChildren[i], transform);
}

// ---------------------------------------------------------------------------
// Scene::processMesh
// ---------------------------------------------------------------------------

Mesh Scene::processMesh(const aiScene* /*scene*/, const aiMesh* aiM,
                        const glm::mat4& transform)
{
    Mesh mesh;
    mesh.name          = aiM->mName.C_Str();
    mesh.materialIndex = aiM->mMaterialIndex;

    const glm::mat3 normalMat =
        glm::mat3(glm::transpose(glm::inverse(transform)));

    mesh.vertices.reserve(aiM->mNumVertices);
    for (unsigned int i = 0; i < aiM->mNumVertices; ++i)
    {
        Vertex v;
        const aiVector3D& p = aiM->mVertices[i];
        v.position = glm::vec3(transform * glm::vec4(p.x, p.y, p.z, 1.f));

        if (aiM->HasNormals())
        {
            const aiVector3D& n = aiM->mNormals[i];
            v.normal = glm::normalize(normalMat * glm::vec3(n.x, n.y, n.z));
        }

        if (aiM->HasTextureCoords(0))
        {
            v.texCoord = glm::vec2(aiM->mTextureCoords[0][i].x,
                                   aiM->mTextureCoords[0][i].y);
        }

        if (aiM->HasTangentsAndBitangents())
        {
            const aiVector3D& t = aiM->mTangents[i];
            v.tangent = glm::normalize(normalMat * glm::vec3(t.x, t.y, t.z));
        }

        mesh.vertices.push_back(v);
    }

    mesh.indices.reserve(static_cast<size_t>(aiM->mNumFaces) * 3);
    for (unsigned int i = 0; i < aiM->mNumFaces; ++i)
    {
        const aiFace& face = aiM->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; ++j)
            mesh.indices.push_back(face.mIndices[j]);
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// Scene::loadMaterials
// ---------------------------------------------------------------------------

void Scene::loadMaterials(const aiScene* scene, const std::string& directory)
{
    m_materials.reserve(scene->mNumMaterials);
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
        m_materials.push_back(std::move(convertMaterial(scene->mMaterials[i], directory)));
}

Material Scene::convertMaterial(const aiMaterial* aiMat,
                                 const std::string& directory)
{
    Material mat;

    aiString name;
    if (aiMat->Get(AI_MATKEY_NAME, name) == AI_SUCCESS)
        mat.name = name.C_Str();

    // Base color / diffuse
    aiColor4D color;
    if (aiMat->Get(AI_MATKEY_BASE_COLOR, color) == AI_SUCCESS ||
        aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
    {
        mat.baseColor = glm::vec3(color.r, color.g, color.b);
    }

    // Metallic
    float metallic = 0.f;
    if (aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
        mat.metallic = metallic;

    // Roughness
    float roughness = 0.5f;
    if (aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
        mat.roughness = roughness;

    // Emission
    aiColor3D emission;
    if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emission) == AI_SUCCESS)
        mat.emission = glm::vec3(emission.r, emission.g, emission.b);

    // IOR
    float ior = 1.45f;
    if (aiMat->Get(AI_MATKEY_REFRACTI, ior) == AI_SUCCESS)
        mat.ior = ior;

    // Transmission
    float transmission = 0.f;
    if (aiMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission) == AI_SUCCESS)
        mat.transmission = transmission;

    // Texture paths
    mat.baseColorTexPath           = texturePath(aiMat, aiTextureType_BASE_COLOR,        directory);
    if (mat.baseColorTexPath.empty())
        mat.baseColorTexPath       = texturePath(aiMat, aiTextureType_DIFFUSE,           directory);
    mat.normalTexPath              = texturePath(aiMat, aiTextureType_NORMALS,           directory);
    mat.metallicRoughnessTexPath   = texturePath(aiMat, aiTextureType_METALNESS,         directory);
    mat.emissionTexPath            = texturePath(aiMat, aiTextureType_EMISSIVE,          directory);

    // Load texture pixel data if texture path exists
    // 注意：如果纹理已经加载(例如通过PBRT加载器)，则跳过重新加载
    if (!mat.baseColorTexPath.empty() && !mat.baseColorTexData.pixels)
    {
        std::cout << "[Scene] Loading baseColor texture for material '" << mat.name 
                  << "': " << mat.baseColorTexPath << "\n";
        mat.baseColorTexData = loadTexture(mat.baseColorTexPath);
        std::cout << "[Scene] Loaded texture: width=" << mat.baseColorTexData.width 
                  << " height=" << mat.baseColorTexData.height 
                  << " channels=" << mat.baseColorTexData.channels << "\n";
    }
    else if (mat.baseColorTexData.pixels)
    {
        std::cout << "[Scene] baseColor texture already loaded for material '" << mat.name 
                  << "' (" << mat.baseColorTexData.width << "x" << mat.baseColorTexData.height << ")\n";
    }

    // Load normal map
    if (!mat.normalTexPath.empty() && !mat.normalTexData.pixels)
    {
        std::cout << "[Scene] Loading normal texture for material '" << mat.name 
                  << "': " << mat.normalTexPath << "\n";
        mat.normalTexData = loadTexture(mat.normalTexPath);
        std::cout << "[Scene] Loaded normal: width=" << mat.normalTexData.width 
                  << " height=" << mat.normalTexData.height << "\n";
    }

    // Load metallic/roughness map
    if (!mat.metallicRoughnessTexPath.empty() && !mat.metallicRoughnessTexData.pixels)
    {
        std::cout << "[Scene] Loading metallicRoughness texture for material '" << mat.name 
                  << "': " << mat.metallicRoughnessTexPath << "\n";
        mat.metallicRoughnessTexData = loadTexture(mat.metallicRoughnessTexPath);
        std::cout << "[Scene] Loaded metallicRoughness: width=" << mat.metallicRoughnessTexData.width 
                  << " height=" << mat.metallicRoughnessTexData.height << "\n";
    }

    // Load emission map
    if (!mat.emissionTexPath.empty() && !mat.emissionTexData.pixels)
    {
        std::cout << "[Scene] Loading emission texture for material '" << mat.name 
                  << "': " << mat.emissionTexPath << "\n";
        mat.emissionTexData = loadTexture(mat.emissionTexPath);
        std::cout << "[Scene] Loaded emission: width=" << mat.emissionTexData.width 
                  << " height=" << mat.emissionTexData.height << "\n";
    }

    // Load transmission map
    if (!mat.transmissionTexPath.empty() && !mat.transmissionTexData.pixels)
    {
        std::cout << "[Scene] Loading transmission texture for material '" << mat.name 
                  << "': " << mat.transmissionTexPath << "\n";
        mat.transmissionTexData = loadTexture(mat.transmissionTexPath);
        std::cout << "[Scene] Loaded transmission: width=" << mat.transmissionTexData.width 
                  << " height=" << mat.transmissionTexData.height << "\n";
    }

    // Fallback: if baseColor is too dark but a texture exists, use a reasonable default
    // This helps when materials rely on textures that aren't yet loaded
    if (!mat.baseColorTexPath.empty())
    {
        float brightness = mat.baseColor.r + mat.baseColor.g + mat.baseColor.b;
        if (brightness < 0.1f)
        {
            mat.baseColor = glm::vec3(0.7f, 0.7f, 0.7f); // Light gray fallback
            // Log this for debugging
            // std::cout << "[Scene] Material '" << mat.name << "' has texture but dark baseColor, using fallback\\n";
        }
    }

    return mat;
}

// ---------------------------------------------------------------------------
// Simple inline YAML parsing helpers
// ---------------------------------------------------------------------------

static std::string trim(const std::string& str)
{
    if (str.empty()) return str;
    
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start])))
        ++start;
    
    if (start == str.size()) return "";
    
    size_t end = str.size() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end])))
        --end;
    
    return str.substr(start, end - start + 1);
}

static std::vector<std::string> parseYamlList(const std::string& line)
{
    std::vector<std::string> result;
    size_t start = line.find('[');
    size_t end = line.rfind(']');
    if (start == std::string::npos || end == std::string::npos)
        return result;
    
    std::string content = line.substr(start + 1, end - start - 1);
    std::istringstream iss(content);
    std::string item;
    while (std::getline(iss, item, ','))
        result.push_back(trim(item));
    
    return result;
}

// ---------------------------------------------------------------------------
// Scene::loadFromYaml
// ---------------------------------------------------------------------------

bool Scene::loadFromYaml(const std::string& yamlPath)
{
    clear();
    
    try {
        YAML::Node config = YAML::LoadFile(yamlPath);
        const std::string dir = fs::path(yamlPath).parent_path().string();
        
        // Parse Camera (use first camera if multiple)
        if (config["Cameras"] && config["Cameras"].IsSequence() && config["Cameras"].size() > 0) {
            const auto& camNode = config["Cameras"][0];
            if (camNode["Eye"]) {
                auto eye = camNode["Eye"].as<std::vector<float>>();
                if (eye.size() >= 3) {
                    m_camera.eye = glm::vec3(eye[0], eye[1], eye[2]);
                }
            }
            if (camNode["Target"]) {
                auto target = camNode["Target"].as<std::vector<float>>();
                if (target.size() >= 3) {
                    m_camera.target = glm::vec3(target[0], target[1], target[2]);
                }
            }
            if (camNode["Up"]) {
                auto up = camNode["Up"].as<std::vector<float>>();
                if (up.size() >= 3) {
                    m_camera.up = glm::vec3(up[0], up[1], up[2]);
                }
            }
            if (camNode["Fovy"]) {
                m_camera.fovy = camNode["Fovy"].as<float>();
            }
            if (camNode["ZNear"]) {
                m_camera.zNear = camNode["ZNear"].as<float>();
            }
            if (camNode["ZFar"]) {
                m_camera.zFar = camNode["ZFar"].as<float>();
            }
            m_camera.valid = true;
            std::cout << "[Scene] Camera loaded: Eye=(" << m_camera.eye.x << ", " << m_camera.eye.y << ", " << m_camera.eye.z << ")";
            std::cout << " FOV=" << m_camera.fovy << "\n";
        }
        
        // Parse AmbientIntensity
        if (config["AmbientIntensity"]) {
            auto ambient = config["AmbientIntensity"].as<std::vector<float>>();
            if (ambient.size() >= 3) {
                m_ambientInt = glm::vec3(ambient[0], ambient[1], ambient[2]);
            }
        }
        
        // Parse Lights
        if (config["Lights"]) {
            for (const auto& lightNode : config["Lights"]) {
                Light light;
                std::string type = lightNode["Type"].as<std::string>();
                
                if (type == "Directional") {
                    light.type = LightType::Directional;
                    auto dir = lightNode["Direction"].as<std::vector<float>>();
                    if (dir.size() >= 3) {
                        light.direction = glm::normalize(glm::vec3(dir[0], dir[1], dir[2]));
                    }
                } else if (type == "Point") {
                    light.type = LightType::Point;
                    auto pos = lightNode["Position"].as<std::vector<float>>();
                    if (pos.size() >= 3) {
                        light.position = glm::vec3(pos[0], pos[1], pos[2]);
                    }
                }
                
                auto intensity = lightNode["Intensity"].as<std::vector<float>>();
                if (intensity.size() >= 3) {
                    light.intensity = glm::vec3(intensity[0], intensity[1], intensity[2]);
                }
                
                m_lights.push_back(light);
            }
        }
        
        // Parse Materials first (so we have indices ready)
        std::map<std::string, uint32_t> materialMap;  // name -> index
        if (config["Materials"]) {
            for (const auto& matNode : config["Materials"]) {
                std::string name = matNode["Name"].as<std::string>();
                std::vector<std::string> matData;
                
                if (matNode["Diffuse"]) {
                    matData.push_back("Diffuse");
                    std::stringstream ss;
                    auto diffuse = matNode["Diffuse"].as<std::vector<float>>();
                    ss << "[" << diffuse[0] << ", " << diffuse[1] << ", " << diffuse[2] << "]";
                    matData.push_back(ss.str());
                }
                
                if (matNode["Specular"]) {
                    matData.push_back("Specular");
                    std::stringstream ss;
                    auto specular = matNode["Specular"].as<std::vector<float>>();
                    ss << "[";
                    for (size_t i = 0; i < specular.size(); ++i) {
                        if (i > 0) ss << ", ";
                        ss << specular[i];
                    }
                    ss << "]";
                    matData.push_back(ss.str());
                }
                
                uint32_t matIdx = loadYamlMaterial(name, matData, dir);
                materialMap[name] = matIdx;
            }
        }
        
        // Parse Models section and apply material mappings
        if (config["Models"]) {
            for (const auto& modelNode : config["Models"]) {
                std::string meshFile = modelNode["Mesh"].as<std::string>();
                const fs::path meshPath = fs::path(dir) / meshFile;
                if (fs::exists(meshPath)) {
                    size_t meshCountBefore = m_meshes.size();
                    loadAppend(meshPath.string());  // Use loadAppend instead of load
                    size_t meshCountAfter = m_meshes.size();
                    
                    // Apply material to newly loaded meshes
                    if (modelNode["Material"]) {
                        std::string matName = modelNode["Material"].as<std::string>();
                        auto it = materialMap.find(matName);
                        if (it != materialMap.end()) {
                            uint32_t matIdx = it->second;
                            for (size_t i = meshCountBefore; i < meshCountAfter; ++i) {
                                m_meshes[i].materialIndex = matIdx;
                            }
                            std::cout << "[Scene] Applied material '" << matName << "' (idx " << matIdx << ") to " << (meshCountAfter - meshCountBefore) << " meshes from " << meshFile << "\n";
                        }
                    }
                }
            }
        }
        
        // Parse ComplexModels section
        if (config["ComplexModels"]) {
            for (const auto& modelNode : config["ComplexModels"]) {
                std::string meshFile = modelNode["Mesh"].as<std::string>();
                const fs::path meshPath = fs::path(dir) / meshFile;
                if (fs::exists(meshPath)) {
                    loadAppend(meshPath.string());  // Use loadAppend instead of load
                }
            }
        }
        
        std::cout << "[Scene] Loaded from YAML: " << m_lights.size() << " lights, "
                  << m_meshes.size() << " meshes, " << m_materials.size() << " materials\n";
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "[Scene] YAML parsing error: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Scene] Error loading YAML: " << e.what() << "\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// Scene::loadYamlMaterial
// ---------------------------------------------------------------------------

uint32_t Scene::loadYamlMaterial(const std::string& name,
                                 const std::vector<std::string>& matData,
                                 const std::string& sceneDir)
{
    Material mat;
    mat.name = name;

    // matData is [key1, value1, key2, value2, ...] format
    for (size_t i = 0; i + 1 < matData.size(); i += 2)
    {
        const std::string& key = matData[i];
        const std::string& value = matData[i + 1];

        // Parse Diffuse/BaseColor color (format: [r, g, b])
        if (key == "Diffuse" || key == "BaseColor")
        {
            // Handle both "[.4, .4, .4]" and ".4, .4, .4" formats
            std::string listStr = value;
            if (value.find('[') == std::string::npos)
                listStr = "[" + value + "]";
            std::vector<std::string> values = parseYamlList(listStr);
            if (values.size() >= 3)
            {
                try
                {
                    mat.baseColor = glm::vec3(std::stof(values[0]),
                                              std::stof(values[1]),
                                              std::stof(values[2]));
                }
                catch (...) { }
            }
        }
        // Parse Metallic
        else if (key == "Metallic")
        {
            try { mat.metallic = std::stof(value); }
            catch (...) { }
        }
        // Parse Roughness
        else if (key == "Roughness")
        {
            try { mat.roughness = std::stof(value); }
            catch (...) { }
        }
        // Parse Specular (format: [r, g, b, shininess])
        else if (key == "Specular")
        {
            std::string listStr = value;
            if (value.find('[') == std::string::npos)
                listStr = "[" + value + "]";
            std::vector<std::string> values = parseYamlList(listStr);
            if (values.size() >= 4)
            {
                try
                {
                    float shininess = std::stof(values[3]);
                    mat.roughness = 1.0f / (shininess + 1.0f);
                }
                catch (...) { }
            }
        }
        // Parse Emission
        else if (key == "Emission")
        {
            std::string listStr = value;
            if (value.find('[') == std::string::npos)
                listStr = "[" + value + "]";
            std::vector<std::string> values = parseYamlList(listStr);
            if (values.size() >= 3)
            {
                try
                {
                    mat.emission = glm::vec3(std::stof(values[0]),
                                             std::stof(values[1]),
                                             std::stof(values[2]));
                }
                catch (...) { }
            }
        }
        // Parse IOR
        else if (key == "IOR")
        {
            try { mat.ior = std::stof(value); }
            catch (...) { }
        }
        // Parse Transmission
        else if (key == "Transmission")
        {
            try { mat.transmission = std::stof(value); }
            catch (...) { }
        }
    }

    m_materials.push_back(std::move(mat));
    return static_cast<uint32_t>(m_materials.size() - 1);
}

// 字符串转vec3辅助函数
static glm::vec3 parseVec3(const std::string& str) {
    std::string s = str;
    if (s.find('[') == std::string::npos)
        s = "[" + s + "]";
    std::vector<std::string> values = parseYamlList(s);
    if (values.size() >= 3) {
        try {
            return glm::vec3(std::stof(values[0]), std::stof(values[1]), std::stof(values[2]));
        } catch (...) {}
    }
    return glm::vec3(0.f);
}
