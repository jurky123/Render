#include "Scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/material.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <filesystem>

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
    m_stats  = {};
    m_path.clear();
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
        m_materials.push_back(convertMaterial(scene->mMaterials[i], directory));
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

    return mat;
}
