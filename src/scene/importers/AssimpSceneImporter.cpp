#include "AssimpSceneImporter.h"

#include "MaterialLoader.h"

#include <assimp/Importer.hpp>
#include <assimp/config.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>

namespace
{
glm::mat4 toGlm(const aiMatrix4x4& m)
{
    return glm::transpose(glm::make_mat4(&m.a1));
}

bool findNodeTransform(const aiNode* node, const std::string& name, const glm::mat4& parent, glm::mat4& out)
{
    const glm::mat4 current = parent * toGlm(node->mTransformation);
    if (name == node->mName.C_Str())
    {
        out = current;
        return true;
    }
    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        if (findNodeTransform(node->mChildren[i], name, current, out))
            return true;
    }
    return false;
}

Mesh processMesh(const aiMesh* aiM, const glm::mat4& transform)
{
    Mesh mesh;
    mesh.name = aiM->mName.C_Str();
    mesh.materialIndex = aiM->mMaterialIndex;

    const glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(transform)));
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
            v.texCoord = glm::vec2(aiM->mTextureCoords[0][i].x, aiM->mTextureCoords[0][i].y);

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

void processNode(Scene& dst, const aiScene* scene, const aiNode* node, const glm::mat4& parentTransform)
{
    const glm::mat4 transform = parentTransform * toGlm(node->mTransformation);
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        dst.addMesh(processMesh(mesh, transform));
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        processNode(dst, scene, node->mChildren[i], transform);
}

void loadLights(Scene& dst, const aiScene* scene)
{
    for (unsigned int i = 0; i < scene->mNumLights; ++i)
    {
        const aiLight* src = scene->mLights[i];
        if (!src)
            continue;

        glm::mat4 xf(1.f);
        findNodeTransform(scene->mRootNode, src->mName.C_Str(), glm::mat4(1.f), xf);
        const glm::vec3 pos = glm::vec3(xf * glm::vec4(src->mPosition.x, src->mPosition.y, src->mPosition.z, 1.f));
        glm::vec3 dir = glm::vec3(xf * glm::vec4(src->mDirection.x, src->mDirection.y, src->mDirection.z, 0.f));
        if (glm::length(dir) > 0.f)
            dir = glm::normalize(dir);

        glm::vec3 intensity(src->mColorDiffuse.r, src->mColorDiffuse.g, src->mColorDiffuse.b);
        if (glm::length(intensity) <= 0.f)
            intensity = glm::vec3(1.f);

        Light l{};
        switch (src->mType)
        {
        case aiLightSource_DIRECTIONAL:
            l.type = LightType::Directional;
            l.direction = dir;
            l.intensity = intensity;
            break;
        case aiLightSource_POINT:
            l.type = LightType::Point;
            l.position = pos;
            l.intensity = intensity;
            break;
        case aiLightSource_SPOT:
            l.type = LightType::Spot;
            l.position = pos;
            l.direction = dir;
            l.intensity = intensity;
            l.cosInner = std::cos(src->mAngleInnerCone * 0.5f);
            l.cosOuter = std::cos(src->mAngleOuterCone * 0.5f);
            break;
        default:
            continue;
        }
        dst.addLight(l);
    }
}

void loadCamera(Scene& dst, const aiScene* scene)
{
    if (scene->mNumCameras == 0)
        return;

    const aiCamera* camSrc = scene->mCameras[0];
    if (!camSrc)
        return;

    glm::mat4 xf(1.f);
    findNodeTransform(scene->mRootNode, camSrc->mName.C_Str(), glm::mat4(1.f), xf);
    const glm::vec3 eye = glm::vec3(xf * glm::vec4(camSrc->mPosition.x, camSrc->mPosition.y, camSrc->mPosition.z, 1.f));
    glm::vec3 dir = glm::vec3(xf * glm::vec4(camSrc->mLookAt.x, camSrc->mLookAt.y, camSrc->mLookAt.z, 0.f));
    if (glm::length(dir) > 0.f)
        dir = glm::normalize(dir);
    glm::vec3 up = glm::vec3(xf * glm::vec4(camSrc->mUp.x, camSrc->mUp.y, camSrc->mUp.z, 0.f));
    if (glm::length(up) > 0.f)
        up = glm::normalize(up);

    const float aspect = camSrc->mAspect > 0.f ? camSrc->mAspect : 16.f / 9.f;
    float fovy = 45.f;
    if (camSrc->mHorizontalFOV > 0.f)
    {
        const float vRad = 2.f * std::atan(std::tan(camSrc->mHorizontalFOV * 0.5f) / aspect);
        fovy = vRad * 57.2957795f;
    }

    CameraInfo cam;
    cam.eye = eye;
    cam.target = eye + dir;
    cam.up = up;
    cam.fovy = fovy;
    cam.zNear = camSrc->mClipPlaneNear > 0.f ? camSrc->mClipPlaneNear : 0.1f;
    cam.zFar = camSrc->mClipPlaneFar > cam.zNear ? camSrc->mClipPlaneFar : 1000.f;
    cam.valid = true;
    dst.setCamera(cam);
}
}

bool AssimpSceneImporter::appendToScene(Scene& dst, const std::string& path) const
{
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    Assimp::Importer importer;
    if (ext == ".fbx")
    {
        importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
        importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_ALL_GEOMETRY_LAYERS, true);
        importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_MATERIALS, true);
        importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_TEXTURES, true);
    }

    const unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_FlipUVs |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices |
        aiProcess_OptimizeMeshes;

    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode)
    {
        std::cerr << "[Scene] Assimp error: " << importer.GetErrorString() << "\n";
        return false;
    }

    const uint32_t materialBase = static_cast<uint32_t>(dst.materials().size());
    MaterialLoader materialLoader;
    const std::string dir = std::filesystem::path(path).parent_path().string();
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
        dst.addMaterial(materialLoader.convertAssimpMaterial(scene->mMaterials[i], dir, scene));

    const size_t meshCountBefore = dst.meshes().size();
    processNode(dst, scene, scene->mRootNode, glm::mat4(1.f));
    for (size_t i = meshCountBefore; i < dst.meshes().size(); ++i)
        dst.meshes()[i].materialIndex += materialBase;

    loadLights(dst, scene);
    loadCamera(dst, scene);
    dst.setLoadedPath(path);
    dst.rebuildStats();
    return true;
}
