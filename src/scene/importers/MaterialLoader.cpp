#include "MaterialLoader.h"

#include "TextureRepository.h"

#include <assimp/material.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace
{
std::string texturePath(const aiMaterial* mat, aiTextureType type, const std::string& dir)
{
    if (mat->GetTextureCount(type) == 0)
        return {};

    aiString p;
    mat->GetTexture(type, 0, &p);
    const std::string raw = p.C_Str();
    if (!raw.empty() && raw[0] == '*')
        return raw;
    return (std::filesystem::path(dir) / raw).string();
}

std::string trim(const std::string& str)
{
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start])))
        ++start;
    size_t end = str.size();
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1])))
        --end;
    return str.substr(start, end - start);
}

std::vector<std::string> parseYamlList(const std::string& line)
{
    std::vector<std::string> result;
    size_t start = line.find('[');
    size_t end = line.rfind(']');
    if (start == std::string::npos || end == std::string::npos)
        return result;
    std::string content = line.substr(start + 1, end - start - 1);
    std::stringstream ss(content);
    std::string item;
    while (std::getline(ss, item, ','))
        result.push_back(trim(item));
    return result;
}
}

Material MaterialLoader::convertAssimpMaterial(const aiMaterial* aiMat,
                                               const std::string& directory,
                                               const aiScene* scene) const
{
    Material mat;

    aiString name;
    if (aiMat->Get(AI_MATKEY_NAME, name) == AI_SUCCESS)
        mat.name = name.C_Str();

    aiColor4D color;
    float opacityFromColor = 1.f;
    if (aiMat->Get(AI_MATKEY_BASE_COLOR, color) == AI_SUCCESS ||
        aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
    {
        mat.baseColor = glm::vec3(color.r, color.g, color.b);
        opacityFromColor = color.a;
    }

    aiMat->Get(AI_MATKEY_METALLIC_FACTOR, mat.metallic);
    aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, mat.roughness);

    aiColor3D emission;
    if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emission) == AI_SUCCESS)
        mat.emission = glm::vec3(emission.r, emission.g, emission.b);

    aiMat->Get(AI_MATKEY_REFRACTI, mat.ior);

    aiMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, mat.transmission);
    float transparency = 1.f;
    if (aiMat->Get(AI_MATKEY_TRANSPARENCYFACTOR, transparency) == AI_SUCCESS)
        mat.transmission = std::max(mat.transmission, 1.f - transparency);

    float opacity = 1.f;
    aiMat->Get(AI_MATKEY_OPACITY, opacity);
    if (opacityFromColor < opacity)
        opacity = opacityFromColor;
    if (opacity < 1.f)
        mat.transmission = std::max(mat.transmission, 1.f - opacity);

    mat.baseColorTexPath = texturePath(aiMat, aiTextureType_BASE_COLOR, directory);
    if (mat.baseColorTexPath.empty())
        mat.baseColorTexPath = texturePath(aiMat, aiTextureType_DIFFUSE, directory);
    mat.normalTexPath = texturePath(aiMat, aiTextureType_NORMALS, directory);
    mat.metallicRoughnessTexPath = texturePath(aiMat, aiTextureType_METALNESS, directory);
    mat.emissionTexPath = texturePath(aiMat, aiTextureType_EMISSIVE, directory);
    mat.transmissionTexPath = texturePath(aiMat, aiTextureType_TRANSMISSION, directory);
    if (mat.transmissionTexPath.empty())
        mat.transmissionTexPath = texturePath(aiMat, aiTextureType_OPACITY, directory);
    if (mat.transmissionTexPath.empty())
        mat.transmissionTexPath = texturePath(aiMat, aiTextureType_DIFFUSE, directory);

    auto loadMaybeEmbedded = [&](const std::string& path) -> TextureData {
        if (path.empty())
            return {};
        if (TextureRepository::isEmbeddedTextureRef(path) && scene)
        {
            int idx = std::atoi(path.c_str() + 1);
            if (idx >= 0 && idx < static_cast<int>(scene->mNumTextures))
                return TextureRepository::loadEmbeddedTexture(scene->mTextures[idx]);
            return {};
        }
        return TextureRepository::loadTexture(path);
    };

    if (!mat.baseColorTexPath.empty() && !mat.baseColorTexData.pixels)
        mat.baseColorTexData = loadMaybeEmbedded(mat.baseColorTexPath);
    if (!mat.normalTexPath.empty() && !mat.normalTexData.pixels)
        mat.normalTexData = loadMaybeEmbedded(mat.normalTexPath);
    if (!mat.metallicRoughnessTexPath.empty() && !mat.metallicRoughnessTexData.pixels)
        mat.metallicRoughnessTexData = loadMaybeEmbedded(mat.metallicRoughnessTexPath);
    if (!mat.emissionTexPath.empty() && !mat.emissionTexData.pixels)
        mat.emissionTexData = loadMaybeEmbedded(mat.emissionTexPath);
    if (!mat.transmissionTexPath.empty() && !mat.transmissionTexData.pixels)
        mat.transmissionTexData = loadMaybeEmbedded(mat.transmissionTexPath);

    return mat;
}

uint32_t MaterialLoader::loadYamlMaterial(Scene& scene,
                                          const std::string& name,
                                          const std::vector<std::string>& matData,
                                          const std::string& sceneDir) const
{
    (void)sceneDir;
    Material mat;
    mat.name = name;

    for (size_t i = 0; i + 1 < matData.size(); i += 2)
    {
        const std::string& key = matData[i];
        const std::string& value = matData[i + 1];

        auto parseVec = [&](const std::string& raw) {
            std::string listStr = raw;
            if (raw.find('[') == std::string::npos)
                listStr = "[" + raw + "]";
            return parseYamlList(listStr);
        };

        if (key == "Diffuse" || key == "BaseColor")
        {
            const auto values = parseVec(value);
            if (values.size() >= 3)
            {
                mat.baseColor = glm::vec3(std::stof(values[0]), std::stof(values[1]), std::stof(values[2]));
            }
        }
        else if (key == "Metallic")
        {
            mat.metallic = std::stof(value);
        }
        else if (key == "Roughness")
        {
            mat.roughness = std::stof(value);
        }
        else if (key == "Specular")
        {
            const auto values = parseVec(value);
            if (values.size() >= 4)
            {
                float shininess = std::stof(values[3]);
                mat.roughness = 1.0f / (shininess + 1.0f);
            }
        }
        else if (key == "Emission")
        {
            const auto values = parseVec(value);
            if (values.size() >= 3)
                mat.emission = glm::vec3(std::stof(values[0]), std::stof(values[1]), std::stof(values[2]));
        }
        else if (key == "IOR")
        {
            mat.ior = std::stof(value);
        }
        else if (key == "Transmission")
        {
            mat.transmission = std::stof(value);
        }
    }

    scene.addMaterial(mat);
    return static_cast<uint32_t>(scene.materials().size() - 1);
}
