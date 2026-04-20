#pragma once

#include "Scene.h"

#include <string>
#include <vector>

struct aiMaterial;
struct aiScene;

class MaterialLoader
{
public:
    Material convertAssimpMaterial(const aiMaterial* aiMat,
                                   const std::string& directory,
                                   const aiScene* scene) const;

    uint32_t loadYamlMaterial(Scene& scene,
                              const std::string& name,
                              const std::vector<std::string>& matData,
                              const std::string& sceneDir) const;
};
