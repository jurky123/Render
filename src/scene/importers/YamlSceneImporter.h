#pragma once

#include <string>

class Scene;

class YamlSceneImporter
{
public:
    bool importScene(Scene& scene, const std::string& yamlPath) const;
};
