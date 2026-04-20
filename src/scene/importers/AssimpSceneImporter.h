#pragma once

#include <string>

class Scene;

class AssimpSceneImporter
{
public:
    bool appendToScene(Scene& scene, const std::string& path) const;
};
