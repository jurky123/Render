#pragma once

#include <memory>
#include <string>

class Scene;

class SceneImporter
{
public:
    SceneImporter();
    ~SceneImporter();

    std::unique_ptr<Scene> importScene(const std::string& path, std::string* resolvedPath = nullptr) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
