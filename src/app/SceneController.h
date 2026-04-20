#pragma once

#include "AppState.h"

#include <future>
#include <memory>
#include <string>

class Camera;
class Renderer;
class Scene;
class SceneImporter;

class SceneController
{
public:
    explicit SceneController(AppState& state);
    ~SceneController();

    void scanSceneDirectories();
    void update(Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene);
    bool loadScene(const std::string& path, Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene);
    bool tryAutoLoad(Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene);

private:
    struct LoadResult
    {
        bool ok = false;
        std::unique_ptr<Scene> scene;
        std::string resolvedPath;
        std::string error;
    };

    LoadResult loadSceneBlocking(const std::string& path) const;
    void applyLoadedScene(LoadResult&& result, Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene);
    void scanDirectory(const std::string& basePath, const std::string& category);

    AppState& m_state;
    std::unique_ptr<SceneImporter> m_importer;
    std::future<LoadResult> m_loadFuture;
};
