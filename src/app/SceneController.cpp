#include "SceneController.h"

#include "Camera.h"
#include "Renderer.h"
#include "Scene.h"
#include "SceneImporter.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

namespace
{
Camera makeCameraFromScene(const Scene& scene)
{
    const auto& cam = scene.camera();
    glm::vec3 front = glm::normalize(cam.target - cam.eye);
    float yaw = glm::degrees(atan2(front.z, front.x));
    float pitch = glm::degrees(asin(front.y));
    return Camera(cam.eye, yaw, pitch, cam.fovy);
}
}

SceneController::SceneController(AppState& state)
    : m_state(state), m_importer(std::make_unique<SceneImporter>())
{
}

SceneController::~SceneController() = default;

void SceneController::scanSceneDirectories()
{
    m_state.sceneList.clear();
    m_state.sceneCategories.clear();

    const std::vector<std::pair<std::string, std::string>> sceneDirs = {
        {"models", "Built-in"}
    };
    const std::vector<std::string> searchPaths = {".", "..", "../.."};

    for (const auto& [dirName, category] : sceneDirs)
    {
        for (const auto& basePath : searchPaths)
        {
            const std::filesystem::path fullPath = std::filesystem::path(basePath) / dirName;
            if (std::filesystem::exists(fullPath) && std::filesystem::is_directory(fullPath))
            {
                scanDirectory(fullPath.string(), category);
                break;
            }
        }
    }

    std::sort(m_state.sceneList.begin(), m_state.sceneList.end(),
              [](const SceneEntry& a, const SceneEntry& b) {
                  return a.category == b.category ? a.name < b.name : a.category < b.category;
              });

    for (const auto& entry : m_state.sceneList)
    {
        if (std::find(m_state.sceneCategories.begin(), m_state.sceneCategories.end(), entry.category) ==
            m_state.sceneCategories.end())
        {
            m_state.sceneCategories.push_back(entry.category);
        }
    }
}

void SceneController::scanDirectory(const std::string& basePath, const std::string& category)
{
    const std::vector<std::string> sceneExtensions = {".yaml", ".yml", ".obj", ".fbx", ".gltf", ".glb", ".usd", ".usda", ".usdc", ".usdz"};
    const std::vector<std::string> skipPatterns = {
        "left.obj", "right.obj", "wall.obj", "short_block.obj", "tall_block.obj",
        "light.obj", "breakfast_room.obj", "light.mtl", "material.mtl"
    };

    try
    {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(basePath))
        {
            if (!entry.is_regular_file())
                continue;

            std::string filename = entry.path().filename().string();
            std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
            if (std::find(skipPatterns.begin(), skipPatterns.end(), filename) != skipPatterns.end())
                continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(sceneExtensions.begin(), sceneExtensions.end(), ext) == sceneExtensions.end())
                continue;

            if (filename.find("_mtl") != std::string::npos ||
                filename.find("texture") != std::string::npos ||
                filename.find("_tex") != std::string::npos ||
                filename.find("_map") != std::string::npos ||
                filename.find("_normal") != std::string::npos ||
                filename.find("_diffuse") != std::string::npos ||
                filename.find("_albedo") != std::string::npos)
            {
                continue;
            }

            SceneEntry sceneEntry;
            const std::filesystem::path relPath = std::filesystem::relative(entry.path(), basePath);
            sceneEntry.name = relPath.parent_path().string();
            if (!sceneEntry.name.empty())
                sceneEntry.name += "/";
            sceneEntry.name += entry.path().stem().string();
            sceneEntry.path = entry.path().string();
            sceneEntry.category = category;
            m_state.sceneList.push_back(sceneEntry);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[PathTracer] Error scanning " << basePath << ": " << e.what() << "\n";
    }
}

SceneController::LoadResult SceneController::loadSceneBlocking(const std::string& path) const
{
    LoadResult result;
    std::ofstream logFile("PathTracer_debug.log", std::ios::app);
    m_state.loading.phase.store(0);
    m_state.loading.progress.store(0.05f);

    try
    {
        result.scene = m_importer->importScene(path, &result.resolvedPath);
        result.ok = result.scene != nullptr;
        if (!result.ok)
            result.error = "Failed to load scene: " + path;
    }
    catch (const std::exception& e)
    {
        result.error = e.what();
    }

    if (!result.ok && logFile)
        logFile << "[PathTracer] " << result.error << "\n";
    return result;
}

void SceneController::applyLoadedScene(LoadResult&& result, Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene)
{
    if (!result.ok || !result.scene)
    {
        m_state.loading.error = result.error;
        m_state.loading.phase.store(6);
        m_state.loading.progress.store(0.f);
        return;
    }

    scene = std::move(result.scene);
    m_state.sceneLoaded = true;
    if (scene->camera().valid)
        camera = makeCameraFromScene(*scene);

    renderer.setScene(scene.get());
    renderer.resetAccumulation();
    m_state.loading.error.clear();
    m_state.loading.phase.store(6);
    m_state.loading.progress.store(1.0f);
}

bool SceneController::loadScene(const std::string& path, Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene)
{
    m_state.loading.path = path;
    m_state.loading.error.clear();
    LoadResult result = loadSceneBlocking(path);
    const bool ok = result.ok;
    applyLoadedScene(std::move(result), camera, renderer, scene);
    return ok;
}

void SceneController::update(Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene)
{
    if (!m_loadFuture.valid())
        return;

    if (m_loadFuture.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        return;

    applyLoadedScene(m_loadFuture.get(), camera, renderer, scene);
}

bool SceneController::tryAutoLoad(Camera& camera, Renderer& renderer, std::unique_ptr<Scene>& scene)
{
    for (size_t i = 0; i < m_state.sceneList.size(); ++i)
    {
        if (m_state.sceneList[i].name.find("bistro") != std::string::npos)
        {
            m_state.selectedSceneIndex = static_cast<int>(i);
            return loadScene(m_state.sceneList[i].path, camera, renderer, scene);
        }
    }

    for (size_t i = 0; i < m_state.sceneList.size(); ++i)
    {
        if (m_state.sceneList[i].name.find("cornell_box_official") != std::string::npos)
        {
            m_state.selectedSceneIndex = static_cast<int>(i);
            return loadScene(m_state.sceneList[i].path, camera, renderer, scene);
        }
    }

    if (!m_state.sceneList.empty())
    {
        m_state.selectedSceneIndex = 0;
        return loadScene(m_state.sceneList[0].path, camera, renderer, scene);
    }

    return false;
}
