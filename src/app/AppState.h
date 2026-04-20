#pragma once

#include <atomic>
#include <string>
#include <vector>

struct SceneEntry
{
    std::string name;
    std::string path;
    std::string category;
};

struct LoadingState
{
    std::atomic<bool> isLoading{false};
    std::atomic<float> progress{0.f};
    std::atomic<int> phase{0};
    std::string path;
    std::string error;
    std::string pendingPath;
    bool hasPendingScene = false;
};

struct RenderPanelState
{
    int samplesPerPixel = 4;
    int maxBounces = 8;
    float exposure = 1.0f;
    bool showAxis = false;
    bool debugDirectional = false;
    int debugMode = 0;
};

struct AppState
{
    int width = 1280;
    int height = 720;
    bool showUI = true;
    bool sceneLoaded = false;
    float moveSpeed = 5.0f;
    float lastFrameTime = 0.0f;
    char scenePathBuf[512] = {};

    RenderPanelState render;
    LoadingState loading;

    std::vector<SceneEntry> sceneList;
    std::vector<std::string> sceneCategories;
    int selectedSceneIndex = -1;
};
