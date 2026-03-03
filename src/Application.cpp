#include "Application.h"

#include "Window.h"
#include "Camera.h"
#include "Scene.h"
#include "Renderer.h"

#include <glm/glm.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

Application::Application(const std::string& title, int width, int height)
    : m_title(title), m_width(width), m_height(height)
{
    // Open log file for debugging scene loading
    static std::ofstream logFile("PathTracer_debug.log");
    logFile << "[PathTracer] Initializing application...\n";
    logFile << "[PathTracer] Working directory: " << std::filesystem::current_path().string() << "\n";
    logFile.flush();
    
    // GLFW + glad
    if (!glfwInit())
        throw std::runtime_error("Failed to initialise GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // vsync

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("Failed to initialise GLAD");

    // Store pointer for static callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback           (m_window, onKeyCallback);
    glfwSetMouseButtonCallback   (m_window, onMouseButtonCallback);
    glfwSetCursorPosCallback     (m_window, onCursorPosCallback);
    glfwSetScrollCallback        (m_window, onScrollCallback);
    glfwSetFramebufferSizeCallback(m_window, onFramebufferResize);

    // Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // Sub-systems
    // Start with Cornell Box camera position (looking along +Z axis)
    // Position: (278, 273, -800), Target: (278, 273, 0) => front = (0, 0, 1)
    // FOV: 39.3 degrees (from Cornell Box YAML)
    m_camera   = std::make_unique<Camera>(glm::vec3(278.f, 273.f, -800.f), 90.f, 0.f, 39.3f);
    m_scene    = std::make_unique<Scene>();
    m_renderer = std::make_unique<Renderer>(m_width, m_height);
    m_renderer->setScene(m_scene.get());

    std::cout << "[PathTracer] Initialised " << m_width << "x" << m_height << "\n";
    
    logFile << "[PathTracer] Application initialized successfully.\n";
    logFile.flush();
    
    // Scan for available scenes
    scanSceneDirectories();
    
    bool autoLoadRequested = false;

    // Auto-load demo scene if available
    for (size_t i = 0; i < m_sceneList.size(); ++i)
    {
        if (m_sceneList[i].name.find("bistro") != std::string::npos)
        {
            m_selectedSceneIndex = static_cast<int>(i);
            std::cout << "[PathTracer] Auto-loading demo scene: " << m_sceneList[i].path << "\n";
            loadScene(m_sceneList[i].path);
            autoLoadRequested = true;
            break;
        }
    }
    
    // If no bistro found, try loading first Cornell Box scene if available
    if (!autoLoadRequested && !m_sceneList.empty())
    {
        for (size_t i = 0; i < m_sceneList.size(); ++i)
        {
            if (m_sceneList[i].name.find("cornell_box_official") != std::string::npos)
            {
                m_selectedSceneIndex = static_cast<int>(i);
                loadScene(m_sceneList[i].path);
                autoLoadRequested = true;
                break;
            }
        }
    }

    // If still no scene selected, try loading any scene
    if (!autoLoadRequested && !m_sceneList.empty())
    {
        m_selectedSceneIndex = 0;
        loadScene(m_sceneList[0].path);
    }
}

Application::~Application()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_window)
        glfwDestroyWindow(m_window);
    glfwTerminate();
}

// ---------------------------------------------------------------------------
// Scene scanning
// ---------------------------------------------------------------------------

void Application::scanSceneDirectories()
{
    m_sceneList.clear();
    m_sceneCategories.clear();
    
    // Define scene directories to scan (only built-in models)
    std::vector<std::pair<std::string, std::string>> sceneDirs = {
        {"models", "Built-in"}
    };
    
    // Try to find directories relative to working directory and parent
    std::vector<std::string> searchPaths = {".", "..", "../.."};
    
    for (const auto& [dirName, category] : sceneDirs)
    {
        for (const auto& basePath : searchPaths)
        {
            std::filesystem::path fullPath = std::filesystem::path(basePath) / dirName;
            if (std::filesystem::exists(fullPath) && std::filesystem::is_directory(fullPath))
            {
                scanDirectory(fullPath.string(), category);
                break;  // Found this directory, no need to check other search paths
            }
        }
    }
    
    // Sort scenes by category then name
    std::sort(m_sceneList.begin(), m_sceneList.end(), 
        [](const SceneEntry& a, const SceneEntry& b) {
            if (a.category != b.category) return a.category < b.category;
            return a.name < b.name;
        });
    
    // Build unique category list
    for (const auto& scene : m_sceneList)
    {
        if (std::find(m_sceneCategories.begin(), m_sceneCategories.end(), scene.category) == m_sceneCategories.end())
        {
            m_sceneCategories.push_back(scene.category);
        }
    }
    
    std::cout << "[PathTracer] Found " << m_sceneList.size() << " scenes in " << m_sceneCategories.size() << " categories\n";
}

void Application::scanDirectory(const std::string& basePath, const std::string& category)
{
    std::vector<std::string> sceneExtensions = {".yaml", ".yml", ".obj", ".fbx", ".gltf", ".glb"};

    // Files to skip - commonly referenced as assets, not entry points
    std::vector<std::string> skipPatterns = {
        "left.obj", "right.obj", "wall.obj", "short_block.obj", "tall_block.obj", // cornell box parts
        "light.obj",  // light component
        "breakfast_room.obj",  // single model file, not scene
        "light.mtl", "material.mtl"  // material files
    };

    try
    {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(basePath))
        {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            std::string lowFilename = filename;
            std::transform(lowFilename.begin(), lowFilename.end(), lowFilename.begin(), ::tolower);

            // Skip files by exact name match
            bool shouldSkip = false;
            for (const auto& skipPattern : skipPatterns)
            {
                if (lowFilename == skipPattern)
                {
                    shouldSkip = true;
                    break;
                }
            }
            if (shouldSkip) continue;

            // Check extension
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            bool isSceneFile = false;
            for (const auto& validExt : sceneExtensions)
            {
                if (ext == validExt)
                {
                    isSceneFile = true;
                    break;
                }
            }

            if (!isSceneFile) continue;

            // Skip common non-scene patterns
            if (lowFilename.find("_mtl") != std::string::npos ||        // Material files
                lowFilename.find("texture") != std::string::npos ||     // Textures
                lowFilename.find("_tex") != std::string::npos ||        // Texture variants
                lowFilename.find("_map") != std::string::npos ||        // Maps
                lowFilename.find("_normal") != std::string::npos ||     // Normal maps
                lowFilename.find("_diffuse") != std::string::npos ||    // Diffuse maps
                lowFilename.find("_albedo") != std::string::npos)       // Albedo maps
            {
                continue;
            }

            // Create scene entry
            SceneEntry sceneEntry;

            // Build display name from path relative to base
            std::filesystem::path relPath = std::filesystem::relative(entry.path(), basePath);
            sceneEntry.name = relPath.parent_path().string();
            if (!sceneEntry.name.empty())
            {
                sceneEntry.name += "/";
            }
            sceneEntry.name += entry.path().stem().string();

            // Store the path relative to working directory
            sceneEntry.path = entry.path().string();
            sceneEntry.category = category;

            m_sceneList.push_back(sceneEntry);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[PathTracer] Error scanning " << basePath << ": " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Async scene loading
// ---------------------------------------------------------------------------

Application::LoadResult Application::loadSceneBlocking(const std::string& path)
{
    LoadResult result;
    std::ofstream logFile("PathTracer_debug.log", std::ios::app);

    m_loadingPhase.store(0);
    m_loadingProgress.store(0.02f);

    std::string resolvedPath = path;
    logFile << "\n[PathTracer] Attempting to load scene: " << path << "\n";
    logFile << "[PathTracer] Current working directory: " << std::filesystem::current_path().string() << "\n";
    logFile.flush();

    if (!std::filesystem::exists(resolvedPath))
    {
        // Try from parent directory (for build/Debug -> models/)
        std::filesystem::path upOneLevelPath = std::filesystem::path("..") / path;
        std::string upOneStr = upOneLevelPath.lexically_normal().string();
        if (std::filesystem::exists(upOneStr))
        {
            resolvedPath = upOneStr;
        }
        else
        {
            // Try from parent-parent directory (for build -> models/)
            std::filesystem::path upTwoLevelsPath = std::filesystem::path("../..") / path;
            std::string upTwoStr = upTwoLevelsPath.lexically_normal().string();
            if (std::filesystem::exists(upTwoStr))
            {
                resolvedPath = upTwoStr;
            }
            else
            {
                result.error = "Path not found: " + path;
                logFile << "[PathTracer] Path not found: " << path << "\n";
                logFile.flush();
                m_loadingProgress.store(0.f);
                return result;
            }
        }
    }

    const std::string ext = std::filesystem::path(resolvedPath).extension().string();
    m_loadingPhase.store(1);
    m_loadingProgress.store(0.08f);
    logFile << "[PathTracer] File extension: " << ext << "\n";
    logFile << "[PathTracer] Loading from: " << resolvedPath << "\n";
    logFile.flush();

    auto scene = std::make_unique<Scene>();
    m_loadingPhase.store(2);
    m_loadingProgress.store(0.12f);
    bool ok = false;
    if (ext == ".yaml" || ext == ".yml")
    {
        logFile << "[PathTracer] Using YAML loader\n";
        logFile.flush();
        m_loadingPhase.store(3);
        m_loadingProgress.store(0.2f);
        ok = scene->loadFromYaml(resolvedPath);
    }
    else
    {
        logFile << "[PathTracer] Using Assimp loader\n";
        logFile.flush();
        m_loadingPhase.store(3);
        m_loadingProgress.store(0.2f);
        ok = scene->load(resolvedPath);
    }

    m_loadingPhase.store(4);
    m_loadingProgress.store(0.85f);

    if (!ok)
    {
        result.error = "Failed to load scene: " + resolvedPath;
        logFile << "[PathTracer] Failed to load scene: " << resolvedPath << "\n";
        logFile.flush();
        m_loadingProgress.store(0.f);
        return result;
    }

    result.ok = true;
    result.scene = std::move(scene);
    result.resolvedPath = resolvedPath;
    logFile << "[PathTracer] Scene loaded successfully: " << resolvedPath << "\n";
    logFile.flush();
    m_loadingPhase.store(5);
    m_loadingProgress.store(0.95f);
    return result;
}

void Application::startAsyncLoad(const std::string& path)
{
    if (m_isLoading)
    {
        m_pendingScenePath = path;
        m_hasPendingScene = true;
        return;
    }

    m_isLoading = true;
    m_loadingPath = path;
    m_loadingError.clear();
    m_loadingProgress.store(0.01f);
    m_loadingPhase.store(0);

    m_loadFuture = std::async(std::launch::async, [this, path]() {
        return loadSceneBlocking(path);
    });
}

void Application::pollAsyncLoad()
{
    if (!m_isLoading || !m_loadFuture.valid())
        return;

    if (m_loadFuture.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        return;

    LoadResult result = m_loadFuture.get();
    m_isLoading = false;

    if (result.ok && result.scene)
    {
        m_scene = std::move(result.scene);
        m_sceneLoaded = true;

        // Apply camera from scene if available (works for both YAML and PBRT)
        if (m_scene->camera().valid)
        {
            const auto& cam = m_scene->camera();
            glm::vec3 front = glm::normalize(cam.target - cam.eye);
            float yaw = glm::degrees(atan2(front.z, front.x));
            float pitch = glm::degrees(asin(front.y));
            m_camera = std::make_unique<Camera>(cam.eye, yaw, pitch, cam.fovy);
            std::cout << "[PathTracer] Applied scene camera: eye=(" 
                      << cam.eye.x << ", " << cam.eye.y << ", " << cam.eye.z 
                      << ") FOV=" << cam.fovy << "\n";
        }

        m_renderer->setScene(m_scene.get());
        m_renderer->resetAccumulation();
        std::cout << "[PathTracer] Scene loaded successfully: " << result.resolvedPath << "\n";
        m_loadingPhase.store(6);
        m_loadingProgress.store(1.0f);
    }
    else
    {
        if (!result.error.empty())
            m_loadingError = result.error;
        if (!m_sceneLoaded)
            m_sceneLoaded = false;
        std::cerr << "[PathTracer] " << (m_loadingError.empty() ? "Failed to load scene" : m_loadingError) << "\n";
        m_loadingPhase.store(6);
        m_loadingProgress.store(0.f);
    }

    if (m_hasPendingScene)
    {
        const std::string nextPath = m_pendingScenePath;
        m_pendingScenePath.clear();
        m_hasPendingScene = false;
        startAsyncLoad(nextPath);
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

bool Application::loadScene(const std::string& path)
{
    // Temporary: load synchronously to ensure the scene is ready before rendering.
    // This avoids an empty scene while async loading is still running.
    LoadResult result = loadSceneBlocking(path);
    if (result.ok && result.scene)
    {
        m_scene = std::move(result.scene);
        m_sceneLoaded = true;

        if (m_scene->camera().valid)
        {
            const auto& cam = m_scene->camera();
            glm::vec3 front = glm::normalize(cam.target - cam.eye);
            float yaw = glm::degrees(atan2(front.z, front.x));
            float pitch = glm::degrees(asin(front.y));
            m_camera = std::make_unique<Camera>(cam.eye, yaw, pitch, cam.fovy);
            std::cout << "[PathTracer] Applied scene camera: eye=("
                      << cam.eye.x << ", " << cam.eye.y << ", " << cam.eye.z
                      << ") FOV=" << cam.fovy << "\n";
        }

        m_renderer->setScene(m_scene.get());
        m_renderer->resetAccumulation();
        std::cout << "[PathTracer] Scene loaded successfully: " << result.resolvedPath << "\n";
        m_loadingPhase.store(6);
        m_loadingProgress.store(1.0f);
        return true;
    }

    if (!result.error.empty())
        m_loadingError = result.error;
    m_loadingPhase.store(6);
    m_loadingProgress.store(0.f);
    std::cerr << "[PathTracer] "
              << (m_loadingError.empty() ? "Failed to load scene" : m_loadingError)
              << "\n";
    return false;
}

void Application::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        const float currentTime = static_cast<float>(glfwGetTime());
        const float deltaTime   = currentTime - m_lastFrameTime;
        m_lastFrameTime = currentTime;

        glfwPollEvents();
        processInput(deltaTime);
        pollAsyncLoad();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (m_showUI)
            renderUI();

        // Render scene
        renderFrame();

        // Draw ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window);
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void Application::processInput(float deltaTime)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard)
        return; // ImGui is consuming keyboard

    const float speed = m_moveSpeed * deltaTime;
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
    {
        m_camera->moveForward(speed);
        m_renderer->resetAccumulation();
    }
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
    {
        m_camera->moveForward(-speed);
        m_renderer->resetAccumulation();
    }
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
    {
        m_camera->moveRight(-speed);
        m_renderer->resetAccumulation();
    }
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
    {
        m_camera->moveRight(speed);
        m_renderer->resetAccumulation();
    }
    if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        m_camera->moveUp(-speed);
        m_renderer->resetAccumulation();
    }
    if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS)
    {
        m_camera->moveUp(speed);
        m_renderer->resetAccumulation();
    }
}

void Application::renderFrame()
{
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_renderer->setSamplesPerPixel(m_samplesPerPixel);
    m_renderer->setMaxBounces(m_maxBounces);
    m_renderer->setExposure(m_exposure);
    m_renderer->setDebugDirectional(m_debugDirectional);
    m_renderer->setDebugMode(m_debugMode);
    m_renderer->render(*m_camera);

    if (m_showAxis)
    {
        const float axisLength = 50.f;
        const ImVec2 origin(80.f, static_cast<float>(m_height) - 80.f);
        ImDrawList* drawList = ImGui::GetForegroundDrawList();

        const glm::vec3 right = m_camera->right();
        const glm::vec3 up = m_camera->up();
        const glm::vec3 front = m_camera->forward();

        auto axisToScreen = [&](const glm::vec3& axis) -> ImVec2 {
            float x = glm::dot(axis, right);
            float y = glm::dot(axis, up);
            float len = std::sqrt(x * x + y * y);
            if (len < 1e-4f)
                return ImVec2(0.f, 0.f);
            x /= len;
            y /= len;
            return ImVec2(origin.x + x * axisLength, origin.y - y * axisLength);
        };

        const ImU32 colX = IM_COL32(230, 80, 70, 255);
        const ImU32 colY = IM_COL32(90, 200, 90, 255);
        const ImU32 colZ = IM_COL32(80, 140, 230, 255);

        const ImVec2 xEnd = axisToScreen(glm::vec3(1.f, 0.f, 0.f));
        const ImVec2 yEnd = axisToScreen(glm::vec3(0.f, 1.f, 0.f));
        const ImVec2 zEnd = axisToScreen(glm::vec3(0.f, 0.f, 1.f));

        drawList->AddLine(origin, xEnd, colX, 2.0f);
        drawList->AddLine(origin, yEnd, colY, 2.0f);
        drawList->AddLine(origin, zEnd, colZ, 2.0f);

        drawList->AddText(ImVec2(xEnd.x + 4.f, xEnd.y - 6.f), colX, "X");
        drawList->AddText(ImVec2(yEnd.x + 4.f, yEnd.y - 6.f), colY, "Y");
        drawList->AddText(ImVec2(zEnd.x + 4.f, zEnd.y - 6.f), colZ, "Z");
        (void)front;
    }

    if (m_debugDirectional && m_sceneLoaded)
    {
        const auto& lights = m_scene->lights();
        const Light* dirLight = nullptr;
        for (const auto& light : lights)
        {
            if (light.type == LightType::Directional)
            {
                dirLight = &light;
                break;
            }
        }

        ImDrawList* drawList = ImGui::GetForegroundDrawList();
        const ImVec2 origin(static_cast<float>(m_width) - 120.f, 120.f);
        const float arrowLen = 50.f;
        const ImU32 arrowColor = IM_COL32(255, 200, 0, 255);

        if (dirLight)
        {
            const glm::vec3 right = m_camera->right();
            const glm::vec3 up = m_camera->up();
            const glm::vec3 incoming = -glm::normalize(dirLight->direction);

            float x = glm::dot(incoming, right);
            float y = glm::dot(incoming, up);
            float len = std::sqrt(x * x + y * y);
            if (len > 1e-4f)
            {
                x /= len;
                y /= len;
            }

            ImVec2 tip(origin.x + x * arrowLen, origin.y - y * arrowLen);
            drawList->AddLine(origin, tip, arrowColor, 2.5f);

            ImVec2 left(tip.x - 6.f, tip.y - 4.f);
            ImVec2 rightPt(tip.x - 6.f, tip.y + 4.f);
            drawList->AddTriangleFilled(tip, left, rightPt, arrowColor);
            drawList->AddText(ImVec2(origin.x - 20.f, origin.y - 10.f), arrowColor, "Dir");
        }
        else
        {
            drawList->AddText(ImVec2(origin.x - 30.f, origin.y - 10.f), arrowColor, "Dir: none");
        }
    }
}

void Application::renderUI()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_Once);
    ImGui::Begin("Path Tracer Controls");

    // -- Scene loading -------------------------------------------------------
    ImGui::SeparatorText("Scene");

    if (m_isLoading)
    {
        static const char* kPhases[] = {
            "Resolving path", "Detecting file type", "Preparing loader", "Parsing scene",
            "Finalizing", "Post-process", "Done"
        };
        ImGui::TextColored(ImVec4(1.f, 0.8f, 0.2f, 1.f), "Loading...");
        ImGui::TextWrapped("%s", m_loadingPath.c_str());
        int phase = m_loadingPhase.load();
        const int phaseCount = 7;
        phase = std::clamp(phase, 0, phaseCount - 1);
        ImGui::Text("%s", kPhases[phase]);
        float progress = m_loadingProgress.load();
        ImGui::ProgressBar(progress, ImVec2(-1.f, 0.f));
    }

    if (!m_loadingError.empty())
    {
        ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f), "%s", m_loadingError.c_str());
    }

    if (m_hasPendingScene)
    {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.9f, 1.f), "Queued: %s", m_pendingScenePath.c_str());
    }

    ImGui::BeginDisabled(m_isLoading);
    
    // Scene dropdown list
    if (!m_sceneList.empty())
    {
        // Current selected scene name for display
        const char* previewValue = (m_selectedSceneIndex >= 0 && m_selectedSceneIndex < static_cast<int>(m_sceneList.size()))
            ? m_sceneList[m_selectedSceneIndex].name.c_str()
            : "Select a scene...";
        
        ImGui::SetNextItemWidth(-1);  // Full width
        if (ImGui::BeginCombo("##SceneCombo", previewValue, ImGuiComboFlags_HeightLarge))
        {
            std::string currentCategory;
            
            for (int i = 0; i < static_cast<int>(m_sceneList.size()); ++i)
            {
                const auto& scene = m_sceneList[i];
                
                // Category separator
                if (scene.category != currentCategory)
                {
                    if (!currentCategory.empty())
                        ImGui::Separator();
                    ImGui::TextDisabled("%s", scene.category.c_str());
                    currentCategory = scene.category;
                }
                
                // Scene entry
                bool isSelected = (m_selectedSceneIndex == i);
                if (ImGui::Selectable(scene.name.c_str(), isSelected))
                {
                    m_selectedSceneIndex = i;
                    loadScene(scene.path);
                }
                
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        // Refresh button
        ImGui::SameLine();
        if (ImGui::Button("Refresh"))
        {
            scanSceneDirectories();
        }
        
        // Show scene count
        ImGui::Text("Available: %d scenes", static_cast<int>(m_sceneList.size()));
    }
    else
    {
        ImGui::TextColored(ImVec4(1,1,0,1), "No scenes found!");
        if (ImGui::Button("Scan Directories"))
        {
            scanSceneDirectories();
        }
    }
    
    // Manual path input (collapsed by default)
    if (ImGui::TreeNode("Manual Path"))
    {
        ImGui::InputText("##scenepath", m_scenePathBuf, sizeof(m_scenePathBuf));
        ImGui::SameLine();
        if (ImGui::Button("Load"))
            loadScene(m_scenePathBuf);
        ImGui::TreePop();
    }

    ImGui::EndDisabled();
    
    if (m_sceneLoaded)
    {
        ImGui::TextColored(ImVec4(0,1,0,1), "Scene loaded");
        const auto& stats = m_scene->stats();
        ImGui::Text("Meshes:    %u", stats.meshCount);
        ImGui::Text("Triangles: %u", stats.triangleCount);
        ImGui::Text("Materials: %u", stats.materialCount);

        const auto& lights = m_scene->lights();
        int dirCount = 0;
        for (const auto& light : lights)
        {
            if (light.type == LightType::Directional)
            {
                ImGui::Text("DirLight %d: (%.3f, %.3f, %.3f)",
                            dirCount,
                            light.direction.x,
                            light.direction.y,
                            light.direction.z);
                ImGui::Text("DirLight %d I: (%.3f, %.3f, %.3f)",
                            dirCount,
                            light.intensity.x,
                            light.intensity.y,
                            light.intensity.z);
                ++dirCount;
            }
        }
        if (dirCount == 0)
            ImGui::TextUnformatted("DirLight: <none>");
        ImGui::Text("Ambient: (%.3f, %.3f, %.3f)",
                    m_scene->ambientInt().x,
                    m_scene->ambientInt().y,
                    m_scene->ambientInt().z);
    }

    // -- Render settings -----------------------------------------------------
    ImGui::SeparatorText("Render Settings");
    bool changed = false;
    changed |= ImGui::SliderInt("Samples / Pixel", &m_samplesPerPixel,  1, 64);
    changed |= ImGui::SliderInt("Max Bounces",     &m_maxBounces,       1, 32);
    changed |= ImGui::SliderFloat("Exposure",      &m_exposure,         0.1f, 10.f);
    ImGui::Checkbox("Show Axis", &m_showAxis);
    if (ImGui::Checkbox("Debug Directional Light", &m_debugDirectional))
        m_renderer->resetAccumulation();
    
    // 诊断模式选择
    const char* debugModes[] = {
        "完整追踪 (0)",
        "法线可视化 (1)",
        "直接光 (2)",
        "材质颜色 (3)"
    };
    if (ImGui::Combo("诊断模式", &m_debugMode, debugModes, 4))
        m_renderer->resetAccumulation();
    
    if (changed)
        m_renderer->resetAccumulation();

    // -- Camera --------------------------------------------------------------
    ImGui::SeparatorText("Camera");
    glm::vec3 pos = m_camera->position();
    ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
    ImGui::Text("Yaw: %.1f  Pitch: %.1f", m_camera->yaw(), m_camera->pitch());
    float fov = m_camera->fovDegrees();
    if (ImGui::SliderFloat("FoV", &fov, 10.f, 120.f))
    {
        m_camera->setFovDegrees(fov);
        m_renderer->resetAccumulation();
    }
    if (ImGui::SliderFloat("Move Speed", &m_moveSpeed, 0.1f, 100.0f, "%.1f"))
    {
        // Speed change doesn't require accumulation reset
    }

    // -- Stats ---------------------------------------------------------------
    ImGui::SeparatorText("Stats");
    ImGui::Text("Frame: %.1f ms  (%.0f FPS)",
                1000.f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    ImGui::Text("Accumulated samples: %u", m_renderer->accumulatedSamples());

    // -- Help ----------------------------------------------------------------
    ImGui::SeparatorText("Controls");
    ImGui::TextUnformatted("W/A/S/D/Q/E  : Move camera");
    ImGui::TextUnformatted("Right-drag   : Rotate camera");
    ImGui::TextUnformatted("Scroll       : Zoom (FoV)");
    ImGui::TextUnformatted("Ctrl+Scroll  : Adjust move speed");
    ImGui::TextUnformatted("F1           : Toggle UI");

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Static GLFW callbacks
// ---------------------------------------------------------------------------

void Application::onKeyCallback(GLFWwindow* window, int key, int /*scancode*/,
                                 int action, int /*mods*/)
{
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (key == GLFW_KEY_F1)
            app->m_showUI = !app->m_showUI;
    }
}

void Application::onMouseButtonCallback(GLFWwindow* window, int button,
                                         int action, int /*mods*/)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        app->m_rightDragging = (action == GLFW_PRESS);
        app->m_firstMouse    = true;
        if (app->m_rightDragging)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void Application::onCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app->m_rightDragging)
        return;

    if (app->m_firstMouse)
    {
        app->m_lastMouseX = xpos;
        app->m_lastMouseY = ypos;
        app->m_firstMouse = false;
    }

    const float sensitivity = 0.15f;
    const float dx = static_cast<float>(xpos - app->m_lastMouseX) * sensitivity;
    const float dy = static_cast<float>(app->m_lastMouseY - ypos) * sensitivity;
    app->m_lastMouseX = xpos;
    app->m_lastMouseY = ypos;

    app->m_camera->rotate(dx, dy);
    app->m_renderer->resetAccumulation();
}

void Application::onScrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    // Ctrl+Scroll: adjust camera move speed
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
    {
        app->m_moveSpeed += static_cast<float>(yoffset) * 0.5f;
        app->m_moveSpeed = std::clamp(app->m_moveSpeed, 0.1f, 100.0f);
        // Don't reset accumulation for speed changes
    }
    else
    {
        // Normal scroll: adjust FOV
        float fov = app->m_camera->fovDegrees() - static_cast<float>(yoffset) * 2.f;
        fov = std::clamp(fov, 10.f, 120.f);
        app->m_camera->setFovDegrees(fov);
        app->m_renderer->resetAccumulation();
    }
}

void Application::onFramebufferResize(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    app->m_width  = width;
    app->m_height = height;
    app->m_renderer->resize(width, height);
}
