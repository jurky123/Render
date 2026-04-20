#include "Application.h"

#include "AppState.h"
#include "Camera.h"
#include "RenderSettings.h"
#include "Renderer.h"
#include "Scene.h"
#include "SceneController.h"
#include "ViewportController.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

#include <glm/glm.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

Application::Application(const std::string& title, int width, int height)
    : m_title(title), m_state(std::make_unique<AppState>())
{
    m_state->width = width;
    m_state->height = height;

    static std::ofstream logFile("PathTracer_debug.log");
    logFile << "[PathTracer] Initializing application...\n";
    logFile << "[PathTracer] Working directory: " << std::filesystem::current_path().string() << "\n";

    if (!glfwInit())
        throw std::runtime_error("Failed to initialise GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(width, height, m_title.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("Failed to initialise GLAD");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    m_camera = std::make_unique<Camera>(glm::vec3(278.f, 273.f, -800.f), 90.f, 0.f, 39.3f);
    m_scene = std::make_unique<Scene>();
    m_renderer = std::make_unique<Renderer>(width, height);
    m_renderer->setScene(m_scene.get());
    m_sceneController = std::make_unique<SceneController>(*m_state);
    m_viewportController = std::make_unique<ViewportController>(*m_state, *m_camera);
    m_viewportController->bind(
        m_window,
        [this]() { m_renderer->resetAccumulation(); },
        [this](int w, int h) { m_renderer->resize(w, h); },
        [this]() { m_state->showUI = !m_state->showUI; });

    m_sceneController->scanSceneDirectories();
    m_sceneController->tryAutoLoad(*m_camera, *m_renderer, m_scene);
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

bool Application::loadScene(const std::string& path)
{
    return m_sceneController->loadScene(path, *m_camera, *m_renderer, m_scene);
}

void Application::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        const float currentTime = static_cast<float>(glfwGetTime());
        const float deltaTime = currentTime - m_state->lastFrameTime;
        m_state->lastFrameTime = currentTime;

        glfwPollEvents();
        m_viewportController->processInput(deltaTime);
        m_sceneController->update(*m_camera, *m_renderer, m_scene);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (m_state->showUI)
            renderUI();

        renderFrame();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(m_window);
    }
}

void Application::renderFrame()
{
    glViewport(0, 0, m_state->width, m_state->height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    RenderSettings settings;
    settings.samplesPerPixel = m_state->render.samplesPerPixel;
    settings.maxBounces = m_state->render.maxBounces;
    settings.exposure = m_state->render.exposure;
    settings.debugDirectional = m_state->render.debugDirectional;
    settings.debugMode = m_state->render.debugMode;
    m_renderer->render(*m_camera, settings);

    if (m_state->render.showAxis)
    {
        const float axisLength = 50.f;
        const ImVec2 origin(80.f, static_cast<float>(m_state->height) - 80.f);
        ImDrawList* drawList = ImGui::GetForegroundDrawList();

        const glm::vec3 right = m_camera->right();
        const glm::vec3 up = m_camera->up();
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
    }
}

void Application::renderUI()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_Once);
    ImGui::Begin("Path Tracer Controls");

    ImGui::SeparatorText("Scene");
    if (!m_state->loading.error.empty())
        ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f), "%s", m_state->loading.error.c_str());

    if (!m_state->sceneList.empty())
    {
        const char* previewValue =
            (m_state->selectedSceneIndex >= 0 && m_state->selectedSceneIndex < static_cast<int>(m_state->sceneList.size()))
            ? m_state->sceneList[m_state->selectedSceneIndex].name.c_str()
            : "Select a scene...";

        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##SceneCombo", previewValue, ImGuiComboFlags_HeightLarge))
        {
            std::string currentCategory;
            for (int i = 0; i < static_cast<int>(m_state->sceneList.size()); ++i)
            {
                const auto& entry = m_state->sceneList[i];
                if (entry.category != currentCategory)
                {
                    if (!currentCategory.empty())
                        ImGui::Separator();
                    ImGui::TextDisabled("%s", entry.category.c_str());
                    currentCategory = entry.category;
                }

                bool isSelected = m_state->selectedSceneIndex == i;
                if (ImGui::Selectable(entry.name.c_str(), isSelected))
                {
                    m_state->selectedSceneIndex = i;
                    loadScene(entry.path);
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::SameLine();
        if (ImGui::Button("Refresh"))
            m_sceneController->scanSceneDirectories();
        ImGui::Text("Available: %d scenes", static_cast<int>(m_state->sceneList.size()));
    }

    if (ImGui::TreeNode("Manual Path"))
    {
        ImGui::InputText("##scenepath", m_state->scenePathBuf, sizeof(m_state->scenePathBuf));
        ImGui::SameLine();
        if (ImGui::Button("Load"))
            loadScene(m_state->scenePathBuf);
        ImGui::TreePop();
    }

    if (m_state->sceneLoaded)
    {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Scene loaded");
        const auto& stats = m_scene->stats();
        ImGui::Text("Meshes: %u", stats.meshCount);
        ImGui::Text("Triangles: %u", stats.triangleCount);
        ImGui::Text("Materials: %u", stats.materialCount);
    }

    ImGui::SeparatorText("Render Settings");
    bool changed = false;
    changed |= ImGui::SliderInt("Samples / Pixel", &m_state->render.samplesPerPixel, 1, 64);
    changed |= ImGui::SliderInt("Max Bounces", &m_state->render.maxBounces, 1, 32);
    changed |= ImGui::SliderFloat("Exposure", &m_state->render.exposure, 0.1f, 10.f);
    ImGui::Checkbox("Show Axis", &m_state->render.showAxis);
    changed |= ImGui::Checkbox("Debug Directional Light", &m_state->render.debugDirectional);

    const char* debugModes[] = { "Full Path Trace (0)", "Normal View (1)", "Direct Light (2)", "Material Base Color (3)" };
    changed |= ImGui::Combo("Debug Mode", &m_state->render.debugMode, debugModes, 4);
    if (changed)
        m_renderer->resetAccumulation();

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
    ImGui::SliderFloat("Move Speed", &m_state->moveSpeed, 0.1f, 100.0f, "%.1f");

    ImGui::SeparatorText("Stats");
    ImGui::Text("Frame: %.1f ms (%.0f FPS)", 1000.f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Accumulated samples: %u", m_renderer->accumulatedSamples());

    ImGui::SeparatorText("Controls");
    ImGui::TextUnformatted("W/A/S/D/Q/E  : Move camera");
    ImGui::TextUnformatted("Right-drag   : Rotate camera");
    ImGui::TextUnformatted("Scroll       : Zoom (FoV)");
    ImGui::TextUnformatted("Ctrl+Scroll  : Adjust move speed");
    ImGui::TextUnformatted("F1           : Toggle UI");

    ImGui::End();
}
