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
#include <stdexcept>
#include <cstring>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

Application::Application(const std::string& title, int width, int height)
    : m_title(title), m_width(width), m_height(height)
{
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
    m_camera   = std::make_unique<Camera>(glm::vec3(0.f, 1.f, 5.f));
    m_scene    = std::make_unique<Scene>();
    m_renderer = std::make_unique<Renderer>(m_width, m_height);

    std::cout << "[PathTracer] Initialised " << m_width << "x" << m_height << "\n";
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
// Public interface
// ---------------------------------------------------------------------------

bool Application::loadScene(const std::string& path)
{
    m_sceneLoaded = m_scene->load(path);
    if (m_sceneLoaded)
    {
        m_renderer->setScene(m_scene.get());
        m_renderer->resetAccumulation();
        std::cout << "[PathTracer] Scene loaded: " << path << "\n";
    }
    else
    {
        std::cerr << "[PathTracer] Failed to load scene: " << path << "\n";
    }
    return m_sceneLoaded;
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

    const float speed = 5.0f * deltaTime;
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
    m_renderer->render(*m_camera);
}

void Application::renderUI()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_Once);
    ImGui::Begin("Path Tracer Controls");

    // -- Scene loading -------------------------------------------------------
    ImGui::SeparatorText("Scene");
    ImGui::InputText("##scenepath", m_scenePathBuf, sizeof(m_scenePathBuf));
    ImGui::SameLine();
    if (ImGui::Button("Load"))
        loadScene(m_scenePathBuf);
    if (m_sceneLoaded)
    {
        ImGui::TextColored(ImVec4(0,1,0,1), "Scene loaded");
        const auto& stats = m_scene->stats();
        ImGui::Text("Meshes:    %u", stats.meshCount);
        ImGui::Text("Triangles: %u", stats.triangleCount);
        ImGui::Text("Materials: %u", stats.materialCount);
    }

    // -- Render settings -----------------------------------------------------
    ImGui::SeparatorText("Render Settings");
    bool changed = false;
    changed |= ImGui::SliderInt("Samples / Pixel", &m_samplesPerPixel,  1, 64);
    changed |= ImGui::SliderInt("Max Bounces",     &m_maxBounces,       1, 32);
    changed |= ImGui::SliderFloat("Exposure",      &m_exposure,         0.1f, 10.f);
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
    float fov = app->m_camera->fovDegrees() - static_cast<float>(yoffset) * 2.f;
    fov = std::clamp(fov, 10.f, 120.f);
    app->m_camera->setFovDegrees(fov);
    app->m_renderer->resetAccumulation();
}

void Application::onFramebufferResize(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    app->m_width  = width;
    app->m_height = height;
    app->m_renderer->resize(width, height);
}
