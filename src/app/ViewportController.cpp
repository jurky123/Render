#include "ViewportController.h"

#include "AppState.h"
#include "Camera.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>

ViewportController::ViewportController(AppState& state, Camera& camera)
    : m_state(state), m_camera(camera)
{
}

void ViewportController::bind(GLFWwindow* window,
                              std::function<void()> onCameraChanged,
                              std::function<void(int, int)> onResize,
                              std::function<void()> onToggleUi)
{
    m_window = window;
    m_onCameraChanged = std::move(onCameraChanged);
    m_onResize = std::move(onResize);
    m_onToggleUi = std::move(onToggleUi);

    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, onKeyCallback);
    glfwSetMouseButtonCallback(window, onMouseButtonCallback);
    glfwSetCursorPosCallback(window, onCursorPosCallback);
    glfwSetScrollCallback(window, onScrollCallback);
    glfwSetFramebufferSizeCallback(window, onFramebufferResize);
}

void ViewportController::processInput(float deltaTime)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || !m_window)
        return;

    const float speed = m_state.moveSpeed * deltaTime;
    bool changed = false;
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) { m_camera.moveForward(speed); changed = true; }
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) { m_camera.moveForward(-speed); changed = true; }
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) { m_camera.moveRight(-speed); changed = true; }
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) { m_camera.moveRight(speed); changed = true; }
    if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) { m_camera.moveUp(-speed); changed = true; }
    if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) { m_camera.moveUp(speed); changed = true; }

    if (changed && m_onCameraChanged)
        m_onCameraChanged();
}

void ViewportController::onKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;
    auto* self = static_cast<ViewportController*>(glfwGetWindowUserPointer(window));
    if (self)
        self->handleKey(key, action);
}

void ViewportController::onMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    (void)mods;
    auto* self = static_cast<ViewportController*>(glfwGetWindowUserPointer(window));
    if (self)
        self->handleMouseButton(button, action);
}

void ViewportController::onCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* self = static_cast<ViewportController*>(glfwGetWindowUserPointer(window));
    if (self)
        self->handleCursorPos(xpos, ypos);
}

void ViewportController::onScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    (void)xoffset;
    auto* self = static_cast<ViewportController*>(glfwGetWindowUserPointer(window));
    if (self)
        self->handleScroll(window, yoffset);
}

void ViewportController::onFramebufferResize(GLFWwindow* window, int width, int height)
{
    auto* self = static_cast<ViewportController*>(glfwGetWindowUserPointer(window));
    if (self)
        self->handleResize(width, height);
}

void ViewportController::handleKey(int key, int action)
{
    if (action != GLFW_PRESS || !m_window)
        return;

    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    else if (key == GLFW_KEY_F1 && m_onToggleUi)
        m_onToggleUi();
}

void ViewportController::handleMouseButton(int button, int action)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse || !m_window)
        return;

    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        m_rightDragging = (action == GLFW_PRESS);
        m_firstMouse = true;
        glfwSetInputMode(m_window, GLFW_CURSOR, m_rightDragging ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
}

void ViewportController::handleCursorPos(double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse || !m_rightDragging)
        return;

    if (m_firstMouse)
    {
        m_lastMouseX = xpos;
        m_lastMouseY = ypos;
        m_firstMouse = false;
    }

    const float sensitivity = 0.15f;
    const float dx = static_cast<float>(xpos - m_lastMouseX) * sensitivity;
    const float dy = static_cast<float>(m_lastMouseY - ypos) * sensitivity;
    m_lastMouseX = xpos;
    m_lastMouseY = ypos;

    m_camera.rotate(dx, dy);
    if (m_onCameraChanged)
        m_onCameraChanged();
}

void ViewportController::handleScroll(GLFWwindow* window, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
    {
        m_state.moveSpeed += static_cast<float>(yoffset) * 0.5f;
        m_state.moveSpeed = std::clamp(m_state.moveSpeed, 0.1f, 100.0f);
        return;
    }

    float fov = m_camera.fovDegrees() - static_cast<float>(yoffset) * 2.f;
    fov = std::clamp(fov, 10.f, 120.f);
    m_camera.setFovDegrees(fov);
    if (m_onCameraChanged)
        m_onCameraChanged();
}

void ViewportController::handleResize(int width, int height)
{
    m_state.width = width;
    m_state.height = height;
    if (m_onResize)
        m_onResize(width, height);
}
