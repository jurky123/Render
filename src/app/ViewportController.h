#pragma once

#include <functional>

struct GLFWwindow;

class AppState;
class Camera;

class ViewportController
{
public:
    ViewportController(AppState& state, Camera& camera);

    void bind(GLFWwindow* window,
              std::function<void()> onCameraChanged,
              std::function<void(int, int)> onResize,
              std::function<void()> onToggleUi);

    void processInput(float deltaTime);

private:
    static void onKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void onMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void onCursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void onScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void onFramebufferResize(GLFWwindow* window, int width, int height);

    void handleKey(int key, int action);
    void handleMouseButton(int button, int action);
    void handleCursorPos(double xpos, double ypos);
    void handleScroll(GLFWwindow* window, double yoffset);
    void handleResize(int width, int height);

    AppState& m_state;
    Camera& m_camera;
    GLFWwindow* m_window = nullptr;
    std::function<void()> m_onCameraChanged;
    std::function<void(int, int)> m_onResize;
    std::function<void()> m_onToggleUi;
    bool m_firstMouse = true;
    bool m_rightDragging = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
};
