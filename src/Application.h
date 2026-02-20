#pragma once

#include <string>
#include <vector>
#include <memory>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Forward declarations
class Scene;
class Camera;
class Renderer;

/**
 * @brief Top-level application class.
 *
 * Owns the window, camera, scene and renderer, and drives the main loop.
 * Mouse and keyboard input is processed here and forwarded to the camera.
 */
class Application
{
public:
    explicit Application(const std::string& title = "Path Tracer",
                         int width  = 1280,
                         int height = 720);
    ~Application();

    /** Run the main loop until the window is closed. */
    void run();

    /** Load a scene file (any format supported by Assimp). */
    bool loadScene(const std::string& path);

private:
    /* ---- GLFW callbacks (static, forwarded to member functions) ---- */
    static void onKeyCallback       (GLFWwindow*, int key, int scancode, int action, int mods);
    static void onMouseButtonCallback(GLFWwindow*, int button, int action, int mods);
    static void onCursorPosCallback  (GLFWwindow*, double xpos, double ypos);
    static void onScrollCallback     (GLFWwindow*, double xoffset, double yoffset);
    static void onFramebufferResize  (GLFWwindow*, int width, int height);

    /* ---- per-frame helpers ---- */
    void processInput (float deltaTime);
    void renderFrame  ();
    void renderUI     ();

    /* ---- members ---- */
    GLFWwindow* m_window = nullptr;
    int         m_width;
    int         m_height;
    std::string m_title;

    std::unique_ptr<Camera>   m_camera;
    std::unique_ptr<Scene>    m_scene;
    std::unique_ptr<Renderer> m_renderer;

    /* mouse state */
    bool   m_firstMouse   = true;
    double m_lastMouseX   = 0.0;
    double m_lastMouseY   = 0.0;
    bool   m_rightDragging = false;

    /* UI state */
    int   m_samplesPerPixel  = 4;
    int   m_maxBounces       = 8;
    float m_exposure         = 1.0f;
    bool  m_showUI           = true;
    bool  m_sceneLoaded      = false;
    char  m_scenePathBuf[512] = {};

    /* timing */
    float m_lastFrameTime = 0.0f;
};
