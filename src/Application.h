#pragma once

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <atomic>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Forward declarations
class Scene;
class Camera;
class Renderer;

/**
 * @brief Scene entry for the scene browser
 */
struct SceneEntry
{
    std::string name;      ///< Display name
    std::string path;      ///< Relative path to scene file
    std::string category;  ///< Category (e.g., "Cornell Box", "Built-in")
};

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
    struct LoadResult
    {
        bool ok = false;
        std::unique_ptr<Scene> scene;
        std::string resolvedPath;
        std::string error;
    };

    /* ---- Scene scanning ---- */
    void scanSceneDirectories();
    void scanDirectory(const std::string& basePath, const std::string& category);

    /* ---- Async scene loading ---- */
    LoadResult loadSceneBlocking(const std::string& path);
    void startAsyncLoad(const std::string& path);
    void pollAsyncLoad();
    
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
    bool  m_showAxis         = false;
    bool  m_debugDirectional = false;
    int   m_debugMode        = 0;     // 0=完整追踪, 1=法线, 2=直接光, 3=材质颜色
    bool  m_sceneLoaded      = false;
    char  m_scenePathBuf[512] = {};

    /* async loading state */
    std::atomic<bool> m_isLoading{false};
    std::future<LoadResult> m_loadFuture;
    std::string m_loadingPath;
    std::string m_loadingError;
    std::string m_pendingScenePath;
    bool m_hasPendingScene = false;
    std::atomic<float> m_loadingProgress{0.f};
    std::atomic<int> m_loadingPhase{0};
    
    /* Scene browser */
    std::vector<SceneEntry> m_sceneList;
    int m_selectedSceneIndex = -1;
    std::vector<std::string> m_sceneCategories;  ///< Unique category names

    /* camera movement */
    float m_moveSpeed        = 5.0f;  ///< camera movement speed (units per second)

    /* timing */
    float m_lastFrameTime = 0.0f;
};
