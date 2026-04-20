#pragma once

#include <memory>
#include <string>

struct GLFWwindow;

class Camera;
class Renderer;
class Scene;
class SceneController;
class ViewportController;
struct AppState;

class Application
{
public:
    explicit Application(const std::string& title = "Path Tracer", int width = 1280, int height = 720);
    ~Application();

    void run();
    bool loadScene(const std::string& path);

private:
    void renderFrame();
    void renderUI();

    GLFWwindow* m_window = nullptr;
    std::string m_title;
    std::unique_ptr<AppState> m_state;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<SceneController> m_sceneController;
    std::unique_ptr<ViewportController> m_viewportController;
};
