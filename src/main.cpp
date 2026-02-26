#include "Application.h"
#include "PBRTLoader.h"
#include "Scene.h"
#include "OptixRenderer.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cstdlib>

static void printHelp()
{
    std::cout <<
        "Usage: PathTracer [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --scene <path>   Load scene file on startup (OBJ/FBX/GLTF/…)\n"
        "  --width  <px>    Window width  (default: 1280)\n"
        "  --height <px>    Window height (default: 720)\n"
        "  --spp   <n>      Samples per pixel (default: 4)\n"
        "  --help           Show this message\n"
        "\n"
        "Controls:\n"
        "  W/A/S/D/Q/E      Fly camera\n"
        "  Right-drag       Rotate camera\n"
        "  Scroll           Zoom (FoV)\n"
        "  F1               Toggle UI\n"
        "  Esc              Quit\n";
}

int main(int argc, char** argv)
{
    int         width      = 1280;
    int         height     = 720;
    int         spp        = 4;
    std::string scenePath;
    std::string logPath;
    std::unique_ptr<std::ofstream> logFile;
    std::streambuf* coutBuf = nullptr;
    std::streambuf* cerrBuf = nullptr;
    bool        testMode = false;
    std::string testPbrtPath;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printHelp();
            return 0;
        }
        else if (arg == "--scene" && i + 1 < argc)
        {
            scenePath = argv[++i];
        }
        else if (arg == "--width" && i + 1 < argc)
        {
            width = std::stoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            height = std::stoi(argv[++i]);
        }
        else if (arg == "--spp" && i + 1 < argc)
        {
            spp = std::stoi(argv[++i]);
        }
        else if (arg == "--log" && i + 1 < argc)
        {
            logPath = argv[++i];
        }
        else if (arg == "--test-pbrt" && i + 1 < argc)
        {
            testMode = true;
            testPbrtPath = argv[++i];
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            printHelp();
            return 1;
        }
    }

    if (!logPath.empty())
    {
        logFile = std::make_unique<std::ofstream>(logPath, std::ios::out | std::ios::trunc);
        if (logFile->is_open())
        {
            coutBuf = std::cout.rdbuf(logFile->rdbuf());
            cerrBuf = std::cerr.rdbuf(logFile->rdbuf());
#ifdef _WIN32
            _putenv_s("PATHTRACER_LOG", logPath.c_str());
#else
            setenv("PATHTRACER_LOG", logPath.c_str(), 1);
#endif
        }
        else
        {
            std::cerr << "Failed to open log file: " << logPath << "\n";
        }
    }

    if (testMode)
    {
        if (testPbrtPath.empty() || !std::filesystem::exists(testPbrtPath))
        {
            std::cerr << "File not found: " << testPbrtPath << "\n";
            if (coutBuf) std::cout.rdbuf(coutBuf);
            if (cerrBuf) std::cerr.rdbuf(cerrBuf);
            return 1;
        }
        Scene scene;
        bool ok = PBRTLoader::load(testPbrtPath, scene);
        
        std::cout << "\n=== RESULT ===\n";
        std::cout << "Success: " << (ok ? "yes" : "no") << "\n";
        std::cout << "Meshes: " << scene.meshes().size() << "\n";
        std::cout << "Materials: " << scene.materials().size() << "\n";
        std::cout << "Lights: " << scene.lights().size() << "\n";
        
        // Print environment map info
        const auto& env = scene.environment();
        if (env.valid) {
            std::cout << "Environment Map: " << (env.isHDR ? "HDR" : "LDR")
                      << " (" << env.hdrWidth << "x" << env.hdrHeight << ")"
                      << " scale: (" << env.scale.x << ", " << env.scale.y << ", " << env.scale.z << ")\n";
        }

        // CAMERA
        std::cout << "\n=== CAMERA ===\n";
        const auto& cam = scene.camera();
        std::cout << "Eye: (" << cam.eye.x << ", " << cam.eye.y << ", " << cam.eye.z << ")\n";
        std::cout << "Target: (" << cam.target.x << ", " << cam.target.y << ", " << cam.target.z << ")\n";
        std::cout << "Up: (" << cam.up.x << ", " << cam.up.y << ", " << cam.up.z << ")\n";
        std::cout << "FOVy: " << cam.fovy << "\n";
        std::cout << "Valid: " << (cam.valid ? "yes" : "no") << "\n";

        // LIGHTS
        std::cout << "\n=== LIGHTS (first 10) ===\n";
        for (size_t i = 0; i < std::min<size_t>(10, scene.lights().size()); ++i) {
            const auto& light = scene.lights()[i];
            std::cout << "[" << i << "] type=" << (int)light.type
                      << " pos=(" << light.position.x << "," << light.position.y << "," << light.position.z << ")"
                      << " dir=(" << light.direction.x << "," << light.direction.y << "," << light.direction.z << ")"
                      << " intensity=(" << light.intensity.x << "," << light.intensity.y << "," << light.intensity.z << ")"
                      << "\n";
        }

        std::cout << "\n=== MATERIALS ===\n";
        for (size_t i = 0; i < scene.materials().size(); ++i) {
            const auto& m = scene.materials()[i];
            std::cout << "[" << i << "] name=\"" << m.name << "\""
                      << " color=(" << m.baseColor.x << "," << m.baseColor.y << "," << m.baseColor.z << ")"
                      << " metal=" << m.metallic << " rough=" << m.roughness
                      << " ior=" << m.ior << " trans=" << m.transmission
                      << " emit=(" << m.emission.x << "," << m.emission.y << "," << m.emission.z << ")"
                      << "\n";
        }
        std::cout << "\n=== FIRST 10 MESHES ===\n";
        for (size_t i = 0; i < std::min<size_t>(10, scene.meshes().size()); ++i) {
            const auto& mesh = scene.meshes()[i];
            std::cout << "[" << i << "] name=\"" << mesh.name << "\""
                      << " verts=" << mesh.vertices.size()
                      << " tris=" << mesh.indices.size()/3
                      << " matIdx=" << mesh.materialIndex << "\n";
        }
        if (coutBuf) std::cout.rdbuf(coutBuf);
        if (cerrBuf) std::cerr.rdbuf(cerrBuf);
        return ok ? 0 : 1;
    }

    try
    {
        Application app("Path Tracer", width, height);

        if (!scenePath.empty())
            app.loadScene(scenePath);

        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
