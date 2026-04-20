#include "Application.h"

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
