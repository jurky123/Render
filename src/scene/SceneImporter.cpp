#include "SceneImporter.h"

#include "Scene.h"
#include "ScenePostProcessor.h"
#include "importers/AssimpSceneImporter.h"
#include "importers/YamlSceneImporter.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
std::string resolveBlenderExecutable()
{
    namespace fs = std::filesystem;
    auto existsFile = [](const std::string& p) { return !p.empty() && fs::exists(fs::path(p)); };

    if (const char* env = std::getenv("BLENDER_EXE"))
        if (existsFile(env)) return env;
    if (const char* env = std::getenv("BLENDER_PATH"))
        if (existsFile(env)) return env;

    if (const char* pathEnv = std::getenv("PATH"))
    {
        std::string pathStr = pathEnv;
        size_t start = 0;
        while (start <= pathStr.size())
        {
            size_t end = pathStr.find(';', start);
            if (end == std::string::npos) end = pathStr.size();
            std::string dir = pathStr.substr(start, end - start);
            if (!dir.empty())
            {
                fs::path candExe = fs::path(dir) / "blender.exe";
                fs::path candNoExt = fs::path(dir) / "blender";
                if (existsFile(candExe.string()))
                    return candExe.string();
                if (existsFile(candNoExt.string()))
                    return candNoExt.string();
            }
            start = end + 1;
        }
    }

    const std::vector<std::string> commonDirs = {
        "C:/Program Files/Blender Foundation/Blender/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.1/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
        "E:/Blender/blender.exe"
    };
    for (const auto& path : commonDirs)
        if (existsFile(path))
            return path;

    return {};
}

std::string convertUsdToGlbWithBlender(const std::string& usdPath)
{
    namespace fs = std::filesystem;
    fs::path input = fs::absolute(usdPath);
    fs::path output = input.parent_path() / (input.stem().string() + ".glb");

    std::error_code ec;
    auto outTime = fs::last_write_time(output, ec);
    auto inTime = fs::last_write_time(input, ec);
    if (!ec && fs::exists(output) && outTime >= inTime)
        return output.string();

    std::string blenderPath = resolveBlenderExecutable();
    if (blenderPath.empty())
        return {};

    fs::path scriptPath = fs::temp_directory_path() / "usd_to_glb.py";
    {
        std::ofstream scriptFile(scriptPath);
        scriptFile << R"PY(
import sys, bpy
args = sys.argv[sys.argv.index('--')+1:]
input_usd, output_glb = args[0], args[1]
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.wm.usd_import(filepath=input_usd)
bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB', export_yup=True, export_apply=True, export_lights=True, export_cameras=True, export_extras=True)
)PY";
    }

    auto normalizePath = [](std::string path) {
        std::replace(path.begin(), path.end(), '\\', '/');
        return path;
    };

    std::ostringstream cmd;
    cmd << "cmd /C \"\"" << normalizePath(blenderPath) << "\" -b -noaudio -P \""
        << normalizePath(scriptPath.string()) << "\" -- \"" << normalizePath(input.string())
        << "\" \"" << normalizePath(output.string()) << "\"\"";

    int rc = std::system(cmd.str().c_str());
    if (rc != 0 || !fs::exists(output))
        return {};
    return output.string();
}

std::string resolveScenePath(const std::string& path)
{
    auto tryResolve = [](const std::filesystem::path& candidate) -> std::string {
        if (std::filesystem::exists(candidate))
            return candidate.lexically_normal().string();

        static const std::vector<std::string> kExts = {".fbx", ".obj", ".gltf", ".glb", ".yaml", ".yml", ".usd", ".usda", ".usdc", ".usdz"};
        std::filesystem::path stem = candidate;
        stem.replace_extension();
        for (const auto& ext : kExts)
        {
            std::filesystem::path alt = stem;
            alt += ext;
            if (std::filesystem::exists(alt))
                return alt.lexically_normal().string();
        }
        return {};
    };

    for (const auto& candidate : { std::filesystem::path(path), std::filesystem::path("..") / path, std::filesystem::path("../..") / path })
    {
        std::string resolved = tryResolve(candidate);
        if (!resolved.empty())
            return resolved;
    }
    return {};
}
}

class SceneImporter::Impl
{
public:
    AssimpSceneImporter assimpImporter;
    YamlSceneImporter yamlImporter;
    ScenePostProcessor postProcessor;
};

SceneImporter::SceneImporter()
    : m_impl(std::make_unique<Impl>())
{
}

SceneImporter::~SceneImporter() = default;

std::unique_ptr<Scene> SceneImporter::importScene(const std::string& path, std::string* resolvedPath) const
{
    std::string resolved = resolveScenePath(path);
    if (resolved.empty())
        throw std::runtime_error("Path not found: " + path);

    std::string loadPath = resolved;
    std::string ext = std::filesystem::path(resolved).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz")
    {
        loadPath = convertUsdToGlbWithBlender(resolved);
        if (loadPath.empty())
            throw std::runtime_error("USD->GLB conversion failed (Blender required).");
        ext = ".glb";
    }

    auto scene = std::make_unique<Scene>();
    bool ok = false;
    if (ext == ".yaml" || ext == ".yml")
        ok = m_impl->yamlImporter.importScene(*scene, loadPath);
    else
        ok = m_impl->assimpImporter.appendToScene(*scene, loadPath);

    if (!ok)
        return nullptr;

    scene->setLoadedPath(loadPath);
    m_impl->postProcessor.apply(*scene);
    if (resolvedPath)
        *resolvedPath = loadPath;
    return scene;
}
