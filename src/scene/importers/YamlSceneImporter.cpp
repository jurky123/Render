#include "YamlSceneImporter.h"

#include "AssimpSceneImporter.h"
#include "MaterialLoader.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>

bool YamlSceneImporter::importScene(Scene& scene, const std::string& yamlPath) const
{
    try
    {
        YAML::Node config = YAML::LoadFile(yamlPath);
        const std::string dir = std::filesystem::path(yamlPath).parent_path().string();
        scene.setLoadedPath(yamlPath);

        if (config["Cameras"] && config["Cameras"].IsSequence() && config["Cameras"].size() > 0)
        {
            const auto& camNode = config["Cameras"][0];
            CameraInfo camera;
            if (camNode["Eye"])
            {
                auto eye = camNode["Eye"].as<std::vector<float>>();
                if (eye.size() >= 3)
                    camera.eye = glm::vec3(eye[0], eye[1], eye[2]);
            }
            if (camNode["Target"])
            {
                auto target = camNode["Target"].as<std::vector<float>>();
                if (target.size() >= 3)
                    camera.target = glm::vec3(target[0], target[1], target[2]);
            }
            if (camNode["Up"])
            {
                auto up = camNode["Up"].as<std::vector<float>>();
                if (up.size() >= 3)
                    camera.up = glm::vec3(up[0], up[1], up[2]);
            }
            if (camNode["Fovy"])
                camera.fovy = camNode["Fovy"].as<float>();
            if (camNode["ZNear"])
                camera.zNear = camNode["ZNear"].as<float>();
            if (camNode["ZFar"])
                camera.zFar = camNode["ZFar"].as<float>();
            camera.valid = true;
            scene.setCamera(camera);
        }

        if (config["AmbientIntensity"])
        {
            auto ambient = config["AmbientIntensity"].as<std::vector<float>>();
            if (ambient.size() >= 3)
                scene.setAmbientIntensity(glm::vec3(ambient[0], ambient[1], ambient[2]));
        }

        if (config["Lights"])
        {
            for (const auto& lightNode : config["Lights"])
            {
                Light light{};
                const std::string type = lightNode["Type"].as<std::string>();
                if (type == "Directional")
                {
                    light.type = LightType::Directional;
                    auto dirValues = lightNode["Direction"].as<std::vector<float>>();
                    if (dirValues.size() >= 3)
                        light.direction = glm::normalize(glm::vec3(dirValues[0], dirValues[1], dirValues[2]));
                }
                else if (type == "Point")
                {
                    light.type = LightType::Point;
                    auto pos = lightNode["Position"].as<std::vector<float>>();
                    if (pos.size() >= 3)
                        light.position = glm::vec3(pos[0], pos[1], pos[2]);
                }

                auto intensity = lightNode["Intensity"].as<std::vector<float>>();
                if (intensity.size() >= 3)
                    light.intensity = glm::vec3(intensity[0], intensity[1], intensity[2]);
                scene.addLight(light);
            }
        }

        std::map<std::string, uint32_t> materialMap;
        MaterialLoader materialLoader;
        if (config["Materials"])
        {
            for (const auto& matNode : config["Materials"])
            {
                std::string name = matNode["Name"].as<std::string>();
                std::vector<std::string> matData;

                if (matNode["Diffuse"])
                {
                    matData.push_back("Diffuse");
                    std::stringstream ss;
                    auto diffuse = matNode["Diffuse"].as<std::vector<float>>();
                    ss << "[" << diffuse[0] << ", " << diffuse[1] << ", " << diffuse[2] << "]";
                    matData.push_back(ss.str());
                }

                if (matNode["Specular"])
                {
                    matData.push_back("Specular");
                    std::stringstream ss;
                    auto specular = matNode["Specular"].as<std::vector<float>>();
                    ss << "[";
                    for (size_t i = 0; i < specular.size(); ++i)
                    {
                        if (i > 0)
                            ss << ", ";
                        ss << specular[i];
                    }
                    ss << "]";
                    matData.push_back(ss.str());
                }

                materialMap[name] = materialLoader.loadYamlMaterial(scene, name, matData, dir);
            }
        }

        AssimpSceneImporter assimpImporter;
        auto appendModel = [&](const YAML::Node& modelNode) {
            std::string meshFile = modelNode["Mesh"].as<std::string>();
            std::filesystem::path meshPath = std::filesystem::path(dir) / meshFile;
            if (!std::filesystem::exists(meshPath))
                return;

            size_t meshCountBefore = scene.meshes().size();
            assimpImporter.appendToScene(scene, meshPath.string());
            size_t meshCountAfter = scene.meshes().size();

            if (modelNode["Material"])
            {
                std::string matName = modelNode["Material"].as<std::string>();
                auto it = materialMap.find(matName);
                if (it != materialMap.end())
                {
                    for (size_t i = meshCountBefore; i < meshCountAfter; ++i)
                        scene.meshes()[i].materialIndex = it->second;
                }
            }
        };

        if (config["Models"])
        {
            for (const auto& modelNode : config["Models"])
                appendModel(modelNode);
        }

        if (config["ComplexModels"])
        {
            for (const auto& modelNode : config["ComplexModels"])
                appendModel(modelNode);
        }

        scene.rebuildStats();
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Scene] YAML parsing error: " << e.what() << "\n";
        return false;
    }
}
