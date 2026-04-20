#include "ScenePostProcessor.h"

#include "Scene.h"

#include <glm/glm.hpp>

#include <iostream>
#include <limits>

void ScenePostProcessor::apply(Scene& scene) const
{
    scene.rebuildStats();

    if (scene.lights().empty() && !scene.environment().valid)
    {
        Light l;
        l.type = LightType::Directional;
        l.direction = glm::normalize(glm::vec3(-0.3f, -1.f, -0.4f));
        l.intensity = glm::vec3(3.f);
        scene.addLight(l);
        scene.setAmbientIntensity(glm::vec3(0.15f));
        std::cout << "[Scene] No lights in scene; added default directional light and boosted ambient\n";
    }

    if (!scene.camera().valid && !scene.meshes().empty())
    {
        glm::vec3 minP(std::numeric_limits<float>::max());
        glm::vec3 maxP(std::numeric_limits<float>::lowest());

        for (const auto& mesh : scene.meshes())
        {
            for (const auto& v : mesh.vertices)
            {
                minP = glm::min(minP, v.position);
                maxP = glm::max(maxP, v.position);
            }
        }

        glm::vec3 center = 0.5f * (minP + maxP);
        glm::vec3 extent = maxP - minP;
        float radius = glm::length(extent) * 0.5f;
        if (radius < 1e-3f)
            radius = 1.f;

        CameraInfo cam;
        cam.target = center;
        cam.eye = center + glm::vec3(0.f, radius * 0.8f + 1.f, radius * 2.5f);
        cam.up = glm::vec3(0.f, 1.f, 0.f);
        cam.fovy = 45.f;
        cam.zNear = 0.1f;
        cam.zFar = radius * 10.f + 10.f;
        cam.valid = true;
        scene.setCamera(cam);

        std::cout << "[Scene] No camera found; synthesized framing camera at eye=("
                  << cam.eye.x << ", " << cam.eye.y << ", " << cam.eye.z << ")\n";
    }

    scene.rebuildStats();
}
