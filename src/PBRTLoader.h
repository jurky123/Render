#pragma once

#include <string>

class Scene;

/**
 * @brief PBRT v4 Scene File Parser
 *
 * Token-based parser matching the official pbrt-v4 architecture:
 *   Character-based tokenizer  ->  Token dispatch  ->  Scene building
 *
 * Supports: trianglemesh, plymesh, bilinearmesh, sphere, disk, cylinder,
 *           loopsubdiv shapes; all transform types; materials with named
 *           texture resolution; area/point/distant/infinite lights; camera;
 *           Include/Import; ObjectBegin/End/Instance; named materials;
 *           named coordinate systems.
 */
class PBRTLoader
{
public:
    static bool load(const std::string& pbrtPath, Scene& scene);
};
