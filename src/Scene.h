#pragma once

#include <string>
#include <vector>
#include <memory>

#include <glm/glm.hpp>

// Forward
struct aiScene;
struct aiMesh;
struct aiNode;
struct aiMaterial;

// ---------------------------------------------------------------------------
// Vertex
// ---------------------------------------------------------------------------

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
};

// ---------------------------------------------------------------------------
// Material (PBR metallic-roughness)
// ---------------------------------------------------------------------------

struct Material
{
    std::string name;

    glm::vec3 baseColor    = {0.8f, 0.8f, 0.8f};
    float     metallic     = 0.0f;
    float     roughness    = 0.5f;
    float     ior          = 1.45f;      ///< index of refraction
    float     transmission = 0.0f;       ///< 0 = opaque, 1 = fully transmissive
    glm::vec3 emission     = {0.f, 0.f, 0.f};

    /* texture paths (empty = none) */
    std::string baseColorTexPath;
    std::string normalTexPath;
    std::string metallicRoughnessTexPath;
    std::string emissionTexPath;
};

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

struct Mesh
{
    std::string           name;
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    uint32_t              materialIndex = 0;
};

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

struct SceneStats
{
    uint32_t meshCount     = 0;
    uint32_t triangleCount = 0;
    uint32_t materialCount = 0;
};

/**
 * @brief Holds all geometry and materials for a loaded scene.
 *
 * Supported file formats are those that Assimp can read, including:
 *   .obj, .fbx, .gltf/.glb, .dae (Collada), .3ds, .ply, .stl, …
 */
class Scene
{
public:
    Scene() = default;

    /**
     * Load a scene from @p path.
     * @return true on success, false on failure (error printed to stderr).
     */
    bool load(const std::string& path);

    /** Clear all loaded data. */
    void clear();

    bool empty() const { return m_meshes.empty(); }

    const std::vector<Mesh>&     meshes()    const { return m_meshes;    }
    const std::vector<Material>& materials() const { return m_materials; }
    const SceneStats&            stats()     const { return m_stats;     }
    const std::string&           loadedPath()const { return m_path;      }

private:
    void processNode     (const aiScene* scene, const aiNode* node,
                          const glm::mat4& parentTransform);
    Mesh processMesh     (const aiScene* scene, const aiMesh* mesh,
                          const glm::mat4& transform);
    void loadMaterials   (const aiScene* scene, const std::string& directory);
    Material convertMaterial(const aiMaterial* aiMat,
                             const std::string& directory);

    std::vector<Mesh>     m_meshes;
    std::vector<Material> m_materials;
    SceneStats            m_stats;
    std::string           m_path;
};
