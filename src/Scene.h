#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

#include <glm/glm.hpp>
#include "TextureLoader.h"

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

// Material type enum (matching shader definition)
enum class MaterialType {
    BasicPBR = 0,      // Legacy metallic-roughness
    Diffuse = 1,
    Conductor = 2,
    Dielectric = 3,
    RoughDielectric = 4,
    CoatedDiffuse = 5,
    CoatedConductor = 6,
    Subsurface = 7
};

struct Material
{
    std::string name;

    // Material type identifier
    MaterialType type = MaterialType::BasicPBR;

    // Legacy PBR parameters (for BasicPBR type)
    glm::vec3 baseColor    = {0.8f, 0.8f, 0.8f};
    float     metallic     = 0.0f;
    float     roughness    = 0.5f;
    float     ior          = 1.45f;      ///< index of refraction
    float     transmission = 0.0f;       ///< 0 = opaque, 1 = fully transmissive
    glm::vec3 emission     = {0.f, 0.f, 0.f};

    // pbrt-v4 material parameters
    float     sigma        = 0.0f;       ///< Oren-Nayar roughness for diffuse
    glm::vec3 eta          = {0.f, 0.f, 0.f};  ///< Conductor IOR (complex, real part)
    glm::vec3 k            = {0.f, 0.f, 0.f};  ///< Conductor extinction coefficient
    float     uroughness   = 0.1f;       ///< U-direction roughness
    float     vroughness   = 0.1f;       ///< V-direction roughness
    bool      remapRoughness = true;     ///< Whether to remap roughness values

    /* texture paths (empty = none) */
    std::string baseColorTexPath;
    std::string normalTexPath;
    std::string metallicRoughnessTexPath;
    std::string emissionTexPath;
    std::string transmissionTexPath;
    
    /* texture pixel data (loaded from files) */
    TextureData baseColorTexData;
    TextureData normalTexData;
    TextureData metallicRoughnessTexData;
    TextureData emissionTexData;
    TextureData transmissionTexData;
};

// ---------------------------------------------------------------------------
// Light
// ---------------------------------------------------------------------------

enum class LightType
{
    Point,
    Directional,
    Spot
};

struct Light
{
    LightType type;
    glm::vec3 position;      // For point lights
    glm::vec3 direction;     // For directional lights (normalized)
    glm::vec3 intensity;     // Color * brightness
    float     cosInner = 0.f; // Spot light inner cone cosine
    float     cosOuter = 0.f; // Spot light outer cone cosine
};

// ---------------------------------------------------------------------------
// Curve Types / Data
// ---------------------------------------------------------------------------

enum class CurveType
{
    Flat,
    Cylinder,
    Ribbon
};

struct CurveSegment
{
    glm::vec3 cp[4];
    float     width0 = 0.0f;
    float     width1 = 0.0f;
    CurveType type = CurveType::Flat;
    glm::vec3 n0 = {0.f, 1.f, 0.f};
    glm::vec3 n1 = {0.f, 1.f, 0.f};
    bool      hasNormals = false;
    uint32_t  materialIndex = 0;
};

// ---------------------------------------------------------------------------
// Environment Map
// ---------------------------------------------------------------------------

struct EnvironmentMap
{
    TextureData texData;
    std::vector<float> hdrPixels; // RGBA float, linear
    int         hdrWidth = 0;
    int         hdrHeight = 0;
    int         hdrChannels = 0;
    bool        isHDR = false;
    glm::vec3   scale = {1.f, 1.f, 1.f};
    glm::mat3   lightTransform = glm::mat3(1.f);  // Rotation transform for environment light
    bool        valid = false;
};

// ---------------------------------------------------------------------------
// Camera Info
// ---------------------------------------------------------------------------

struct CameraInfo
{
    glm::vec3 eye    = {0.f, 0.f, -5.f};
    glm::vec3 target = {0.f, 0.f, 0.f};
    glm::vec3 up     = {0.f, 1.f, 0.f};
    float     fovy   = 45.0f;  // Field of view in degrees
    float     zNear  = 0.1f;
    float     zFar   = 1000.0f;
    bool      valid  = false;  // Whether camera was loaded from file
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
 *   .obj, .fbx, .gltf/.glb, .dae (Collada), .3ds, .ply, .stl, etc.
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
    
    /**
     * Load meshes from a file and append to existing scene (without clearing).
     * @return true on success, false on failure.
     */
    bool loadAppend(const std::string& path);

    /**
     * Load a scene from a YAML configuration file.
     * YAML format supports:
     *   - Materials section with material definitions
     *   - Models section with model mesh files (e.g., cornell_box.yaml)
     *   - ComplexModels section with complex model files (e.g., breakfast_room.yaml)
     * @return true on success, false on failure (error printed to stderr).
     */
    bool loadFromYaml(const std::string& yamlPath);

    /** Clear all loaded data. */
    void clear();

    void addMesh(const Mesh& mesh);
    void addMaterial(const Material& material);
    void addLight(const Light& light);
    void addCurve(const CurveSegment& curve);
    void setCamera(const CameraInfo& camera);
    void setAmbientIntensity(const glm::vec3& ambient);
    void setEnvironmentMap(const TextureData& texData, const glm::vec3& scale, const glm::mat3& lightTransform = glm::mat3(1.f));
    void setEnvironmentMapHDR(const HDRTextureData& texData, const glm::vec3& scale, const glm::mat3& lightTransform = glm::mat3(1.f));

    bool empty() const { return m_meshes.empty() && m_curves.empty(); }

    const std::vector<Mesh>&     meshes()     const { return m_meshes;     }
    const std::vector<Material>& materials()  const { return m_materials;  }
    const std::vector<Light>&    lights()     const { return m_lights;     }
    const std::vector<CurveSegment>& curves()  const { return m_curves;    }
    const glm::vec3&             ambientInt() const { return m_ambientInt; }
    const SceneStats&            stats()      const { return m_stats;      }
    const EnvironmentMap&        environment() const { return m_environment; }
    const std::string&           loadedPath() const { return m_path;       }
    const CameraInfo&            camera()     const { return m_camera;     }

private:
    void processNode     (const aiScene* scene, const aiNode* node,
                          const glm::mat4& parentTransform);
    Mesh processMesh     (const aiScene* scene, const aiMesh* mesh,
                          const glm::mat4& transform);
    void loadMaterials   (const aiScene* scene, const std::string& directory);
    Material convertMaterial(const aiMaterial* aiMat,
                             const std::string& directory);

    /**
     * Parse material definition from YAML data and add to materials list.
     * @return the material index in m_materials
     */
    uint32_t loadYamlMaterial(const std::string& name,
                              const std::vector<std::string>& matData,
                              const std::string& sceneDir);

    std::vector<Mesh>     m_meshes;
    std::vector<Material> m_materials;
    std::vector<Light>    m_lights;
    std::vector<CurveSegment> m_curves;
    glm::vec3             m_ambientInt = {0.05f, 0.05f, 0.05f};
    CameraInfo            m_camera;       // Camera info from YAML (if available)
    EnvironmentMap        m_environment;  // Environment map (if any)
    SceneStats            m_stats;
    std::string           m_path;
};

// Declared in Scene.cpp
static glm::vec3 parseVec3(const std::string& str);
