#pragma once

#include "Spectrum.h"
#include "Texture.h"
#include "PBRTv4Materials.h"
#include "PBRTv4Lights.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stack>
#include <optional>
#include <functional>

// Complete pbrt-v4 parser aligned with official implementation
// Handles all pbrt directives, graphics state, transforms, etc.

// Forward declaration - Scene is in global namespace
class Scene;

namespace pbrt {

// Forward declarations
struct Mesh;

// Camera structure (simplified, for parser communication with Application)
struct Camera {
    std::string type = "perspective";
    float fov = 90.0f;
    glm::mat4 worldToCamera = glm::mat4(1.0f);
    glm::vec3 position = glm::vec3(0, 0, 0);
    glm::vec3 target = glm::vec3(0, 0, 1);
    glm::vec3 up = glm::vec3(0, 1, 0);
};

// Parameter parsing helpers
class ParameterDictionary {
public:
    void AddFloat(const std::string& name, const std::vector<float>& values);
    void AddInt(const std::string& name, const std::vector<int>& values);
    void AddBool(const std::string& name, const std::vector<bool>& values);
    void AddString(const std::string& name, const std::vector<std::string>& values);
    void AddPoint3(const std::string& name, const std::vector<glm::vec3>& values);
    void AddVector3(const std::string& name, const std::vector<glm::vec3>& values);
    void AddNormal3(const std::string& name, const std::vector<glm::vec3>& values);
    void AddRGB(const std::string& name, const std::vector<RGB>& values);
    void AddSpectrum(const std::string& name, const std::vector<Spectrum>& values);
    void AddTexture(const std::string& name, const std::string& texname);
    
    std::optional<float> GetOneFloat(const std::string& name, float def = 0) const;
    std::optional<int> GetOneInt(const std::string& name, int def = 0) const;
    std::optional<bool> GetOneBool(const std::string& name, bool def = false) const;
    std::optional<std::string> GetOneString(const std::string& name, const std::string& def = "") const;
    std::optional<glm::vec3> GetOnePoint3(const std::string& name, const glm::vec3& def = glm::vec3(0)) const;
    std::optional<glm::vec3> GetOneVector3(const std::string& name, const glm::vec3& def = glm::vec3(0)) const;
    std::optional<RGB> GetOneRGB(const std::string& name, const RGB& def = RGB(0, 0, 0)) const;
    std::optional<Spectrum> GetOneSpectrum(const std::string& name, const Spectrum& def = Spectrum::CreateConstant(0)) const;
    
    std::vector<float> GetFloatArray(const std::string& name) const;
    std::vector<int> GetIntArray(const std::string& name) const;
    std::vector<glm::vec3> GetPoint3Array(const std::string& name) const;
    
    std::string GetTexture(const std::string& name) const;
    
private:
    std::unordered_map<std::string, std::vector<float>> floats;
    std::unordered_map<std::string, std::vector<int>> ints;
    std::unordered_map<std::string, std::vector<bool>> bools;
    std::unordered_map<std::string, std::vector<std::string>> strings;
    std::unordered_map<std::string, std::vector<glm::vec3>> point3s;
    std::unordered_map<std::string, std::vector<glm::vec3>> vector3s;
    std::unordered_map<std::string, std::vector<RGB>> rgbs;
    std::unordered_map<std::string, std::vector<Spectrum>> spectra;
    std::unordered_map<std::string, std::string> textures;
};

// Graphics state (active transform, material, etc.)
struct GraphicsState {
    glm::mat4 ctm = glm::mat4(1.0f);  // Current Transform Matrix
    glm::mat4 ctmInverse = glm::mat4(1.0f);
    
    std::string currentMaterial = "";
    std::string currentAreaLight = "";
    bool reverseOrientation = false;
    
    // Named coordinate systems
    std::unordered_map<std::string, glm::mat4> namedCoordinateSystems;
};

// Transform management
class TransformSet {
public:
    glm::mat4 t[2];  // [0] = start, [1] = end (for motion blur)
    
    TransformSet() { t[0] = t[1] = glm::mat4(1.0f); }
    TransformSet(const glm::mat4& m) { t[0] = t[1] = m; }
    
    bool IsAnimated() const { return t[0] != t[1]; }
    glm::mat4 Get(int index = 0) const { return t[index]; }
};

// PBRT v4 Parser
class PBRTv4Parser {
public:
    PBRTv4Parser();
    ~PBRTv4Parser() = default;
    
    // Main parsing entry
    bool ParseFile(const std::string& filename);
    bool ParseString(const std::string& content, const std::string& sourceFile = "<string>");
    
    // Access parsed scene
    std::shared_ptr<::Scene> GetScene() { return scene; }
    
    // Directive handlers (called by token parser)
    void Identity();
    void Translate(float dx, float dy, float dz);
    void Scale(float sx, float sy, float sz);
    void Rotate(float angle, float ax, float ay, float az);
    void LookAt(float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz);
    void CoordinateSystem(const std::string& name);
    void CoordSysTransform(const std::string& name);
    void Transform(const std::vector<float>& m);
    void ConcatTransform(const std::vector<float>& m);
    void TransformTimes(float start, float end);
    void ActiveTransform(const std::string& which);
    
    void Camera(const std::string& type, const ParameterDictionary& params);
    void Sampler(const std::string& type, const ParameterDictionary& params);
    void Film(const std::string& type, const ParameterDictionary& params);
    void Integrator(const std::string& type, const ParameterDictionary& params);
    void Accelerator(const std::string& type, const ParameterDictionary& params);
    
    void WorldBegin();
    void AttributeBegin();
    void AttributeEnd();
    void TransformBegin();
    void TransformEnd();
    void ObjectBegin(const std::string& name);
    void ObjectEnd();
    void ObjectInstance(const std::string& name);
    
    void Texture(const std::string& name, const std::string& type, const std::string& textype, const ParameterDictionary& params);
    void Material(const std::string& type, const ParameterDictionary& params);
    void MakeNamedMaterial(const std::string& name, const ParameterDictionary& params);
    void NamedMaterial(const std::string& name);
    
    void LightSource(const std::string& type, const ParameterDictionary& params);
    void AreaLightSource(const std::string& type, const ParameterDictionary& params);
    
    void Shape(const std::string& type, const ParameterDictionary& params);
    void ReverseOrientation();
    
    void Include(const std::string& filename);
    
private:
    // Tokenizer
    struct Token {
        std::string text;
        int line;
    };
    std::vector<Token> Tokenize(const std::string& content);
    ParameterDictionary ParseParameters(std::vector<Token>::iterator& it, const std::vector<Token>::iterator& end);
    
    // Helper methods
    std::shared_ptr<FloatTexture> GetFloatTexture(const std::string& name, float defaultValue);
    std::shared_ptr<SpectrumTexture> GetSpectrumTexture(const std::string& name, const Spectrum& defaultValue);
    std::shared_ptr<FloatTexture> GetFloatTextureOrDefault(const ParameterDictionary& params, const std::string& name, float defaultValue);
    std::shared_ptr<SpectrumTexture> GetSpectrumTextureOrDefault(const ParameterDictionary& params, const std::string& name, const Spectrum& defaultValue);
    
    Spectrum GetSpectrumParameter(const ParameterDictionary& params, const std::string& name, const Spectrum& defaultValue);
    
    std::shared_ptr<::pbrt::Material> CreateMaterial(const std::string& type, const ParameterDictionary& params);
    std::shared_ptr<::pbrt::Light> CreateLight(const std::string& type, const ParameterDictionary& params);
    std::shared_ptr<::pbrt::Camera> CreateCamera(const std::string& type, const ParameterDictionary& params);
    
    void CreateShapePrimitive(const std::string& type, const ParameterDictionary& params);
    
    // State
    std::shared_ptr<::Scene> scene;
    GraphicsState curState;
    std::stack<GraphicsState> pushedStates;
    std::stack<TransformSet> pushedTransforms;
    
    TransformSet activeTransformBits;  // For motion blur
    bool inWorldBlock = false;
    
    // Named entities
    std::unordered_map<std::string, std::shared_ptr<::pbrt::Material>> namedMaterials;
    std::unordered_map<std::string, std::shared_ptr<FloatTexture>> namedFloatTextures;
    std::unordered_map<std::string, std::shared_ptr<SpectrumTexture>> namedSpectrumTextures;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Mesh>>> namedObjects;
    
    // Camera state
    struct CameraParams {
        std::string type = "perspective";
        float fov = 90.0f;
        glm::mat4 worldToCamera = glm::mat4(1.0f);
    } cameraParams;
    
    // Film state
    struct FilmParams {
        int xresolution = 1280;
        int yresolution = 720;
        std::string filename = "pbrt.exr";
    } filmParams;
    
    // Base directory for relative paths
    std::string baseDir;
    
    // For progress reporting
    std::function<void(float, const std::string&)> progressCallback;
};

} // namespace pbrt
