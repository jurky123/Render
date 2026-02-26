#include "PBRTv4Parser.h"
#include "Scene.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace pbrt {

// ============================================================================
// ParameterDictionary Implementation
// ============================================================================

void ParameterDictionary::AddFloat(const std::string& name, const std::vector<float>& values) {
    floats[name] = values;
}

void ParameterDictionary::AddInt(const std::string& name, const std::vector<int>& values) {
    ints[name] = values;
}

void ParameterDictionary::AddBool(const std::string& name, const std::vector<bool>& values) {
    bools[name] = values;
}

void ParameterDictionary::AddString(const std::string& name, const std::vector<std::string>& values) {
    strings[name] = values;
}

void ParameterDictionary::AddPoint3(const std::string& name, const std::vector<glm::vec3>& values) {
    point3s[name] = values;
}

void ParameterDictionary::AddVector3(const std::string& name, const std::vector<glm::vec3>& values) {
    vector3s[name] = values;
}

void ParameterDictionary::AddNormal3(const std::string& name, const std::vector<glm::vec3>& values) {
    vector3s[name] = values;  // Treat as vectors for now
}

void ParameterDictionary::AddRGB(const std::string& name, const std::vector<RGB>& values) {
    rgbs[name] = values;
}

void ParameterDictionary::AddSpectrum(const std::string& name, const std::vector<Spectrum>& values) {
    spectra[name] = values;
}

void ParameterDictionary::AddTexture(const std::string& name, const std::string& texname) {
    textures[name] = texname;
}

std::optional<float> ParameterDictionary::GetOneFloat(const std::string& name, float def) const {
    auto it = floats.find(name);
    if (it != floats.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<int> ParameterDictionary::GetOneInt(const std::string& name, int def) const {
    auto it = ints.find(name);
    if (it != ints.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<bool> ParameterDictionary::GetOneBool(const std::string& name, bool def) const {
    auto it = bools.find(name);
    if (it != bools.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<std::string> ParameterDictionary::GetOneString(const std::string& name, const std::string& def) const {
    auto it = strings.find(name);
    if (it != strings.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<glm::vec3> ParameterDictionary::GetOnePoint3(const std::string& name, const glm::vec3& def) const {
    auto it = point3s.find(name);
    if (it != point3s.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<glm::vec3> ParameterDictionary::GetOneVector3(const std::string& name, const glm::vec3& def) const {
    auto it = vector3s.find(name);
    if (it != vector3s.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<RGB> ParameterDictionary::GetOneRGB(const std::string& name, const RGB& def) const {
    auto it = rgbs.find(name);
    if (it != rgbs.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::optional<Spectrum> ParameterDictionary::GetOneSpectrum(const std::string& name, const Spectrum& def) const {
    auto it = spectra.find(name);
    if (it != spectra.end() && !it->second.empty()) return it->second[0];
    return def;
}

std::vector<float> ParameterDictionary::GetFloatArray(const std::string& name) const {
    auto it = floats.find(name);
    if (it != floats.end()) return it->second;
    return {};
}

std::vector<int> ParameterDictionary::GetIntArray(const std::string& name) const {
    auto it = ints.find(name);
    if (it != ints.end()) return it->second;
    return {};
}

std::vector<glm::vec3> ParameterDictionary::GetPoint3Array(const std::string& name) const {
    auto it = point3s.find(name);
    if (it != point3s.end()) return it->second;
    return {};
}

std::string ParameterDictionary::GetTexture(const std::string& name) const {
    auto it = textures.find(name);
    if (it != textures.end()) return it->second;
    return "";
}

// ============================================================================
// PBRTv4Parser Implementation
// ============================================================================

PBRTv4Parser::PBRTv4Parser() {
    scene = std::make_shared<Scene>();
    activeTransformBits = TransformSet();
}

bool PBRTv4Parser::ParseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    baseDir = fs::path(filename).parent_path().string();
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return ParseString(buffer.str(), filename);
}

bool PBRTv4Parser::ParseString(const std::string& content, const std::string& sourceFile) {
    auto tokens = Tokenize(content);
    
    auto it = tokens.begin();
    while (it != tokens.end()) {
        std::string cmd = it->text;
        ++it;
        
        // Transform directives
        if (cmd == "Identity") {
            Identity();
        } else if (cmd == "Translate") {
            float dx = std::stof((it++)->text);
            float dy = std::stof((it++)->text);
            float dz = std::stof((it++)->text);
            Translate(dx, dy, dz);
        } else if (cmd == "Scale") {
            float sx = std::stof((it++)->text);
            float sy = std::stof((it++)->text);
            float sz = std::stof((it++)->text);
            Scale(sx, sy, sz);
        } else if (cmd == "Rotate") {
            float angle = std::stof((it++)->text);
            float ax = std::stof((it++)->text);
            float ay = std::stof((it++)->text);
            float az = std::stof((it++)->text);
            Rotate(angle, ax, ay, az);
        } else if (cmd == "LookAt") {
            float ex = std::stof((it++)->text);
            float ey = std::stof((it++)->text);
            float ez = std::stof((it++)->text);
            float lx = std::stof((it++)->text);
            float ly = std::stof((it++)->text);
            float lz = std::stof((it++)->text);
            float ux = std::stof((it++)->text);
            float uy = std::stof((it++)->text);
            float uz = std::stof((it++)->text);
            LookAt(ex, ey, ez, lx, ly, lz, ux, uy, uz);
        } else if (cmd == "CoordinateSystem") {
            std::string name = (it++)->text;
            if (name.front() == '"') name = name.substr(1, name.size() - 2);
            CoordinateSystem(name);
        } else if (cmd == "CoordSysTransform") {
            std::string name = (it++)->text;
            if (name.front() == '"') name = name.substr(1, name.size() - 2);
            CoordSysTransform(name);
        } else if (cmd == "Transform") {
            std::vector<float> m;
            if (it->text == "[") {
                ++it;
                while (it != tokens.end() && it->text != "]") {
                    m.push_back(std::stof((it++)->text));
                }
                ++it;  // Skip "]"
            }
            Transform(m);
        } else if (cmd == "ConcatTransform") {
            std::vector<float> m;
            if (it->text == "[") {
                ++it;
                while (it != tokens.end() && it->text != "]") {
                    m.push_back(std::stof((it++)->text));
                }
                ++it;
            }
            ConcatTransform(m);
        }
        // Scene structure
        else if (cmd == "WorldBegin") {
            WorldBegin();
        } else if (cmd == "AttributeBegin") {
            AttributeBegin();
        } else if (cmd == "AttributeEnd") {
            AttributeEnd();
        } else if (cmd == "TransformBegin") {
            TransformBegin();
        } else if (cmd == "TransformEnd") {
            TransformEnd();
        }
        // Scene description
        else if (cmd == "Camera") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Camera(type, params);
        } else if (cmd == "Sampler") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Sampler(type, params);
        } else if (cmd == "Film") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Film(type, params);
        } else if (cmd == "Integrator") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Integrator(type, params);
        } else if (cmd == "Texture") {
            std::string name = (it++)->text;
            if (name.front() == '"') name = name.substr(1, name.size() - 2);
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            std::string textype = (it++)->text;
            if (textype.front() == '"') textype = textype.substr(1, textype.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Texture(name, type, textype, params);
        } else if (cmd == "Material") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Material(type, params);
        } else if (cmd == "MakeNamedMaterial") {
            std::string name = (it++)->text;
            if (name.front() == '"') name = name.substr(1, name.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            MakeNamedMaterial(name, params);
        } else if (cmd == "NamedMaterial") {
            std::string name = (it++)->text;
            if (name.front() == '"') name = name.substr(1, name.size() - 2);
            NamedMaterial(name);
        } else if (cmd == "LightSource") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            LightSource(type, params);
        } else if (cmd == "AreaLightSource") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            AreaLightSource(type, params);
        } else if (cmd == "Shape") {
            std::string type = (it++)->text;
            if (type.front() == '"') type = type.substr(1, type.size() - 2);
            auto params = ParseParameters(it, tokens.end());
            Shape(type, params);
        } else if (cmd == "ReverseOrientation") {
            ReverseOrientation();
        } else if (cmd == "Include") {
            std::string filename = (it++)->text;
            if (filename.front() == '"') filename = filename.substr(1, filename.size() - 2);
            Include(filename);
        }
    }
    
    return true;
}

// ============================================================================
// Tokenizer
// ============================================================================

std::vector<PBRTv4Parser::Token> PBRTv4Parser::Tokenize(const std::string& content) {
    std::vector<Token> tokens;
    size_t pos = 0;
    int line = 1;
    
    while (pos < content.size()) {
        // Skip whitespace
        while (pos < content.size() && std::isspace(content[pos])) {
            if (content[pos] == '\n') line++;
            pos++;
        }
        if (pos >= content.size()) break;
        
        // Skip comments
        if (content[pos] == '#') {
            while (pos < content.size() && content[pos] != '\n') pos++;
            continue;
        }
        
        // Quoted string
        if (content[pos] == '"') {
            size_t start = pos++;
            while (pos < content.size() && content[pos] != '"') {
                if (content[pos] == '\\') pos++;  // Escape
                pos++;
            }
            pos++;  // Skip closing quote
            tokens.push_back({content.substr(start, pos - start), line});
            continue;
        }
        
        // Brackets
        if (content[pos] == '[' || content[pos] == ']') {
            tokens.push_back({std::string(1, content[pos]), line});
            pos++;
            continue;
        }
        
        // Number or identifier
        size_t start = pos;
        while (pos < content.size() && !std::isspace(content[pos]) && 
               content[pos] != '[' && content[pos] != ']' && content[pos] != '#') {
            pos++;
        }
        tokens.push_back({content.substr(start, pos - start), line});
    }
    
    return tokens;
}

ParameterDictionary PBRTv4Parser::ParseParameters(std::vector<Token>::iterator& it, 
                                                   const std::vector<Token>::iterator& end) {
    ParameterDictionary params;
    
    while (it != end && it->text[0] == '"' && it->text.find(' ') != std::string::npos) {
        // Parse "type name" [values]
        std::string typeAndName = it->text.substr(1, it->text.size() - 2);
        ++it;
        
        size_t spacePos = typeAndName.find(' ');
        std::string type = typeAndName.substr(0, spacePos);
        std::string name = typeAndName.substr(spacePos + 1);
        
        // Parse values
        if (it == end) break;
        
        if (it->text == "[") {
            ++it;
            std::vector<std::string> values;
            while (it != end && it->text != "]") {
                values.push_back(it->text);
                ++it;
            }
            if (it != end) ++it;  // Skip "]"
            
            // Convert based on type
            if (type == "float") {
                std::vector<float> floats;
                for (const auto& v : values) floats.push_back(std::stof(v));
                params.AddFloat(name, floats);
            } else if (type == "integer") {
                std::vector<int> ints;
                for (const auto& v : values) ints.push_back(std::stoi(v));
                params.AddInt(name, ints);
            } else if (type == "bool") {
                std::vector<bool> bools;
                for (const auto& v : values) bools.push_back(v == "true" || v == "1");
                params.AddBool(name, bools);
            } else if (type == "string" || type == "texture") {
                std::vector<std::string> strings;
                for (auto v : values) {
                    if (v.front() == '"') v = v.substr(1, v.size() - 2);
                    strings.push_back(v);
                }
                if (type == "texture" && !strings.empty()) {
                    params.AddTexture(name, strings[0]);
                } else {
                    params.AddString(name, strings);
                }
            } else if (type == "point" || type == "point3" || type == "point2") {
                std::vector<glm::vec3> points;
                for (size_t i = 0; i + 2 < values.size(); i += 3) {
                    points.push_back(glm::vec3(std::stof(values[i]), std::stof(values[i+1]), std::stof(values[i+2])));
                }
                params.AddPoint3(name, points);
            } else if (type == "vector" || type == "vector3" || type == "normal" || type == "normal3") {
                std::vector<glm::vec3> vectors;
                for (size_t i = 0; i + 2 < values.size(); i += 3) {
                    vectors.push_back(glm::vec3(std::stof(values[i]), std::stof(values[i+1]), std::stof(values[i+2])));
                }
                params.AddVector3(name, vectors);
            } else if (type == "rgb" || type == "color") {
                std::vector<RGB> rgbs;
                for (size_t i = 0; i + 2 < values.size(); i += 3) {
                    rgbs.push_back(RGB(std::stof(values[i]), std::stof(values[i+1]), std::stof(values[i+2])));
                }
                params.AddRGB(name, rgbs);
            } else if (type == "spectrum") {
                // Simplified: treat as RGB for now
                if (values.size() == 3) {
                    RGB rgb(std::stof(values[0]), std::stof(values[1]), std::stof(values[2]));
                    params.AddSpectrum(name, {Spectrum::CreateRGBAlbedo(RGBColorSpace::sRGB, rgb)});
                } else {
                    // SPD file
                    params.AddString(name, values);
                }
            } else if (type == "blackbody") {
                if (!values.empty()) {
                    float temp = std::stof(values[0]);
                    params.AddSpectrum(name, {Spectrum::CreateBlackbody(temp)});
                }
            }
        } else {
            // Single value
            std::string value = it->text;
            ++it;
            
            if (type == "float") {
                params.AddFloat(name, {std::stof(value)});
            } else if (type == "integer") {
                params.AddInt(name, {std::stoi(value)});
            } else if (type == "bool") {
                params.AddBool(name, {value == "true" || value == "1"});
            } else if (type == "string") {
                if (value.front() == '"') value = value.substr(1, value.size() - 2);
                params.AddString(name, {value});
            }
        }
    }
    
    return params;
}

// ============================================================================
// Transform Directives
// ============================================================================

void PBRTv4Parser::Identity() {
    curState.ctm = glm::mat4(1.0f);
    curState.ctmInverse = glm::mat4(1.0f);
}

void PBRTv4Parser::Translate(float dx, float dy, float dz) {
    glm::mat4 tr = glm::translate(glm::mat4(1.0f), glm::vec3(dx, dy, dz));
    curState.ctm = curState.ctm * tr;
    curState.ctmInverse = glm::inverse(curState.ctm);
}

void PBRTv4Parser::Scale(float sx, float sy, float sz) {
    glm::mat4 sc = glm::scale(glm::mat4(1.0f), glm::vec3(sx, sy, sz));
    curState.ctm = curState.ctm * sc;
    curState.ctmInverse = glm::inverse(curState.ctm);
}

void PBRTv4Parser::Rotate(float angle, float ax, float ay, float az) {
    glm::mat4 rot = glm::rotate(glm::mat4(1.0f), glm::radians(angle), glm::vec3(ax, ay, az));
    curState.ctm = curState.ctm * rot;
    curState.ctmInverse = glm::inverse(curState.ctm);
}

void PBRTv4Parser::LookAt(float ex, float ey, float ez, float lx, float ly, float lz, 
                          float ux, float uy, float uz) {
    glm::vec3 eye(ex, ey, ez);
    glm::vec3 look(lx, ly, lz);
    glm::vec3 up(ux, uy, uz);
    // glm::lookAt returns world-to-camera, but pbrt needs camera-to-world
    glm::mat4 worldToCamera = glm::lookAt(eye, look, up);
    glm::mat4 cameraToWorld = glm::inverse(worldToCamera);
    curState.ctm = cameraToWorld;
    curState.ctmInverse = worldToCamera;
}

void PBRTv4Parser::CoordinateSystem(const std::string& name) {
    curState.namedCoordinateSystems[name] = curState.ctm;
}

void PBRTv4Parser::CoordSysTransform(const std::string& name) {
    auto it = curState.namedCoordinateSystems.find(name);
    if (it != curState.namedCoordinateSystems.end()) {
        curState.ctm = it->second;
        curState.ctmInverse = glm::inverse(curState.ctm);
    }
}

void PBRTv4Parser::Transform(const std::vector<float>& m) {
    if (m.size() != 16) return;
    glm::mat4 mat;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat[j][i] = m[i * 4 + j];
    curState.ctm = mat;
    curState.ctmInverse = glm::inverse(mat);
}

void PBRTv4Parser::ConcatTransform(const std::vector<float>& m) {
    if (m.size() != 16) return;
    glm::mat4 mat;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat[j][i] = m[i * 4 + j];
    curState.ctm = curState.ctm * mat;
    curState.ctmInverse = glm::inverse(curState.ctm);
}

void PBRTv4Parser::TransformTimes(float start, float end) {
    // Motion blur support (simplified)
}

void PBRTv4Parser::ActiveTransform(const std::string& which) {
    // Motion blur support (simplified)
}

// ============================================================================
// Scene Structure
// ============================================================================

void PBRTv4Parser::WorldBegin() {
    inWorldBlock = true;
    // Save camera transform
    cameraParams.worldToCamera = curState.ctm;
}

void PBRTv4Parser::AttributeBegin() {
    pushedStates.push(curState);
}

void PBRTv4Parser::AttributeEnd() {
    if (!pushedStates.empty()) {
        curState = pushedStates.top();
        pushedStates.pop();
    }
}

void PBRTv4Parser::TransformBegin() {
    pushedStates.push(curState);
}

void PBRTv4Parser::TransformEnd() {
    if (!pushedStates.empty()) {
        curState = pushedStates.top();
        pushedStates.pop();
    }
}

void PBRTv4Parser::ObjectBegin(const std::string& name) {
    // Object instancing (simplified)
}

void PBRTv4Parser::ObjectEnd() {
    // Object instancing (simplified)
}

void PBRTv4Parser::ObjectInstance(const std::string& name) {
    // Object instancing (simplified)
}

// ============================================================================
// Camera, Film, Integrator
// ============================================================================

void PBRTv4Parser::Camera(const std::string& type, const ParameterDictionary& params) {
    cameraParams.type = type;
    if (type == "perspective") {
        cameraParams.fov = params.GetOneFloat("fov", 90.0f).value();
    }
}

void PBRTv4Parser::Sampler(const std::string& type, const ParameterDictionary& params) {
    // Sampler settings (simplified)
}

void PBRTv4Parser::Film(const std::string& type, const ParameterDictionary& params) {
    filmParams.xresolution = params.GetOneInt("xresolution", 1280).value();
    filmParams.yresolution = params.GetOneInt("yresolution", 720).value();
    filmParams.filename = params.GetOneString("filename", "pbrt.exr").value();
}

void PBRTv4Parser::Integrator(const std::string& type, const ParameterDictionary& params) {
    // Integrator settings (simplified)
}

void PBRTv4Parser::Accelerator(const std::string& type, const ParameterDictionary& params) {
    // Accelerator settings (simplified)
}

// ============================================================================
// Texture
// ============================================================================

void PBRTv4Parser::Texture(const std::string& name, const std::string& type, 
                           const std::string& textype, const ParameterDictionary& params) {
    if (type == "float") {
        std::shared_ptr<FloatTexture> tex;
        
        if (textype == "constant") {
            float value = params.GetOneFloat("value", 1.0f).value();
            tex = std::make_shared<FloatConstantTexture>(value);
        } else if (textype == "imagemap") {
            std::string filename = params.GetOneString("filename", "").value();
            if (!filename.empty()) {
                if (!baseDir.empty() && !fs::path(filename).is_absolute()) {
                    filename = (fs::path(baseDir) / filename).string();
                }
                auto mapping = std::make_shared<UVMapping>();
                tex = std::make_shared<FloatImageTexture>(mapping, filename);
            }
        } else if (textype == "scale") {
            std::string tex1name = params.GetTexture("tex1");
            float scale = params.GetOneFloat("scale", 1.0f).value();
            auto tex1 = GetFloatTexture(tex1name, 1.0f);
            tex = std::make_shared<FloatScaledTexture>(tex1, scale);
        }
        
        if (tex) namedFloatTextures[name] = tex;
        
    } else if (type == "spectrum" || type == "color") {
        std::shared_ptr<SpectrumTexture> tex;
        
        if (textype == "constant") {
            RGB rgb = params.GetOneRGB("value", RGB(1, 1, 1)).value();
            Spectrum spec = Spectrum::CreateRGBAlbedo(RGBColorSpace::sRGB, rgb);
            tex = std::make_shared<SpectrumConstantTexture>(spec);
        } else if (textype == "imagemap") {
            std::string filename = params.GetOneString("filename", "").value();
            if (!filename.empty()) {
                if (!baseDir.empty() && !fs::path(filename).is_absolute()) {
                    filename = (fs::path(baseDir) / filename).string();
                }
                auto mapping = std::make_shared<UVMapping>();
                tex = std::make_shared<SpectrumImageTexture>(mapping, filename);
            }
        }
        
        if (tex) namedSpectrumTextures[name] = tex;
    }
}

// ============================================================================
// Material
// ============================================================================

void PBRTv4Parser::Material(const std::string& type, const ParameterDictionary& params) {
    auto mat = CreateMaterial(type, params);
    if (mat) {
        curState.currentMaterial = ""; // Store inline material
        // For simplicity, add to named materials with unique name
        std::string uniqueName = "__inline_" + std::to_string(namedMaterials.size());
        namedMaterials[uniqueName] = mat;
        curState.currentMaterial = uniqueName;
    }
}

void PBRTv4Parser::MakeNamedMaterial(const std::string& name, const ParameterDictionary& params) {
    std::string type = params.GetOneString("type", "diffuse").value();
    auto mat = CreateMaterial(type, params);
    if (mat) namedMaterials[name] = mat;
}

void PBRTv4Parser::NamedMaterial(const std::string& name) {
    curState.currentMaterial = name;
}

// ============================================================================
// Light Sources
// ============================================================================

void PBRTv4Parser::LightSource(const std::string& type, const ParameterDictionary& params) {
    auto light = CreateLight(type, params);
    if (light && scene) {
        // Add light to scene (need to extend Scene class)
        // For now, store in a temporary list
    }
}

void PBRTv4Parser::AreaLightSource(const std::string& type, const ParameterDictionary& params) {
    curState.currentAreaLight = type;
    // Area light will be attached to next shape
}

// ============================================================================
// Shape
// ============================================================================

void PBRTv4Parser::Shape(const std::string& type, const ParameterDictionary& params) {
    CreateShapePrimitive(type, params);
}

void PBRTv4Parser::ReverseOrientation() {
    curState.reverseOrientation = !curState.reverseOrientation;
}

// ============================================================================
// Include
// ============================================================================

void PBRTv4Parser::Include(const std::string& filename) {
    std::string fullPath = filename;
    if (!baseDir.empty() && !fs::path(filename).is_absolute()) {
        fullPath = (fs::path(baseDir) / filename).string();
    }
    
    std::string savedDir = baseDir;
    baseDir = fs::path(fullPath).parent_path().string();
    
    ParseFile(fullPath);
    
    baseDir = savedDir;
}

// ============================================================================
// Helper Methods
// ============================================================================

std::shared_ptr<FloatTexture> PBRTv4Parser::GetFloatTexture(const std::string& name, float defaultValue) {
    auto it = namedFloatTextures.find(name);
    if (it != namedFloatTextures.end()) return it->second;
    return std::make_shared<FloatConstantTexture>(defaultValue);
}

std::shared_ptr<SpectrumTexture> PBRTv4Parser::GetSpectrumTexture(const std::string& name, const Spectrum& defaultValue) {
    auto it = namedSpectrumTextures.find(name);
    if (it != namedSpectrumTextures.end()) return it->second;
    return std::make_shared<SpectrumConstantTexture>(defaultValue);
}

std::shared_ptr<FloatTexture> PBRTv4Parser::GetFloatTextureOrDefault(const ParameterDictionary& params, 
                                                                      const std::string& name, float defaultValue) {
    std::string texName = params.GetTexture(name);
    if (!texName.empty()) return GetFloatTexture(texName, defaultValue);
    float value = params.GetOneFloat(name, defaultValue).value();
    return std::make_shared<FloatConstantTexture>(value);
}

std::shared_ptr<SpectrumTexture> PBRTv4Parser::GetSpectrumTextureOrDefault(const ParameterDictionary& params,
                                                                            const std::string& name, const Spectrum& defaultValue) {
    std::string texName = params.GetTexture(name);
    if (!texName.empty()) return GetSpectrumTexture(texName, defaultValue);
    
    auto spec = params.GetOneSpectrum(name, Spectrum());
    if (spec.has_value()) return std::make_shared<SpectrumConstantTexture>(spec.value());
    
    auto rgb = params.GetOneRGB(name, RGB());
    if (rgb.has_value()) {
        Spectrum s = Spectrum::CreateRGBAlbedo(RGBColorSpace::sRGB, rgb.value());
        return std::make_shared<SpectrumConstantTexture>(s);
    }
    
    return std::make_shared<SpectrumConstantTexture>(defaultValue);
}

Spectrum PBRTv4Parser::GetSpectrumParameter(const ParameterDictionary& params, 
                                            const std::string& name, const Spectrum& defaultValue) {
    auto spec = params.GetOneSpectrum(name, Spectrum());
    if (spec.has_value()) return spec.value();
    
    auto rgb = params.GetOneRGB(name, RGB());
    if (rgb.has_value()) return Spectrum::CreateRGBAlbedo(RGBColorSpace::sRGB, rgb.value());
    
    return defaultValue;
}

// ============================================================================
// Factory Methods
// ============================================================================

std::shared_ptr<::pbrt::Material> PBRTv4Parser::CreateMaterial(const std::string& type, const ParameterDictionary& params) {
    if (type == "diffuse") {
        auto reflectance = GetSpectrumTextureOrDefault(params, "reflectance", Spectrum::CreateConstant(0.5f));
        auto sigma = GetFloatTextureOrDefault(params, "sigma", 0.0f);
        return CreateDiffuseMaterial(reflectance, sigma);
        
    } else if (type == "conductor") {
        auto eta = GetSpectrumTextureOrDefault(params, "eta", Spectrum::CreateConstant(1.0f));
        auto k = GetSpectrumTextureOrDefault(params, "k", Spectrum::CreateConstant(1.0f));
        auto uroughness = GetFloatTextureOrDefault(params, "uroughness", 0.01f);
        auto vroughness = GetFloatTextureOrDefault(params, "vroughness", 0.01f);
        bool remaproughness = params.GetOneBool("remaproughness", true).value();
        return CreateConductorMaterial(eta, k, uroughness, vroughness, remaproughness);
        
    } else if (type == "dielectric") {
        float eta = params.GetOneFloat("eta", 1.5f).value();
        return CreateDielectricMaterial(eta);
        
    } else if (type == "roughdielectric" || type == "thindielectric") {
        float eta = params.GetOneFloat("eta", 1.5f).value();
        auto uroughness = GetFloatTextureOrDefault(params, "uroughness", 0.01f);
        auto vroughness = GetFloatTextureOrDefault(params, "vroughness", 0.01f);
        bool remaproughness = params.GetOneBool("remaproughness", true).value();
        return CreateRoughDielectricMaterial(eta, uroughness, vroughness, remaproughness);
        
    } else if (type == "coateddiffuse") {
        auto reflectance = GetSpectrumTextureOrDefault(params, "reflectance", Spectrum::CreateConstant(0.5f));
        float interface_eta = params.GetOneFloat("interface.eta", 1.5f).value();
        auto interface_uroughness = GetFloatTextureOrDefault(params, "interface.uroughness", 0.0f);
        auto interface_vroughness = GetFloatTextureOrDefault(params, "interface.vroughness", 0.0f);
        bool remaproughness = params.GetOneBool("remaproughness", true).value();
        return CreateCoatedDiffuseMaterial(reflectance, interface_eta, interface_uroughness, interface_vroughness, remaproughness);
        
    } else if (type == "coatedconductor") {
        auto conductor_eta = GetSpectrumTextureOrDefault(params, "conductor.eta", Spectrum::CreateConstant(1.0f));
        auto conductor_k = GetSpectrumTextureOrDefault(params, "conductor.k", Spectrum::CreateConstant(1.0f));
        float interface_eta = params.GetOneFloat("interface.eta", 1.5f).value();
        auto conductor_uroughness = GetFloatTextureOrDefault(params, "conductor.uroughness", 0.01f);
        auto conductor_vroughness = GetFloatTextureOrDefault(params, "conductor.vroughness", 0.01f);
        auto interface_uroughness = GetFloatTextureOrDefault(params, "interface.uroughness", 0.0f);
        auto interface_vroughness = GetFloatTextureOrDefault(params, "interface.vroughness", 0.0f);
        bool remaproughness = params.GetOneBool("remaproughness", true).value();
        return CreateCoatedConductorMaterial(conductor_eta, conductor_k, interface_eta,
                                              conductor_uroughness, conductor_vroughness,
                                              interface_uroughness, interface_vroughness, remaproughness);
    }
    
    // Default: diffuse with 50% gray
    auto gray = std::make_shared<SpectrumConstantTexture>(Spectrum::CreateConstant(0.5f));
    return CreateDiffuseMaterial(gray);
}

std::shared_ptr<::pbrt::Light> PBRTv4Parser::CreateLight(const std::string& type, const ParameterDictionary& params) {
    if (type == "point") {
        glm::vec3 from = params.GetOnePoint3("from", glm::vec3(0, 0, 0)).value();
        RGB I = params.GetOneRGB("I", RGB(1, 1, 1)).value();
        float scale = params.GetOneFloat("scale", 1.0f).value();
        Spectrum spec = Spectrum::CreateRGBUnbounded(RGBColorSpace::sRGB, I);
        return CreatePointLight(from, spec, scale);
        
    } else if (type == "distant") {
        glm::vec3 from = params.GetOnePoint3("from", glm::vec3(0, 0, 0)).value();
        glm::vec3 to = params.GetOnePoint3("to", glm::vec3(0, 0, 1)).value();
        glm::vec3 dir = glm::normalize(to - from);
        RGB L = params.GetOneRGB("L", RGB(1, 1, 1)).value();
        float scale = params.GetOneFloat("scale", 1.0f).value();
        Spectrum spec = Spectrum::CreateRGBUnbounded(RGBColorSpace::sRGB, L);
        return CreateDistantLight(dir, spec, scale);
        
    } else if (type == "spot") {
        glm::vec3 from = params.GetOnePoint3("from", glm::vec3(0, 0, 0)).value();
        glm::vec3 to = params.GetOnePoint3("to", glm::vec3(0, 0, 1)).value();
        glm::vec3 dir = glm::normalize(to - from);
        RGB I = params.GetOneRGB("I", RGB(1, 1, 1)).value();
        float coneangle = params.GetOneFloat("coneangle", 30.0f).value();
        float conedeltaangle = params.GetOneFloat("conedeltaangle", 5.0f).value();
        float scale = params.GetOneFloat("scale", 1.0f).value();
        Spectrum spec = Spectrum::CreateRGBUnbounded(RGBColorSpace::sRGB, I);
        return CreateSpotLight(from, dir, spec, coneangle, conedeltaangle, scale);
        
    } else if (type == "infinite") {
        std::string mapname = params.GetOneString("filename", "").value();
        float scale = params.GetOneFloat("scale", 1.0f).value();
        
        if (!mapname.empty()) {
            if (!baseDir.empty() && !fs::path(mapname).is_absolute()) {
                mapname = (fs::path(baseDir) / mapname).string();
            }
            return CreateInfiniteLightFromTexture(mapname, scale);
        } else {
            RGB L = params.GetOneRGB("L", RGB(1, 1, 1)).value();
            Spectrum spec = Spectrum::CreateRGBUnbounded(RGBColorSpace::sRGB, L);
            return CreateInfiniteLight(spec, scale);
        }
    }
    
    return nullptr;
}

std::shared_ptr<::pbrt::Camera> PBRTv4Parser::CreateCamera(const std::string& type, const ParameterDictionary& params) {
    // Camera creation (simplified, returned to Application)
    return nullptr;
}

void PBRTv4Parser::CreateShapePrimitive(const std::string& type, const ParameterDictionary& params) {
    // Shape creation (simplified)
    // In full implementation, this would create geometry and add to scene
    // For now, just placeholder
}

} // namespace pbrt


