// ============================================================================
// PBRTLoader.cpp -- Token-based PBRT v4 Parser
//
// Architecture follows the official pbrt-v4 parser:
//   1. Character-based Tokenizer producing tokens (quoted strings, brackets,
//      keywords/numbers). No line concept.
//   2. Token-based parse() dispatching on the first character of keyword.
//   3. parseParameters() reading "type name" [values...] pairs.
//   4. Callbacks building the Scene (materials, geometry, lights, camera).
//
// This avoids the fundamental flaw of the previous line-based parser which
// failed on multi-line statements, continuations, and bracket-spanning.
// ============================================================================

#include "PBRTLoader.h"
#include "Scene.h"
#include "TextureLoader.h"
#include "stb_image.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>
#include <chrono>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <stack>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace fs = std::filesystem;

// ============================================================================
// Token
// ============================================================================

struct PToken {
    std::string text;  // raw text; quoted strings include quotes
    std::string file;
    int line = 0;

    bool isQuotedString() const {
        return text.size() >= 2 && text.front() == '"' && text.back() == '"';
    }
    std::string dequoted() const {
        if (isQuotedString()) return text.substr(1, text.size() - 2);
        return text;
    }
};

// ============================================================================
// Tokenizer -- character-based, matching official pbrt-v4 Tokenizer::Next()
// ============================================================================

class PBRTTokenizer {
public:
    PBRTTokenizer(const std::string& content, const std::string& filename)
        : m_data(content), m_pos(0), m_filename(filename), m_line(1) {}

    std::optional<PToken> next() {
        while (m_pos < m_data.size()) {
            char ch = getChar();

            // Whitespace
            if (ch == ' ' || ch == '\t' || ch == '\r') continue;
            if (ch == '\n') { m_line++; continue; }

            int startLine = m_line;

            // Quoted string
            if (ch == '"') {
                std::string tok;
                tok += '"';
                while (m_pos < m_data.size()) {
                    char c = getChar();
                    if (c == '\n') { m_line++; break; } // unterminated
                    if (c == '\\') {
                        tok += c;
                        if (m_pos < m_data.size()) {
                            char c2 = getChar();
                            tok += c2;
                        }
                        continue;
                    }
                    tok += c;
                    if (c == '"') break;
                }
                return PToken{tok, m_filename, startLine};
            }

            // Brackets
            if (ch == '[' || ch == ']') {
                return PToken{std::string(1, ch), m_filename, startLine};
            }

            // Comment -- skip to EOL
            if (ch == '#') {
                while (m_pos < m_data.size()) {
                    char c = m_data[m_pos];
                    if (c == '\n' || c == '\r') break;
                    m_pos++;
                }
                continue;  // don't produce a token for comments
            }

            // Keyword or number: scan until whitespace, quote, bracket
            std::string tok;
            tok += ch;
            while (m_pos < m_data.size()) {
                char c = m_data[m_pos];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                    c == '"' || c == '[' || c == ']' || c == '#')
                    break;
                tok += c;
                m_pos++;
            }
            return PToken{tok, m_filename, startLine};
        }
        return std::nullopt;
    }

    const std::string& filename() const { return m_filename; }

private:
    char getChar() { return m_data[m_pos++]; }

    std::string m_data;
    size_t m_pos;
    std::string m_filename;
    int m_line;
};

// ============================================================================
// Parsed parameter: "type name" [values...]
// ============================================================================

struct ParsedParam {
    std::string type;    // e.g. "float", "rgb", "point3", "integer", "string", "bool", "spectrum", "texture", "normal", "normal3", "point", "point2", "vector3", "vector", "float2", "blackbody"
    std::string name;    // e.g. "radius", "P", "reflectance"
    std::vector<float> floats;
    std::vector<int> ints;
    std::vector<std::string> strings;
    std::vector<bool> bools;
};

// ============================================================================
// Utility
// ============================================================================

static float toFloat(const std::string& s) {
    try { return std::stof(s); } catch (...) { return 0.f; }
}

static int toInt(const std::string& s) {
    try { return std::stoi(s); } catch (...) { return 0; }
}

static double toDouble(const std::string& s) {
    try { return std::stod(s); } catch (...) { return 0.0; }
}

// Gzip helpers (defined below)
static std::string decompressGzip(const std::string& data, const std::string& path);

static std::string readFileToString(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string data = ss.str();
    if (fs::path(path).extension() == ".gz") {
        std::string decompressed = decompressGzip(data, path);
        if (!decompressed.empty()) return decompressed;
    }
    return data;
}

static bool hasGzipHeader(const std::string& data) {
    if (data.size() < 10) return false;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(data.data());
    return b[0] == 0x1f && b[1] == 0x8b && b[2] == 0x08;
}

static std::string decompressGzip(const std::string& data, const std::string& path) {
    if (!hasGzipHeader(data)) return {};

    const unsigned char* b = reinterpret_cast<const unsigned char*>(data.data());
    size_t pos = 10;
    unsigned char flags = b[3];

    // Extra field
    if (flags & 0x04) {
        if (pos + 2 > data.size()) return {};
        uint16_t xlen = static_cast<uint16_t>(b[pos] | (b[pos + 1] << 8));
        pos += 2 + xlen;
    }

    // Original filename
    if (flags & 0x08) {
        while (pos < data.size() && b[pos] != 0) pos++;
        if (pos < data.size()) pos++;
    }

    // Comment
    if (flags & 0x10) {
        while (pos < data.size() && b[pos] != 0) pos++;
        if (pos < data.size()) pos++;
    }

    // Header CRC16
    if (flags & 0x02) {
        if (pos + 2 > data.size()) return {};
        pos += 2;
    }

    if (pos + 8 > data.size()) return {};
    size_t compLen = data.size() - pos - 8; // exclude CRC32 + ISIZE

    int outLen = 0;
    const char* comp = data.data() + pos;
    char* out = stbi_zlib_decode_noheader_malloc(comp, static_cast<int>(compLen), &outLen);
    if (!out || outLen <= 0) {
        if (out) stbi_image_free(out);
        std::cerr << "[PBRTLoader] Failed to decompress gzip: " << path << "\n";
        return {};
    }

    std::string result(out, outLen);
    stbi_image_free(out);
    return result;
}

// ============================================================================
// Parse parameters: "type name" [values...]  or "type name" value
// Matches official pbrt-v4 parseParameters() exactly.
// ============================================================================

static std::vector<ParsedParam> parseParameters(
    std::function<std::optional<PToken>(bool)>& nextToken,
    std::function<void(const PToken&)>& ungetToken)
{
    std::vector<ParsedParam> params;

    while (true) {
        auto t = nextToken(false);
        if (!t.has_value()) return params;

        if (!t->isQuotedString()) {
            ungetToken(*t);
            return params;
        }

        ParsedParam param;
        std::string decl = t->dequoted();

        // Split declaration: "type name"
        size_t i = 0;
        while (i < decl.size() && (decl[i] == ' ' || decl[i] == '\t')) i++;
        size_t typeStart = i;
        while (i < decl.size() && decl[i] != ' ' && decl[i] != '\t') i++;
        param.type = decl.substr(typeStart, i - typeStart);
        while (i < decl.size() && (decl[i] == ' ' || decl[i] == '\t')) i++;
        size_t nameStart = i;
        while (i < decl.size() && decl[i] != ' ' && decl[i] != '\t') i++;
        param.name = decl.substr(nameStart, i - nameStart);

        if (param.name.empty()) {
            // Malformed declaration; skip
            continue;
        }

        // Upgrade old types
        if (param.type == "point") param.type = "point3";
        if (param.type == "vector") param.type = "vector3";
        if (param.type == "color") param.type = "rgb";
        if (param.type == "normal") param.type = "normal3";

        bool isInt = (param.type == "integer");

        // Read value(s)
        auto val = nextToken(true);
        if (!val.has_value()) break;

        auto addVal = [&](const PToken& tok) {
            if (tok.isQuotedString()) {
                std::string s = tok.dequoted();
                if (s == "true") param.bools.push_back(true);
                else if (s == "false") param.bools.push_back(false);
                else param.strings.push_back(s);
            } else if (tok.text == "true") {
                param.bools.push_back(true);
            } else if (tok.text == "false") {
                param.bools.push_back(false);
            } else {
                if (isInt)
                    param.ints.push_back(toInt(tok.text));
                else
                    param.floats.push_back(toFloat(tok.text));
            }
        };

        if (val->text == "[") {
            // Array: read until ']'
            while (true) {
                auto v = nextToken(true);
                if (!v.has_value()) break;
                if (v->text == "]") break;
                addVal(*v);
            }
        } else {
            addVal(*val);
        }

        // Handle "bool" as stringified boolean
        if (param.type == "bool" && !param.strings.empty()) {
            for (auto& s : param.strings) {
                if (s == "true") param.bools.push_back(true);
                else if (s == "false") param.bools.push_back(false);
            }
            param.strings.clear();
        }

        params.push_back(std::move(param));
    }
    return params;
}

// ============================================================================
// Parameter lookup helpers
// ============================================================================

static const ParsedParam* findParam(const std::vector<ParsedParam>& params,
                                     const std::string& name) {
    for (auto& p : params)
        if (p.name == name) return &p;
    return nullptr;
}

static float getFloat(const std::vector<ParsedParam>& params,
                       const std::string& name, float fb) {
    auto* p = findParam(params, name);
    if (p && !p->floats.empty()) return p->floats[0];
    return fb;
}

static int getInt(const std::vector<ParsedParam>& params,
                   const std::string& name, int fb) {
    auto* p = findParam(params, name);
    if (p && !p->ints.empty()) return p->ints[0];
    if (p && !p->floats.empty()) return static_cast<int>(p->floats[0]);
    return fb;
}

static glm::vec3 getRgb(const std::vector<ParsedParam>& params,
                          const std::string& name, const glm::vec3& fb) {
    auto* p = findParam(params, name);
    if (!p) return fb;

    auto blackbodyToRgb = [](float kelvin) {
        // Simple blackbody approximation (Kelvin -> RGB in 0..1)
        float t = kelvin / 100.0f;
        float r, g, b;
        if (t <= 66.0f) {
            r = 1.0f;
            g = std::clamp(0.39008157876901960784f * logf(t) - 0.63184144378862745098f, 0.f, 1.f);
            if (t <= 19.0f) b = 0.0f;
            else b = std::clamp(0.54320678911019607843f * logf(t - 10.f) - 1.19625408914f, 0.f, 1.f);
        } else {
            r = std::clamp(1.29293618606274509804f * powf(t - 60.f, -0.1332047592f), 0.f, 1.f);
            g = std::clamp(1.12989086089529411765f * powf(t - 60.f, -0.0755148492f), 0.f, 1.f);
            b = 1.0f;
        }
        return glm::vec3(r, g, b);
    };

    if (p->type == "blackbody" && !p->floats.empty()) {
        float temp = p->floats[0];
        float scale = (p->floats.size() > 1) ? p->floats[1] : 1.f;
        return blackbodyToRgb(temp) * scale;
    }

    if (p->floats.size() >= 3)
        return {p->floats[0], p->floats[1], p->floats[2]};
    if (p->floats.size() == 1)
        return glm::vec3(p->floats[0]);

    if (p->type == "spectrum" && !p->strings.empty()) {
        // Named spectrum/texture reference
        return fb;
    }

    return fb;
}

static std::string getString(const std::vector<ParsedParam>& params,
                              const std::string& name, const std::string& fb = "") {
    auto* p = findParam(params, name);
    if (p && !p->strings.empty()) return p->strings[0];
    return fb;
}

static std::vector<float> getFloats(const std::vector<ParsedParam>& params,
                                     const std::string& name) {
    auto* p = findParam(params, name);
    if (p) return p->floats;
    return {};
}

static std::vector<int> getInts(const std::vector<ParsedParam>& params,
                                 const std::string& name) {
    auto* p = findParam(params, name);
    if (p) return p->ints;
    return {};
}

static std::vector<glm::vec3> getVec3Array(const std::vector<ParsedParam>& params,
                                            const std::string& name) {
    auto* p = findParam(params, name);
    if (!p || p->floats.size() < 3) return {};
    std::vector<glm::vec3> out;
    out.reserve(p->floats.size() / 3);
    for (size_t i = 0; i + 2 < p->floats.size(); i += 3)
        out.emplace_back(p->floats[i], p->floats[i + 1], p->floats[i + 2]);
    return out;
}

// ============================================================================
// Mesh helper functions
// ============================================================================

static void computeNormals(Mesh& mesh) {
    if (mesh.indices.size() % 3 != 0) return;
    for (auto& v : mesh.vertices) v.normal = glm::vec3(0.f);
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        uint32_t i0 = mesh.indices[i], i1 = mesh.indices[i+1], i2 = mesh.indices[i+2];
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size())
            continue;
        glm::vec3 e1 = mesh.vertices[i1].position - mesh.vertices[i0].position;
        glm::vec3 e2 = mesh.vertices[i2].position - mesh.vertices[i0].position;
        glm::vec3 n = glm::cross(e1, e2);
        float len = glm::length(n);
        if (len > 1e-10f) n /= len;
        mesh.vertices[i0].normal += n;
        mesh.vertices[i1].normal += n;
        mesh.vertices[i2].normal += n;
    }
    for (auto& v : mesh.vertices) {
        float len = glm::length(v.normal);
        if (len > 1e-10f) v.normal /= len;
        else v.normal = glm::vec3(0.f, 1.f, 0.f);
    }
}

static void applyTransformToMesh(Mesh& mesh, const glm::mat4& transform) {
    const glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(transform)));
    for (auto& v : mesh.vertices) {
        v.position = glm::vec3(transform * glm::vec4(v.position, 1.f));
        v.normal = glm::normalize(normalMat * v.normal);
    }
}

static Mesh makeSphereMesh(float radius, int stacks = 16, int slices = 32) {
    Mesh mesh;
    mesh.name = "sphere";
    const float PI = 3.14159265358979323846f;
    for (int i = 0; i <= stacks; ++i) {
        float phi = PI * i / stacks;
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.f * PI * j / slices;
            Vertex v{};
            v.normal = glm::vec3(sinf(phi)*cosf(theta), cosf(phi), sinf(phi)*sinf(theta));
            v.position = v.normal * radius;
            v.texCoord = glm::vec2((float)j / slices, (float)i / stacks);
            mesh.vertices.push_back(v);
        }
    }
    for (int i = 0; i < stacks; ++i)
        for (int j = 0; j < slices; ++j) {
            uint32_t a = i * (slices + 1) + j, b = a + slices + 1;
            mesh.indices.insert(mesh.indices.end(), {a, b, a+1, a+1, b, b+1});
        }
    return mesh;
}

static Mesh makeDiskMesh(float radius, float height, int segments = 32) {
    Mesh mesh;
    mesh.name = "disk";
    const float PI = 3.14159265358979323846f;
    Vertex centre{};
    centre.position = glm::vec3(0.f, 0.f, height);
    centre.normal = glm::vec3(0.f, 0.f, 1.f);
    centre.texCoord = glm::vec2(0.5f, 0.5f);
    mesh.vertices.push_back(centre);
    for (int i = 0; i <= segments; ++i) {
        float t = 2.f * PI * i / segments;
        Vertex v{};
        v.position = glm::vec3(radius * cosf(t), radius * sinf(t), height);
        v.normal = glm::vec3(0.f, 0.f, 1.f);
        v.texCoord = glm::vec2(0.5f + 0.5f * cosf(t), 0.5f + 0.5f * sinf(t));
        mesh.vertices.push_back(v);
    }
    for (int i = 0; i < segments; ++i)
        mesh.indices.insert(mesh.indices.end(), {0u, (uint32_t)(i+1), (uint32_t)(i+2)});
    return mesh;
}

static Mesh makeCylinderMesh(float radius, float zmin, float zmax, int segments = 32) {
    Mesh mesh;
    mesh.name = "cylinder";
    const float PI = 3.14159265358979323846f;
    for (int i = 0; i <= segments; ++i) {
        float t = 2.f * PI * i / segments;
        float cx = cosf(t), cy = sinf(t);
        Vertex vb{}, vt{};
        vb.position = glm::vec3(radius*cx, radius*cy, zmin);
        vb.normal = glm::vec3(cx, cy, 0.f);
        vt.position = glm::vec3(radius*cx, radius*cy, zmax);
        vt.normal = glm::vec3(cx, cy, 0.f);
        mesh.vertices.push_back(vb);
        mesh.vertices.push_back(vt);
    }
    for (int i = 0; i < segments; ++i) {
        uint32_t a = i*2, b = i*2+1, c = (i+1)*2, d = (i+1)*2+1;
        mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
    }
    return mesh;
}

// ============================================================================
// Curve helper functions (Bezier/BSpline conversions)
// ============================================================================

static std::array<glm::vec3, 4> elevateQuadraticBezierToCubic(const std::array<glm::vec3, 3>& cp) {
    return {
        cp[0],
        glm::mix(cp[0], cp[1], 2.f / 3.f),
        glm::mix(cp[1], cp[2], 1.f / 3.f),
        cp[2]
    };
}

static std::array<glm::vec3, 3> quadraticBSplineToBezier(const std::array<glm::vec3, 3>& cp) {
    glm::vec3 p11 = glm::mix(cp[0], cp[1], 0.5f);
    glm::vec3 p22 = glm::mix(cp[1], cp[2], 0.5f);
    return {p11, cp[1], p22};
}

static std::array<glm::vec3, 4> cubicBSplineToBezier(const std::array<glm::vec3, 4>& cp) {
    glm::vec3 p122 = glm::mix(cp[0], cp[1], 2.f / 3.f);
    glm::vec3 p223 = glm::mix(cp[1], cp[2], 1.f / 3.f);
    glm::vec3 p233 = glm::mix(cp[1], cp[2], 2.f / 3.f);
    glm::vec3 p334 = glm::mix(cp[2], cp[3], 1.f / 3.f);
    glm::vec3 p222 = glm::mix(p122, p223, 0.5f);
    glm::vec3 p333 = glm::mix(p233, p334, 0.5f);
    return {p222, p223, p233, p333};
}

// ============================================================================
// Custom PLY loader (binary LE + ASCII)
// ============================================================================

static std::string trim(const std::string& str) {
    size_t s = str.find_first_not_of(" \t\r\n");
    if (s == std::string::npos) return "";
    size_t e = str.find_last_not_of(" \t\r\n");
    return str.substr(s, e - s + 1);
}

static size_t plyTypeSize(const std::string& t) {
    if (t == "float" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    if (t == "uchar" || t == "uint8" || t == "char" || t == "int8") return 1;
    if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
    if (t == "int" || t == "int32" || t == "uint" || t == "uint32") return 4;
    return 4;
}

static bool loadPlyDirect(const std::string& filePath,
                           const glm::mat4& transform,
                           uint32_t materialIndex,
                           std::vector<Mesh>& outMeshes) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[PBRTLoader] Cannot open PLY: " << filePath << "\n";
        return false;
    }

    struct PropInfo { std::string name, type, countType, valueType; bool isList = false; };
    struct ElemInfo { std::string name; size_t count = 0; std::vector<PropInfo> props; };
    std::vector<ElemInfo> elements;
    std::string format;
    std::string hline;

    if (!std::getline(file, hline) || trim(hline) != "ply") return false;

    while (std::getline(file, hline)) {
        std::string tl = trim(hline);
        if (tl == "end_header") break;
        if (tl.empty() || tl[0] == 'c') continue;
        std::istringstream iss(tl);
        std::string kw; iss >> kw;
        if (kw == "format") {
            iss >> format;
        } else if (kw == "element") {
            ElemInfo e; iss >> e.name >> e.count;
            elements.push_back(e);
        } else if (kw == "property" && !elements.empty()) {
            PropInfo p; std::string first; iss >> first;
            if (first == "list") {
                p.isList = true;
                iss >> p.countType >> p.valueType >> p.name;
            } else {
                p.type = first;
                iss >> p.name;
            }
            elements.back().props.push_back(p);
        }
    }

    const ElemInfo* vertexElem = nullptr;
    const ElemInfo* faceElem = nullptr;
    for (const auto& e : elements) {
        if (e.name == "vertex") vertexElem = &e;
        else if (e.name == "face") faceElem = &e;
    }
    if (!vertexElem || vertexElem->count == 0) return false;

    int xI = -1, yI = -1, zI = -1;
    int nxI = -1, nyI = -1, nzI = -1;
    int uI = -1, vI = -1;
    for (int i = 0; i < (int)vertexElem->props.size(); ++i) {
        const auto& nm = vertexElem->props[i].name;
        if (nm == "x") xI = i; else if (nm == "y") yI = i; else if (nm == "z") zI = i;
        else if (nm == "nx") nxI = i; else if (nm == "ny") nyI = i; else if (nm == "nz") nzI = i;
        else if (nm == "s" || nm == "u" || nm == "texture_u") uI = i;
        else if (nm == "t" || nm == "v" || nm == "texture_v") vI = i;
    }
    if (xI < 0 || yI < 0 || zI < 0) return false;
    const bool hasN = (nxI >= 0 && nyI >= 0 && nzI >= 0);

    Mesh mesh;
    mesh.name = fs::path(filePath).stem().string();
    mesh.materialIndex = materialIndex;
    mesh.vertices.resize(vertexElem->count);

    const bool isAscii = (format.find("ascii") != std::string::npos);
    const bool isBinaryLE = (format.find("binary_little_endian") != std::string::npos);

    for (const auto& elem : elements) {
        if (&elem == vertexElem) {
            if (isAscii) {
                for (size_t vi = 0; vi < elem.count; ++vi) {
                    std::string vline;
                    if (!std::getline(file, vline)) return false;
                    std::istringstream viss(vline);
                    std::vector<float> vals; float fv;
                    while (viss >> fv) vals.push_back(fv);
                    auto G = [&](int idx) -> float { return (idx >= 0 && idx < (int)vals.size()) ? vals[idx] : 0.f; };
                    mesh.vertices[vi].position = glm::vec3(G(xI), G(yI), G(zI));
                    if (hasN) mesh.vertices[vi].normal = glm::vec3(G(nxI), G(nyI), G(nzI));
                    if (uI >= 0) mesh.vertices[vi].texCoord.x = G(uI);
                    if (vI >= 0) mesh.vertices[vi].texCoord.y = G(vI);
                }
            } else if (isBinaryLE) {
                size_t stride = 0;
                std::vector<size_t> offsets;
                for (const auto& p : elem.props) { offsets.push_back(stride); stride += plyTypeSize(p.type); }
                std::vector<char> buf(stride * elem.count);
                file.read(buf.data(), (std::streamsize)buf.size());
                for (size_t vi = 0; vi < elem.count; ++vi) {
                    const char* ptr = buf.data() + vi * stride;
                    auto readF = [&](int idx) -> float {
                        if (idx < 0) return 0.f;
                        const char* p = ptr + offsets[idx];
                        const auto& tp = elem.props[idx].type;
                        if (tp == "float" || tp == "float32") { float v; memcpy(&v, p, 4); return v; }
                        if (tp == "double" || tp == "float64") { double v; memcpy(&v, p, 8); return (float)v; }
                        if (tp == "uchar" || tp == "uint8") { return (float)(*(const uint8_t*)p); }
                        return 0.f;
                    };
                    mesh.vertices[vi].position = glm::vec3(readF(xI), readF(yI), readF(zI));
                    if (hasN) mesh.vertices[vi].normal = glm::vec3(readF(nxI), readF(nyI), readF(nzI));
                    if (uI >= 0) mesh.vertices[vi].texCoord.x = readF(uI);
                    if (vI >= 0) mesh.vertices[vi].texCoord.y = readF(vI);
                }
            } else { return false; }
        } else if (&elem == faceElem) {
            if (isAscii) {
                for (size_t fi = 0; fi < elem.count; ++fi) {
                    std::string fline;
                    if (!std::getline(file, fline)) break;
                    std::istringstream fiss(fline);
                    int nv; fiss >> nv;
                    std::vector<uint32_t> idx(nv);
                    for (int j = 0; j < nv; ++j) fiss >> idx[j];
                    for (int j = 1; j + 1 < nv; ++j)
                        mesh.indices.insert(mesh.indices.end(), {idx[0], idx[j], idx[j+1]});
                }
            } else if (isBinaryLE) {
                const auto& listProp = elem.props[0];
                size_t csz = plyTypeSize(listProp.countType);
                size_t isz = plyTypeSize(listProp.valueType);
                size_t extraFixed = 0;
                for (size_t pi = 1; pi < elem.props.size(); ++pi)
                    if (!elem.props[pi].isList) extraFixed += plyTypeSize(elem.props[pi].type);

                for (size_t fi = 0; fi < elem.count; ++fi) {
                    uint32_t nv = 0;
                    if (csz == 1) { uint8_t c; file.read((char*)&c, 1); nv = c; }
                    else if (csz == 2) { uint16_t c; file.read((char*)&c, 2); nv = c; }
                    else { file.read((char*)&nv, 4); }

                    std::vector<uint32_t> idx(nv);
                    if (isz == 4) {
                        std::vector<int32_t> raw(nv);
                        file.read((char*)raw.data(), (std::streamsize)(nv * 4));
                        for (uint32_t j = 0; j < nv; ++j) idx[j] = (uint32_t)raw[j];
                    } else if (isz == 2) {
                        std::vector<uint16_t> raw(nv);
                        file.read((char*)raw.data(), (std::streamsize)(nv * 2));
                        for (uint32_t j = 0; j < nv; ++j) idx[j] = raw[j];
                    } else {
                        std::vector<uint8_t> raw(nv);
                        file.read((char*)raw.data(), (std::streamsize)(nv));
                        for (uint32_t j = 0; j < nv; ++j) idx[j] = raw[j];
                    }

                    for (uint32_t j = 1; j + 1 < nv; ++j)
                        mesh.indices.insert(mesh.indices.end(), {idx[0], idx[j], idx[j+1]});

                    if (extraFixed > 0)
                        file.seekg((std::streamoff)extraFixed, std::ios::cur);
                    for (size_t pi = 1; pi < elem.props.size(); ++pi) {
                        if (elem.props[pi].isList) {
                            uint32_t cnt = 0;
                            size_t cs2 = plyTypeSize(elem.props[pi].countType);
                            if (cs2 == 1) { uint8_t c; file.read((char*)&c, 1); cnt = c; }
                            else if (cs2 == 2) { uint16_t c; file.read((char*)&c, 2); cnt = c; }
                            else { file.read((char*)&cnt, 4); }
                            file.seekg((std::streamoff)(cnt * plyTypeSize(elem.props[pi].valueType)), std::ios::cur);
                        }
                    }
                }
            }
        }
        // OTHER elements: skip
        else {
            if (isAscii) {
                for (size_t ei = 0; ei < elem.count; ++ei) {
                    std::string skip; std::getline(file, skip);
                }
            } else if (isBinaryLE) {
                bool hasList = false;
                size_t fixedSz = 0;
                for (auto& p : elem.props) {
                    if (p.isList) { hasList = true; break; }
                    fixedSz += plyTypeSize(p.type);
                }
                if (!hasList) {
                    file.seekg((std::streamoff)(fixedSz * elem.count), std::ios::cur);
                } else {
                    for (size_t ei = 0; ei < elem.count; ++ei) {
                        for (auto& p : elem.props) {
                            if (p.isList) {
                                uint32_t cnt = 0;
                                size_t cs = plyTypeSize(p.countType);
                                if (cs == 1) { uint8_t c; file.read((char*)&c, 1); cnt = c; }
                                else if (cs == 2) { uint16_t c; file.read((char*)&c, 2); cnt = c; }
                                else { file.read((char*)&cnt, 4); }
                                file.seekg((std::streamoff)(cnt * plyTypeSize(p.valueType)), std::ios::cur);
                            } else {
                                file.seekg((std::streamoff)plyTypeSize(p.type), std::ios::cur);
                            }
                        }
                    }
                }
            }
        }
    }

    if (!hasN) computeNormals(mesh);
    applyTransformToMesh(mesh, transform);
    outMeshes.push_back(std::move(mesh));
    return true;
}

// ============================================================================
// Graphics state (matches pbrt-v4 BasicSceneBuilder graphics state)
// ============================================================================

struct GraphicsState {
    glm::mat4 transform = glm::mat4(1.0f);
    uint32_t materialIndex = 0;
    bool reverseOrientation = false;
    std::string areaLightType;
    glm::vec3 areaLightL = glm::vec3(0.f);
    float areaLightScale = 1.0f;
};

// ============================================================================
// SceneBuilder -- receives callbacks from parse(), populates Scene
// ============================================================================

class SceneBuilder {
public:
    SceneBuilder(Scene& scene, const std::string& baseDir)
        : m_scene(scene), m_baseDir(baseDir) {
        m_stateStack.push_back(GraphicsState{});
    }

    GraphicsState& state() { return m_stateStack.back(); }

    // -- Transforms --
    void identity() { state().transform = glm::mat4(1.f); }

    void translate(float dx, float dy, float dz) {
        state().transform = state().transform * glm::translate(glm::mat4(1.f), {dx, dy, dz});
    }

    void rotate(float angle, float ax, float ay, float az) {
        float len = std::sqrt(ax*ax + ay*ay + az*az);
        if (len > 1e-10f) {
            ax /= len; ay /= len; az /= len;
        }
        state().transform = state().transform * glm::rotate(glm::mat4(1.f), glm::radians(angle), {ax, ay, az});
    }

    void scale(float sx, float sy, float sz) {
        state().transform = state().transform * glm::scale(glm::mat4(1.f), {sx, sy, sz});
    }

    void lookAt(float ex, float ey, float ez, float lx, float ly, float lz,
                float ux, float uy, float uz) {
        m_lookAtEye = glm::vec3(ex, ey, ez);
        m_lookAtTarget = glm::vec3(lx, ly, lz);
        m_lookAtUp = glm::normalize(glm::vec3(ux, uy, uz));
        m_hasLookAt = true;

        // PBRT LookAt defines camera-to-world; glm::lookAt returns world-to-camera
        glm::mat4 worldToCamera = glm::lookAt(m_lookAtEye, m_lookAtTarget, m_lookAtUp);
        glm::mat4 cameraToWorld = glm::inverse(worldToCamera);
        state().transform = state().transform * cameraToWorld;
    }

    void setTransform(float m[16]) {
        glm::mat4 mat;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat[j][i] = m[i*4+j];
        state().transform = mat;
    }

    void concatTransform(float m[16]) {
        glm::mat4 mat;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat[j][i] = m[i*4+j];
        state().transform = state().transform * mat;
    }

    // -- Block structure --
    void worldBegin() {
        m_inWorld = true;
        // Save camera transform before resetting
        m_cameraTransform = state().transform;
        // Reset to identity for world
        state().transform = glm::mat4(1.f);
    }

    void attributeBegin() {
        m_stateStack.push_back(state());
    }

    void attributeEnd() {
        if (m_stateStack.size() > 1) m_stateStack.pop_back();
    }

    void reverseOrientation() {
        state().reverseOrientation = !state().reverseOrientation;
    }

    // -- Objects --
    void objectBegin(const std::string& name) {
        attributeBegin();
        m_inObjectBlock = true;
        m_currentObjectName = name;
        m_currentObjectMeshes.clear();
    }

    void objectEnd() {
        if (m_inObjectBlock) {
            m_namedObjects[m_currentObjectName] = m_currentObjectMeshes;
            m_currentObjectMeshes.clear();
            m_inObjectBlock = false;
        }
        attributeEnd();
    }

    void objectInstance(const std::string& name) {
        auto it = m_namedObjects.find(name);
        if (it != m_namedObjects.end()) {
            for (const auto& src : it->second) {
                Mesh m = src;
                applyTransformToMesh(m, state().transform);
                addMeshToScene(m);
            }
        }
    }

    // -- Camera --
    void camera(const std::string& type, const std::vector<ParsedParam>& params) {
        CameraInfo cam;

        auto isIdentity = [](const glm::mat4& m) {
            const float eps = 1e-6f;
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) {
                    float expected = (c == r) ? 1.f : 0.f;
                    if (fabsf(m[c][r] - expected) > eps) return false;
                }
            return true;
        };

        if (m_hasLookAt || !isIdentity(state().transform)) {
            glm::mat4 camToWorld = state().transform;
            cam.eye = glm::vec3(camToWorld * glm::vec4(0.f, 0.f, 0.f, 1.f));
            glm::vec3 forward = glm::normalize(glm::vec3(camToWorld * glm::vec4(0.f, 0.f, 1.f, 0.f)));
            cam.target = cam.eye + forward;
            cam.up = glm::normalize(glm::vec3(camToWorld * glm::vec4(0.f, 1.f, 0.f, 0.f)));
        } else {
            cam.eye = glm::vec3(0, 0, -5);
            cam.target = glm::vec3(0, 0, 0);
            cam.up = glm::vec3(0, 1, 0);
        }

        if (type == "perspective") {
            cam.fovy = getFloat(params, "fov", 45.f);
        } else if (type == "orthographic") {
            cam.fovy = 45.f;
        } else {
            cam.fovy = getFloat(params, "fov", 45.f);
        }

        cam.valid = true;
        m_scene.setCamera(cam);
    }

    // -- Materials --
    void makeNamedMaterial(const std::string& name, const std::vector<ParsedParam>& params) {
        Material mat;
        mat.name = name;

        std::string matType = getString(params, "type");
        applyMaterialParams(mat, matType, params);

        m_scene.addMaterial(mat);
        m_namedMaterials[name] = (uint32_t)(m_scene.materials().size() - 1);
    }

    void namedMaterial(const std::string& name) {
        auto it = m_namedMaterials.find(name);
        if (it != m_namedMaterials.end())
            state().materialIndex = it->second;
        else {
            // Create placeholder material
            Material mat;
            mat.name = name;
            m_scene.addMaterial(mat);
            uint32_t idx = (uint32_t)(m_scene.materials().size() - 1);
            m_namedMaterials[name] = idx;
            state().materialIndex = idx;
        }
    }

    void material(const std::string& type, const std::vector<ParsedParam>& params) {
        Material mat;
        mat.name = type + "_" + std::to_string(m_anonMatCount++);
        applyMaterialParams(mat, type, params);
        m_scene.addMaterial(mat);
        state().materialIndex = (uint32_t)(m_scene.materials().size() - 1);
    }

    // -- Textures --
    void texture(const std::string& name, const std::string& type,
                 const std::string& texClass, const std::vector<ParsedParam>& params) {
        // Track texture definitions for material resolution
        TextureDef td;
        td.name = name;
        td.type = type;
        td.texClass = texClass;
        td.filename = getString(params, "filename");

        auto getNamedValue = [&](const std::string& texName, const glm::vec3& fb) {
            auto it = m_namedTextures.find(texName);
            if (it != m_namedTextures.end())
                return it->second.value;
            return fb;
        };
        auto getNamedFloat = [&](const std::string& texName, float fb) {
            auto it = m_namedTextures.find(texName);
            if (it != m_namedTextures.end())
                return it->second.floatValue;
            return fb;
        };

        // Default value
        td.value = getRgb(params, "value", glm::vec3(0.5f));
        td.floatValue = getFloat(params, "value", 0.5f);

        // Scale texture: multiply two texture values
        if (texClass == "scale") {
            std::string t1 = getString(params, "tex1");
            std::string t2 = getString(params, "tex2");
            glm::vec3 v1 = getNamedValue(t1, td.value);
            glm::vec3 v2 = getNamedValue(t2, td.value);
            td.value = v1 * v2;
            td.floatValue = getNamedFloat(t1, td.floatValue) * getNamedFloat(t2, td.floatValue);
        }

        // Mix texture: linear blend
        if (texClass == "mix") {
            std::string t1 = getString(params, "tex1");
            std::string t2 = getString(params, "tex2");
            float amount = getFloat(params, "amount", 0.5f);
            glm::vec3 v1 = getNamedValue(t1, td.value);
            glm::vec3 v2 = getNamedValue(t2, td.value);
            td.value = v1 * (1.f - amount) + v2 * amount;
            float f1 = getNamedFloat(t1, td.floatValue);
            float f2 = getNamedFloat(t2, td.floatValue);
            td.floatValue = f1 * (1.f - amount) + f2 * amount;
        }

        // Checkerboard: average color for preview
        if (texClass == "checkerboard") {
            std::string t1 = getString(params, "tex1");
            std::string t2 = getString(params, "tex2");
            glm::vec3 v1 = getNamedValue(t1, td.value);
            glm::vec3 v2 = getNamedValue(t2, td.value);
            td.value = 0.5f * (v1 + v2);
            td.floatValue = 0.5f * (getNamedFloat(t1, td.floatValue) + getNamedFloat(t2, td.floatValue));
        }
        
        // For imagemap textures, try to load the image and compute average color
        if (texClass == "imagemap" && !td.filename.empty()) {
            // Try to find the texture file relative to scene base directory
            std::string texPath = td.filename;
            if (!fs::exists(texPath) && !m_baseDir.empty()) {
                // Try relative to base directory
                fs::path relPath = fs::path(m_baseDir) / td.filename;
                if (fs::exists(relPath)) {
                    texPath = relPath.string();
                }
            }
            
            if (fs::exists(texPath)) {
                TextureData texData = loadTexture(texPath);
                if (texData.pixels && texData.width > 0 && texData.height > 0) {
                    // Compute average color from texture pixels
                    // texData is loaded in RGBA format (4 bytes per pixel)
                    uint32_t pixelCount = texData.width * texData.height;
                    double r = 0, g = 0, b = 0;
                    
                    for (uint32_t i = 0; i < pixelCount; ++i) {
                        uint8_t* pix = texData.pixels + i * 4;
                        r += pix[0];
                        g += pix[1];
                        b += pix[2];
                        // Alpha (pix[3]) is ignored
                    }
                    
                    // Normalize to [0, 1]
                    double invCount = 1.0 / pixelCount;
                    td.value.x = (float)(r * invCount / 255.0);
                    td.value.y = (float)(g * invCount / 255.0);
                    td.value.z = (float)(b * invCount / 255.0);
                    td.floatValue = (float)((r + g + b) * invCount / (3.0 * 255.0));
                    
                    // 保存完整的纹理数据
                    td.texData = std::move(texData);
                    
                    // Log to both cout and file
                    std::cout << "[PBRTLoader] Loaded imagemap '" << name 
                              << "' from '" << texPath << "' -> avg color ("
                              << td.value.x << ", " << td.value.y << ", " << td.value.z << ")\n";
                    std::cerr << "[PBRTLoader] Loaded imagemap '" << name 
                              << "' from '" << texPath << "' -> avg color ("
                              << td.value.x << ", " << td.value.y << ", " << td.value.z << ")\n";
                } else if (texData.pixels) {
                    std::cerr << "[PBRTLoader] WARNING: imagemap '" << name 
                              << "' loaded but has invalid dimensions (" 
                              << texData.width << "x" << texData.height << ")\n";
                }
            } else {
                std::cerr << "[PBRTLoader] WARNING: imagemap texture file not found: " << td.filename << " (resolved as: " << texPath << ")\n";
            }
        }
        
        m_namedTextures[name] = td;
    }

    // -- Lights --
    void lightSource(const std::string& type, const std::vector<ParsedParam>& params) {
        Light light;
        bool addLight = true;

        glm::vec3 L = getRgb(params, "L", glm::vec3(1.f));
        glm::vec3 I = getRgb(params, "I", L);
        float sc = getFloat(params, "scale", 1.f);
        float power = getFloat(params, "power", 0.f);

        if (type == "distant") {
            light.type = LightType::Directional;
            // from / to params
            auto pFrom = findParam(params, "from");
            auto pTo = findParam(params, "to");
            glm::vec3 from(0, 0, 0), to(0, 0, 1);
            if (pFrom && pFrom->floats.size() >= 3)
                from = {pFrom->floats[0], pFrom->floats[1], pFrom->floats[2]};
            if (pTo && pTo->floats.size() >= 3)
                to = {pTo->floats[0], pTo->floats[1], pTo->floats[2]};
            // Transform by current CTM
            from = glm::vec3(state().transform * glm::vec4(from, 1.f));
            to = glm::vec3(state().transform * glm::vec4(to, 1.f));
            light.direction = glm::normalize(to - from);
            light.position = from;
            light.intensity = I * sc;
            if (power > 0.f) light.intensity = glm::vec3(power);
        } else if (type == "infinite") {
            glm::vec3 envScale = L * sc;
            std::string mapname = getString(params, "mapname");
            if (mapname.empty()) mapname = getString(params, "filename");
            
            // Extract 3x3 rotation matrix from the transform
            glm::mat3 lightTransform = glm::mat3(state().transform);

            bool envLoaded = false;
            
            if (!mapname.empty()) {
                std::string texPath = mapname;
                
                // Try to resolve relative path
                if (!fs::exists(texPath) && !m_baseDir.empty()) {
                    fs::path relPath = fs::path(m_baseDir) / mapname;
                    std::string relPathStr = relPath.string();
                    // Normalize path to use forward slashes
                    for (char& c : relPathStr) {
                        if (c == '\\') c = '/';
                    }
                    if (fs::exists(relPath)) texPath = relPathStr;
                }
                
                // Normalize final path to use forward slashes
                for (char& c : texPath) {
                    if (c == '\\') c = '/';
                }
                
                std::cerr << "[PBRTLoader] Loading environment map: " << texPath << "\n";
                
                if (fs::exists(texPath)) {
                    std::string lowerTexPath = texPath;
                    std::transform(lowerTexPath.begin(), lowerTexPath.end(), lowerTexPath.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                    if (lowerTexPath.size() >= 4 && lowerTexPath.substr(lowerTexPath.size() - 4) == ".exr") {
                        HDRTextureData hdrData = loadEXRFloat(texPath);
                        if (hdrData.valid()) {
                            m_scene.setEnvironmentMapHDR(hdrData, envScale, lightTransform);

                            // Compute ambient from the envmap average (linear HDR)
                            uint64_t pixelCount = static_cast<uint64_t>(hdrData.width) *
                                                  static_cast<uint64_t>(hdrData.height);
                            double r = 0, g = 0, b = 0;
                            for (uint64_t i = 0; i < pixelCount; ++i) {
                                const float* pix = &hdrData.pixels[i * 4];
                                r += pix[0];
                                g += pix[1];
                                b += pix[2];
                            }
                            double invCount = (pixelCount > 0) ? (1.0 / static_cast<double>(pixelCount)) : 0.0;
                            glm::vec3 avg(
                                float(r * invCount),
                                float(g * invCount),
                                float(b * invCount));
                            m_scene.setAmbientIntensity(avg * envScale);
                            envLoaded = true;
                        }
                    } else {
                        TextureData texData = loadTexture(texPath);

                        if (texData.pixels && texData.width > 0 && texData.height > 0) {
                            m_scene.setEnvironmentMap(texData, envScale, lightTransform);

                            // Compute ambient from the envmap average
                            uint32_t pixelCount = texData.width * texData.height;
                            double r = 0, g = 0, b = 0;
                            for (uint32_t i = 0; i < pixelCount; ++i) {
                                uint8_t* pix = texData.pixels + i * 4;
                                r += pix[0];
                                g += pix[1];
                                b += pix[2];
                            }
                            double invCount = 1.0 / pixelCount;
                            glm::vec3 avg(
                                float(r * invCount / 255.0),
                                float(g * invCount / 255.0),
                                float(b * invCount / 255.0));
                            m_scene.setAmbientIntensity(avg * envScale);
                            envLoaded = true;
                        }
                    }
                }
            }

            if (power > 0.f) envScale = glm::vec3(power);
            
            // Use L*sc as fallback only if envmap was not loaded
            if (!envLoaded) {
                m_scene.setAmbientIntensity(envScale);
            }
            addLight = false;
        } else if (type == "spot") {
            light.type = LightType::Spot;
            auto pFrom = findParam(params, "from");
            auto pTo = findParam(params, "to");
            glm::vec3 from(0, 0, 0), to(0, 0, 1);
            if (pFrom && pFrom->floats.size() >= 3)
                from = {pFrom->floats[0], pFrom->floats[1], pFrom->floats[2]};
            if (pTo && pTo->floats.size() >= 3)
                to = {pTo->floats[0], pTo->floats[1], pTo->floats[2]};

            from = glm::vec3(state().transform * glm::vec4(from, 1.f));
            to = glm::vec3(state().transform * glm::vec4(to, 1.f));

            light.position = from;
            light.direction = glm::normalize(to - from);
            light.intensity = I * sc;
            if (power > 0.f) light.intensity = glm::vec3(power);

            float coneAngle = getFloat(params, "coneangle", 30.f);
            float coneDelta = getFloat(params, "conedeltaangle", 5.f);
            float inner = std::max(0.f, coneAngle - coneDelta);
            light.cosInner = std::cos(glm::radians(inner));
            light.cosOuter = std::cos(glm::radians(coneAngle));
        } else {
            // point, etc.
            light.type = LightType::Point;
            auto pFrom = findParam(params, "from");
            glm::vec3 pos(0, 0, 0);
            if (pFrom && pFrom->floats.size() >= 3)
                pos = {pFrom->floats[0], pFrom->floats[1], pFrom->floats[2]};
            light.position = glm::vec3(state().transform * glm::vec4(pos, 1.f));
            light.direction = glm::vec3(0.f);
            light.intensity = I * sc;
            if (power > 0.f) light.intensity = glm::vec3(power);
        }

        if (addLight)
            m_scene.addLight(light);
    }

    void areaLightSource(const std::string& type, const std::vector<ParsedParam>& params) {
        state().areaLightType = type;
        state().areaLightL = getRgb(params, "L", glm::vec3(1.f));
        state().areaLightScale = getFloat(params, "scale", 1.f);
        float power = getFloat(params, "power", 0.f);
        if (power > 0.f) state().areaLightL = glm::vec3(power);
    }

    // -- Shapes --
    void shape(const std::string& type, const std::vector<ParsedParam>& params) {
        if (type == "plymesh") {
            shapePlyMesh(params);
        } else if (type == "trianglemesh") {
            shapeTriangleMesh(params);
        } else if (type == "loopsubdiv") {
            shapeTriangleMesh(params);  // approximate: treat as triangle mesh
        } else if (type == "bilinearmesh") {
            shapeBilinearMesh(params);
        } else if (type == "sphere") {
            shapeSphere(params);
        } else if (type == "disk") {
            shapeDisk(params);
        } else if (type == "cylinder") {
            shapeCylinder(params);
        } else if (type == "curve") {
            shapeCurve(params);
        }
    }

    // -- Coordinate systems --
    void coordinateSystem(const std::string& name) {
        m_coordSystems[name] = state().transform;
    }

    void coordSysTransform(const std::string& name) {
        auto it = m_coordSystems.find(name);
        if (it != m_coordSystems.end())
            state().transform = it->second;
    }

    // -- Stats --
    size_t meshCount() const { return m_scene.meshes().size(); }

private:
    void addMeshToScene(Mesh& mesh) {
        // If area light is active, set material emission
        if (!state().areaLightType.empty()) {
            glm::vec3 emission = state().areaLightL * state().areaLightScale;
            if (glm::length(emission) > 0.001f) {
                // Create a new emissive material
                Material mat;
                if (state().materialIndex < m_scene.materials().size())
                    mat = m_scene.materials()[state().materialIndex];
                mat.emission = emission;
                mat.name = mat.name + "_emissive";
                m_scene.addMaterial(mat);
                mesh.materialIndex = (uint32_t)(m_scene.materials().size() - 1);
            }
        }

        if (m_inObjectBlock) {
            m_currentObjectMeshes.push_back(mesh);
        } else {
            m_scene.addMesh(mesh);
        }
    }

    void applyMaterialParams(Material& mat, const std::string& type,
                              const std::vector<ParsedParam>& params) {
        mat.baseColor = glm::vec3(0.5f);

        // Parse remapRoughness flag
        mat.remapRoughness = true;
        auto* remapParam = findParam(params, "remaproughness");
        if (remapParam && !remapParam->bools.empty())
            mat.remapRoughness = remapParam->bools[0];

        // Parse roughness parameters
        float roughness = getFloat(params, "roughness", 0.5f);
        mat.uroughness = getFloat(params, "uroughness", roughness);
        mat.vroughness = getFloat(params, "vroughness", roughness);

        // Set material type and type-specific parameters
        if (type == "diffuse") {
            mat.type = MaterialType::Diffuse;
            mat.baseColor = resolveRgb(params, "reflectance", glm::vec3(0.5f));
            mat.sigma = getFloat(params, "sigma", 0.0f);
        } 
        else if (type == "conductor") {
            mat.type = MaterialType::Conductor;
            mat.metallic = 1.f;
            mat.roughness = (mat.uroughness + mat.vroughness) * 0.5f;
            
            // Try reflectance first (direct RGB)
            mat.baseColor = resolveRgb(params, "reflectance", glm::vec3(-1.f));
            
            // Parse eta and k (complex IOR spectrum)
            auto* etaParam = findParam(params, "eta");
            auto* kParam = findParam(params, "k");
            if (etaParam && etaParam->floats.size() >= 3) {
                mat.eta = glm::vec3(etaParam->floats[0], etaParam->floats[1], etaParam->floats[2]);
            } else {
                // Default copper eta
                mat.eta = glm::vec3(0.2f, 0.9f, 1.1f);
            }
            if (kParam && kParam->floats.size() >= 3) {
                mat.k = glm::vec3(kParam->floats[0], kParam->floats[1], kParam->floats[2]);
            } else {
                // Default copper k
                mat.k = glm::vec3(3.6f, 2.6f, 2.3f);
            }
            
            if (mat.baseColor.x < 0.f) {
                mat.baseColor = resolveConductorColor(params);
            }
        } 
        else if (type == "coatedconductor") {
            mat.type = MaterialType::CoatedConductor;
            mat.metallic = 1.f;
            mat.roughness = (mat.uroughness + mat.vroughness) * 0.5f;
            mat.ior = getFloat(params, "eta", 1.5f);  // Interface IOR
            
            // Parse conductor base eta and k
            auto* etaParam = findParam(params, "conductor.eta");
            auto* kParam = findParam(params, "conductor.k");
            if (etaParam && etaParam->floats.size() >= 3) {
                mat.eta = glm::vec3(etaParam->floats[0], etaParam->floats[1], etaParam->floats[2]);
            } else {
                mat.eta = glm::vec3(0.2f, 0.9f, 1.1f);
            }
            if (kParam && kParam->floats.size() >= 3) {
                mat.k = glm::vec3(kParam->floats[0], kParam->floats[1], kParam->floats[2]);
            } else {
                mat.k = glm::vec3(3.6f, 2.6f, 2.3f);
            }
            
            mat.baseColor = resolveConductorColor(params);
        }
        else if (type == "dielectric" || type == "thindielectric") {
            mat.type = MaterialType::Dielectric;
            mat.transmission = 1.f;
            mat.ior = getFloat(params, "eta", 1.5f);
            mat.baseColor = glm::vec3(1.f);
            mat.roughness = (mat.uroughness + mat.vroughness) * 0.5f;
            
            // If roughness is non-zero, use RoughDielectric
            if (mat.roughness > 0.001f) {
                mat.type = MaterialType::RoughDielectric;
            }
        } 
        else if (type == "coateddiffuse") {
            mat.type = MaterialType::CoatedDiffuse;
            mat.baseColor = resolveRgb(params, "reflectance", glm::vec3(0.5f));
            mat.ior = getFloat(params, "eta", 1.5f);  // Interface IOR
            mat.roughness = (mat.uroughness + mat.vroughness) * 0.5f;
        } 
        else if (type == "diffusetransmission") {
            mat.type = MaterialType::Diffuse;
            mat.baseColor = resolveRgb(params, "reflectance", glm::vec3(0.25f));
            mat.transmission = 0.5f;
            mat.roughness = (mat.uroughness + mat.vroughness) * 0.5f;
        } 
        else if (type == "hair") {
            mat.type = MaterialType::BasicPBR;  // Fallback to BasicPBR
            mat.baseColor = resolveRgb(params, "sigma_a", glm::vec3(0.06f, 0.10f, 0.20f));
            mat.roughness = 0.3f;
        } 
        else if (type == "subsurface") {
            mat.type = MaterialType::Subsurface;
            mat.baseColor = resolveRgb(params, "reflectance", glm::vec3(0.5f));
            mat.roughness = getFloat(params, "roughness", 0.5f);
            mat.transmission = 0.3f;
        } 
        else if (type == "mix") {
            mat.type = MaterialType::BasicPBR;  // Fallback
            mat.baseColor = glm::vec3(0.5f);
            mat.roughness = 0.5f;
        } 
        else {
            // Fallback: BasicPBR
            mat.type = MaterialType::BasicPBR;
            mat.baseColor = resolveRgb(params, "reflectance", mat.baseColor);
            mat.baseColor = resolveRgb(params, "Kd", mat.baseColor);
            mat.roughness = (mat.uroughness + mat.vroughness) * 0.5f;
        }

        mat.emission = resolveRgb(params, "Le", mat.emission);
        mat.emission = resolveRgb(params, "L", mat.emission);

        // Try to resolve texture paths
        resolveTextureFilename(params, "reflectance", mat.baseColorTexPath);
        resolveTextureFilename(params, "filename", mat.baseColorTexPath);
        resolveTextureFilename(params, "normalmap", mat.normalTexPath);
        resolveTextureFilename(params, "bumpmap", mat.normalTexPath);
        resolveTextureFilename(params, "Le", mat.emissionTexPath);
        resolveTextureFilename(params, "L", mat.emissionTexPath);
        
        // 解析完整纹理数据
        resolveTextureData(params, "reflectance", mat.baseColorTexData);
        // 如果reflectance没有纹理，尝试filename参数
        if (!mat.baseColorTexData.pixels) {
            resolveTextureData(params, "filename", mat.baseColorTexData);
        }
        resolveTextureData(params, "normalmap", mat.normalTexData);
        resolveTextureData(params, "bumpmap", mat.normalTexData);
        resolveTextureData(params, "Le", mat.emissionTexData);
        resolveTextureData(params, "L", mat.emissionTexData);

        // Roughness texture (average only for now)
        TextureData roughTex;
        resolveTextureData(params, "roughness", roughTex);
        if (roughTex.pixels && roughTex.width > 0 && roughTex.height > 0) {
            uint32_t pixelCount = roughTex.width * roughTex.height;
            double sum = 0.0;
            for (uint32_t i = 0; i < pixelCount; ++i) {
                uint8_t* pix = roughTex.pixels + i * 4;
                sum += pix[0];
            }
            double avg = sum / (pixelCount * 255.0);
            mat.roughness = static_cast<float>(avg);
        }
    }

    // Resolve an RGB parameter that might reference a named texture
    glm::vec3 resolveRgb(const std::vector<ParsedParam>& params,
                          const std::string& name, const glm::vec3& fb) {
        auto* p = findParam(params, name);
        if (!p) return fb;

        // Direct RGB values
        if (p->floats.size() >= 3)
            return {p->floats[0], p->floats[1], p->floats[2]};
        if (p->floats.size() == 1)
            return glm::vec3(p->floats[0]);

        // Texture reference (type == "texture" or "spectrum")
        if ((p->type == "texture" || p->type == "spectrum") && !p->strings.empty()) {
            auto it = m_namedTextures.find(p->strings[0]);
            if (it != m_namedTextures.end())
                return it->second.value;
        }

        // String reference
        if (!p->strings.empty()) {
            auto it = m_namedTextures.find(p->strings[0]);
            if (it != m_namedTextures.end())
                return it->second.value;
        }

        return fb;
    }

    // 解析纹理数据（用于加载完整纹理而不只是平均颜色）
    void resolveTextureData(const std::vector<ParsedParam>& params,
                            const std::string& name, TextureData& outTexData) {
        auto* p = findParam(params, name);
        if (!p) return;

        // 只处理纹理引用
        std::string texName;
        if (p->type == "texture" && !p->strings.empty()) {
            texName = p->strings[0];
        } else if (!p->strings.empty()) {
            texName = p->strings[0];
        }

        if (!texName.empty()) {
            auto it = m_namedTextures.find(texName);
            if (it != m_namedTextures.end() && it->second.texData.pixels) {
                // 复制纹理数据
                outTexData = it->second.texData;
                std::cout << "[PBRTLoader] Resolved texture data for '" << name 
                          << "' -> '" << texName << "' (" 
                          << outTexData.width << "x" << outTexData.height << ")\n";
            }
        }
    }

    // Approximate conductor color from known pbrt-v4 metal spectrum names
    glm::vec3 resolveConductorColor(const std::vector<ParsedParam>& params) {
        // Known metal reflectance approximations (sRGB linear)
        static const std::unordered_map<std::string, glm::vec3> knownMetals = {
            {"metal-Ag-eta",  {0.972f, 0.960f, 0.915f}},  // Silver
            {"metal-Au-eta",  {1.000f, 0.766f, 0.336f}},  // Gold
            {"metal-Cu-eta",  {0.955f, 0.638f, 0.538f}},  // Copper
            {"metal-Al-eta",  {0.913f, 0.922f, 0.924f}},  // Aluminum
            {"metal-Fe-eta",  {0.531f, 0.512f, 0.496f}},  // Iron
            {"metal-Ti-eta",  {0.542f, 0.497f, 0.449f}},  // Titanium
            {"metal-Cr-eta",  {0.549f, 0.556f, 0.554f}},  // Chromium
            {"metal-Ni-eta",  {0.660f, 0.609f, 0.526f}},  // Nickel
            {"metal-Pt-eta",  {0.672f, 0.637f, 0.585f}},  // Platinum
            {"metal-W-eta",   {0.504f, 0.498f, 0.478f}},  // Tungsten
            {"metal-CuZn-eta",{0.887f, 0.789f, 0.584f}},  // Brass (CuZn)
            {"metal-TiN-eta", {0.734f, 0.601f, 0.412f}},  // Titanium Nitride
        };

        auto* pEta = findParam(params, "eta");
        if (pEta && !pEta->strings.empty()) {
            auto it = knownMetals.find(pEta->strings[0]);
            if (it != knownMetals.end())
                return it->second;
        }
        // If eta has float values (inline spectrum), approximate as gray
        if (pEta && pEta->floats.size() >= 3) {
            // Average the first 3 eta values as rough approximation
            return glm::vec3((pEta->floats[0] + pEta->floats[1] + pEta->floats[2]) / 3.f);
        }
        // Default conductor appearance (slightly warm silver)
        return glm::vec3(0.8f, 0.8f, 0.8f);
    }

    void resolveTextureFilename(const std::vector<ParsedParam>& params,
                                 const std::string& name, std::string& outPath) {
        auto* p = findParam(params, name);
        if (!p) return;
        if (p->type == "texture" && !p->strings.empty()) {
            auto it = m_namedTextures.find(p->strings[0]);
            if (it != m_namedTextures.end() && !it->second.filename.empty()) {
                fs::path fpath = fs::path(m_baseDir) / it->second.filename;
                if (fs::exists(fpath)) outPath = fpath.string();
            }
        }
        if (!p->strings.empty() && outPath.empty()) {
            fs::path fpath = fs::path(m_baseDir) / p->strings[0];
            if (fs::exists(fpath)) outPath = fpath.string();
        }
    }

    void shapePlyMesh(const std::vector<ParsedParam>& params) {
        std::string file = getString(params, "filename");
        if (file.empty()) return;

        fs::path fp = fs::path(m_baseDir) / file;
        std::vector<Mesh> meshes;
        if (loadPlyDirect(fp.string(), state().transform, state().materialIndex, meshes)) {
            for (auto& m : meshes) {
                m.materialIndex = state().materialIndex;
                addMeshToScene(m);
            }
        }
    }

    void shapeTriangleMesh(const std::vector<ParsedParam>& params) {
        auto P = getFloats(params, "P");
        if (P.size() < 9) return;

        Mesh mesh;
        mesh.name = "trianglemesh";
        mesh.materialIndex = state().materialIndex;

        size_t vc = P.size() / 3;
        mesh.vertices.resize(vc);
        for (size_t i = 0; i < vc; ++i) {
            mesh.vertices[i].position = glm::vec3(P[i*3], P[i*3+1], P[i*3+2]);
            mesh.vertices[i].normal = glm::vec3(0.f);
            mesh.vertices[i].texCoord = glm::vec2(0.f);
            mesh.vertices[i].tangent = glm::vec3(0.f);
        }

        // Normals
        auto N = getFloats(params, "N");
        for (size_t i = 0; i < std::min(vc, N.size()/3); ++i)
            mesh.vertices[i].normal = glm::vec3(N[i*3], N[i*3+1], N[i*3+2]);

        // UVs
        auto uv = getFloats(params, "uv");
        for (size_t i = 0; i < std::min(vc, uv.size()/2); ++i)
            mesh.vertices[i].texCoord = glm::vec2(uv[i*2], uv[i*2+1]);

        // Indices
        auto indices = getInts(params, "indices");
        if (!indices.empty()) {
            for (int v : indices)
                if (v >= 0 && v < (int)vc)
                    mesh.indices.push_back((uint32_t)v);
        } else {
            // No explicit indices; assume sequential triangles
            for (size_t i = 0; i + 2 < vc; i += 3) {
                mesh.indices.push_back((uint32_t)i);
                mesh.indices.push_back((uint32_t)(i+1));
                mesh.indices.push_back((uint32_t)(i+2));
            }
        }

        bool needNormals = N.empty() || N.size()/3 < vc;
        if (needNormals) computeNormals(mesh);
        applyTransformToMesh(mesh, state().transform);
        addMeshToScene(mesh);
    }

    void shapeBilinearMesh(const std::vector<ParsedParam>& params) {
        auto P = getFloats(params, "P");
        if (P.size() < 12) return;

        size_t nverts = P.size() / 3;
        Mesh mesh;
        mesh.name = "bilinearmesh";
        mesh.materialIndex = state().materialIndex;
        mesh.vertices.resize(nverts);
        for (size_t i = 0; i < nverts; ++i) {
            mesh.vertices[i].position = glm::vec3(P[i*3], P[i*3+1], P[i*3+2]);
            mesh.vertices[i].normal = glm::vec3(0.f);
            mesh.vertices[i].texCoord = glm::vec2(0.f);
            mesh.vertices[i].tangent = glm::vec3(0.f);
        }

        auto uv = getFloats(params, "uv");
        for (size_t i = 0; i < std::min(nverts, uv.size()/2); ++i)
            mesh.vertices[i].texCoord = glm::vec2(uv[i*2], uv[i*2+1]);

        auto indices = getInts(params, "indices");
        if (!indices.empty()) {
            for (size_t i = 0; i + 3 < indices.size(); i += 4) {
                mesh.indices.insert(mesh.indices.end(), {
                    (uint32_t)indices[i], (uint32_t)indices[i+1], (uint32_t)indices[i+2],
                    (uint32_t)indices[i], (uint32_t)indices[i+2], (uint32_t)indices[i+3]
                });
            }
        } else {
            for (size_t i = 0; i + 3 < nverts; i += 4) {
                mesh.indices.insert(mesh.indices.end(), {
                    (uint32_t)i, (uint32_t)(i+1), (uint32_t)(i+2),
                    (uint32_t)i, (uint32_t)(i+2), (uint32_t)(i+3)
                });
            }
        }

        computeNormals(mesh);
        applyTransformToMesh(mesh, state().transform);
        addMeshToScene(mesh);
    }

    void shapeSphere(const std::vector<ParsedParam>& params) {
        float radius = getFloat(params, "radius", 1.0f);
        Mesh mesh = makeSphereMesh(radius);
        mesh.materialIndex = state().materialIndex;
        applyTransformToMesh(mesh, state().transform);
        addMeshToScene(mesh);
    }

    void shapeDisk(const std::vector<ParsedParam>& params) {
        float radius = getFloat(params, "radius", 1.0f);
        float height = getFloat(params, "height", 0.0f);
        Mesh mesh = makeDiskMesh(radius, height);
        mesh.materialIndex = state().materialIndex;
        applyTransformToMesh(mesh, state().transform);
        addMeshToScene(mesh);
    }

    void shapeCylinder(const std::vector<ParsedParam>& params) {
        float radius = getFloat(params, "radius", 1.0f);
        float zmin = getFloat(params, "zmin", -1.0f);
        float zmax = getFloat(params, "zmax", 1.0f);
        Mesh mesh = makeCylinderMesh(radius, zmin, zmax);
        mesh.materialIndex = state().materialIndex;
        applyTransformToMesh(mesh, state().transform);
        addMeshToScene(mesh);
    }

    void shapeCurve(const std::vector<ParsedParam>& params) {
        std::vector<float> P = getFloats(params, "P");
        if (P.size() < 12) return; // need at least 4 control points

        std::vector<glm::vec3> cp;
        cp.reserve(P.size() / 3);
        for (size_t i = 0; i + 2 < P.size(); i += 3)
            cp.emplace_back(P[i], P[i + 1], P[i + 2]);

        float width = getFloat(params, "width", 1.0f);
        float width0 = getFloat(params, "width0", width);
        float width1 = getFloat(params, "width1", width);
        int degree = getInt(params, "degree", 3);
        std::string basis = getString(params, "basis", "bezier");
        std::string typeStr = getString(params, "type", "flat");

        if (degree != 2 && degree != 3) return;
        if (basis != "bezier" && basis != "bspline") return;

        int nSegments = 0;
        if (basis == "bezier") {
            if ((int(cp.size()) - 1 - degree) % degree != 0) return;
            nSegments = (int(cp.size()) - 1) / degree;
        } else {
            if (int(cp.size()) < degree + 1) return;
            nSegments = int(cp.size()) - degree;
        }

        CurveType curveType = CurveType::Flat;
        if (typeStr == "ribbon") curveType = CurveType::Ribbon;
        else if (typeStr == "cylinder") curveType = CurveType::Cylinder;

        std::vector<glm::vec3> normals = getVec3Array(params, "N");
        if (curveType == CurveType::Ribbon && normals.size() != size_t(nSegments + 1))
            normals.clear();

        int cpOffset = 0;
        const glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(state().transform)));

        for (int seg = 0; seg < nSegments; ++seg) {
            std::array<glm::vec3, 4> segCp;

            if (basis == "bezier") {
                if (degree == 2) {
                    std::array<glm::vec3, 3> q = {cp[cpOffset], cp[cpOffset + 1], cp[cpOffset + 2]};
                    segCp = elevateQuadraticBezierToCubic(q);
                } else {
                    segCp = {cp[cpOffset], cp[cpOffset + 1], cp[cpOffset + 2], cp[cpOffset + 3]};
                }
                cpOffset += degree;
            } else {
                if (degree == 2) {
                    std::array<glm::vec3, 3> q = {cp[cpOffset], cp[cpOffset + 1], cp[cpOffset + 2]};
                    auto bez = quadraticBSplineToBezier(q);
                    segCp = elevateQuadraticBezierToCubic(bez);
                } else {
                    std::array<glm::vec3, 4> q = {cp[cpOffset], cp[cpOffset + 1], cp[cpOffset + 2], cp[cpOffset + 3]};
                    segCp = cubicBSplineToBezier(q);
                }
                cpOffset += 1;
            }

            float u0 = float(seg) / float(nSegments);
            float u1 = float(seg + 1) / float(nSegments);
            float w0 = glm::mix(width0, width1, u0);
            float w1 = glm::mix(width0, width1, u1);

            CurveSegment curve;
            curve.width0 = w0;
            curve.width1 = w1;
            curve.type = curveType;
            curve.materialIndex = state().materialIndex;

            for (int i = 0; i < 4; ++i)
                curve.cp[i] = glm::vec3(state().transform * glm::vec4(segCp[i], 1.f));

            if (!normals.empty()) {
                curve.hasNormals = true;
                curve.n0 = glm::normalize(normalMat * normals[seg]);
                curve.n1 = glm::normalize(normalMat * normals[seg + 1]);
            }

            m_scene.addCurve(curve);
        }
    }

    struct TextureDef {
        std::string name;
        std::string type;
        std::string texClass;
        std::string filename;
        glm::vec3 value = glm::vec3(0.5f);
        float floatValue = 0.5f;
        TextureData texData;  // 实际纹理像素数据
    };

    Scene& m_scene;
    std::string m_baseDir;
    std::vector<GraphicsState> m_stateStack;
    bool m_inWorld = false;

    // LookAt data
    bool m_hasLookAt = false;
    glm::vec3 m_lookAtEye = glm::vec3(0, 0, -5);
    glm::vec3 m_lookAtTarget = glm::vec3(0);
    glm::vec3 m_lookAtUp = glm::vec3(0, 1, 0);
    glm::mat4 m_cameraTransform = glm::mat4(1.f);

    // Named things
    std::unordered_map<std::string, uint32_t> m_namedMaterials;
    std::unordered_map<std::string, TextureDef> m_namedTextures;
    std::unordered_map<std::string, std::vector<Mesh>> m_namedObjects;
    std::unordered_map<std::string, glm::mat4> m_coordSystems;

    // Object block
    bool m_inObjectBlock = false;
    std::string m_currentObjectName;
    std::vector<Mesh> m_currentObjectMeshes;

    int m_anonMatCount = 0;
};

// ============================================================================
// parse() -- token-based dispatch, matching official pbrt-v4 parse()
// ============================================================================

static void parse(SceneBuilder& builder,
                  std::unique_ptr<PBRTTokenizer> tokenizer,
                  const std::string& baseDir,
                  std::unordered_set<std::string>& visitedFiles) {
    // File stack for Include
    std::vector<std::unique_ptr<PBRTTokenizer>> fileStack;
    fileStack.push_back(std::move(tokenizer));

    std::optional<PToken> ungetBuf;

    std::function<std::optional<PToken>(bool)> nextToken;
    std::function<void(const PToken&)> unget;

    nextToken = [&](bool required) -> std::optional<PToken> {
        if (ungetBuf.has_value()) {
            auto tok = std::move(ungetBuf);
            ungetBuf.reset();
            return tok;
        }

        while (!fileStack.empty()) {
            auto tok = fileStack.back()->next();
            if (tok.has_value()) {
                return tok;
            }
            // EOF on this file, pop stack
            fileStack.pop_back();
        }

        if (required) {
            std::cerr << "[PBRTLoader] Premature end of file\n";
        }
        return std::nullopt;
    };

    unget = [&](const PToken& t) {
        ungetBuf = t;
    };

    // Helper: read a quoted string token
    auto readString = [&]() -> std::string {
        auto t = nextToken(true);
        if (!t.has_value()) return "";
        return t->dequoted();
    };

    // Helper: read N floats
    auto readFloats = [&](int count) -> std::vector<float> {
        std::vector<float> v;
        v.reserve(count);
        for (int i = 0; i < count; ++i) {
            auto t = nextToken(true);
            if (!t.has_value()) break;
            v.push_back(toFloat(t->text));
        }
        return v;
    };

    // Helper: basicParamListEntrypoint - read name string + params
    auto basicParamList = [&]() -> std::pair<std::string, std::vector<ParsedParam>> {
        auto t = nextToken(true);
        std::string name = t.has_value() ? t->dequoted() : "";
        auto params = parseParameters(nextToken, unget);
        return {name, params};
    };

    // Main parsing loop
    while (true) {
        auto tok = nextToken(false);
        if (!tok.has_value()) break;

        const std::string& cmd = tok->text;
        if (cmd.empty()) continue;

        switch (cmd[0]) {
        case 'A':
            if (cmd == "AttributeBegin") {
                builder.attributeBegin();
            } else if (cmd == "AttributeEnd") {
                builder.attributeEnd();
            } else if (cmd == "Attribute") {
                // "Attribute" <target> params -- used for material overrides, skip
                auto [name, params] = basicParamList();
                (void)name; (void)params;
            } else if (cmd == "ActiveTransform") {
                auto t = nextToken(true); // "All", "StartTime", "EndTime"
                // We only support single transform -- ignore
            } else if (cmd == "AreaLightSource") {
                auto [name, params] = basicParamList();
                builder.areaLightSource(name, params);
            } else if (cmd == "Accelerator") {
                auto [name, params] = basicParamList();
                // Not relevant for scene geometry
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd
                          << " at " << tok->file << ":" << tok->line << "\n";
            }
            break;

        case 'C':
            if (cmd == "ConcatTransform") {
                auto bracket = nextToken(true);
                if (!bracket.has_value() || bracket->text != "[") {
                    // Maybe no bracket, unget and try floats
                    if (bracket.has_value()) unget(*bracket);
                }
                auto v = readFloats(16);
                if (v.size() == 16) {
                    float m[16];
                    for (int i = 0; i < 16; ++i) m[i] = v[i];
                    builder.concatTransform(m);
                }
                // Read closing bracket if present
                auto maybeClose = nextToken(false);
                if (maybeClose.has_value() && maybeClose->text != "]")
                    unget(*maybeClose);
            } else if (cmd == "CoordinateSystem") {
                builder.coordinateSystem(readString());
            } else if (cmd == "CoordSysTransform") {
                builder.coordSysTransform(readString());
            } else if (cmd == "ColorSpace") {
                readString(); // ignored
            } else if (cmd == "Camera") {
                auto [name, params] = basicParamList();
                builder.camera(name, params);
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'F':
            if (cmd == "Film") {
                auto [name, params] = basicParamList();
                // Film parameters -- not needed for geometry
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'I':
            if (cmd == "Integrator") {
                auto [name, params] = basicParamList();
            } else if (cmd == "Include" || cmd == "Import") {
                std::string filename = readString();
                if (!filename.empty()) {
                    // Try resolving relative to baseDir first
                    fs::path incPath = fs::path(baseDir) / filename;
                    if (!fs::exists(incPath) && !fileStack.empty()) {
                        // Fallback: resolve relative to the current file's directory
                        fs::path curFileDir = fs::path(fileStack.back()->filename()).parent_path();
                        fs::path alt = curFileDir / filename;
                        if (fs::exists(alt)) incPath = alt;
                    }
                    std::string normPath = fs::absolute(incPath).lexically_normal().string();
                    if (visitedFiles.count(normPath) == 0) {
                        visitedFiles.insert(normPath);
                        std::string content = readFileToString(incPath.string());
                        if (!content.empty()) {
                            auto inc = std::make_unique<PBRTTokenizer>(content, incPath.string());
                            fileStack.push_back(std::move(inc));
                        } else {
                            std::cerr << "[PBRTLoader] Cannot read Include/Import file: " << incPath.string() << "\n";
                        }
                    }
                }
            } else if (cmd == "Identity") {
                builder.identity();
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'L':
            if (cmd == "LightSource") {
                auto [name, params] = basicParamList();
                builder.lightSource(name, params);
            } else if (cmd == "LookAt") {
                auto v = readFloats(9);
                if (v.size() == 9)
                    builder.lookAt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'M':
            if (cmd == "MakeNamedMaterial") {
                auto [name, params] = basicParamList();
                builder.makeNamedMaterial(name, params);
            } else if (cmd == "MakeNamedMedium") {
                auto [name, params] = basicParamList();
                // Media -- skip
            } else if (cmd == "Material") {
                auto [name, params] = basicParamList();
                builder.material(name, params);
            } else if (cmd == "MediumInterface") {
                auto n1 = readString();
                // Check for optional second string
                auto maybe = nextToken(false);
                if (maybe.has_value()) {
                    if (maybe->isQuotedString()) {
                        // Two medium names
                    } else {
                        unget(*maybe);
                    }
                }
                // Media -- skip
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'N':
            if (cmd == "NamedMaterial") {
                builder.namedMaterial(readString());
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'O':
            if (cmd == "ObjectBegin") {
                builder.objectBegin(readString());
            } else if (cmd == "ObjectEnd") {
                builder.objectEnd();
            } else if (cmd == "ObjectInstance") {
                builder.objectInstance(readString());
            } else if (cmd == "Option") {
                readString(); // name
                nextToken(true); // value
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'P':
            if (cmd == "PixelFilter") {
                auto [name, params] = basicParamList();
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'R':
            if (cmd == "ReverseOrientation") {
                builder.reverseOrientation();
            } else if (cmd == "Rotate") {
                auto v = readFloats(4);
                if (v.size() == 4)
                    builder.rotate(v[0], v[1], v[2], v[3]);
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'S':
            if (cmd == "Shape") {
                auto [name, params] = basicParamList();
                builder.shape(name, params);
            } else if (cmd == "Sampler") {
                auto [name, params] = basicParamList();
            } else if (cmd == "Scale") {
                auto v = readFloats(3);
                if (v.size() == 3)
                    builder.scale(v[0], v[1], v[2]);
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'T':
            if (cmd == "TransformBegin") {
                builder.attributeBegin(); // deprecated, maps to attribute
            } else if (cmd == "TransformEnd") {
                builder.attributeEnd();
            } else if (cmd == "Transform") {
                auto bracket = nextToken(true);
                if (bracket.has_value() && bracket->text != "[") {
                    unget(*bracket);
                }
                auto v = readFloats(16);
                if (v.size() == 16) {
                    float m[16];
                    for (int i = 0; i < 16; ++i) m[i] = v[i];
                    builder.setTransform(m);
                }
                auto maybeClose = nextToken(false);
                if (maybeClose.has_value() && maybeClose->text != "]")
                    unget(*maybeClose);
            } else if (cmd == "Translate") {
                auto v = readFloats(3);
                if (v.size() == 3)
                    builder.translate(v[0], v[1], v[2]);
            } else if (cmd == "TransformTimes") {
                readFloats(2); // start, end -- ignore
            } else if (cmd == "Texture") {
                std::string name = readString();
                std::string type = readString();
                std::string texClass = readString();
                auto params = parseParameters(nextToken, unget);
                builder.texture(name, type, texClass, params);
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        case 'W':
            if (cmd == "WorldBegin") {
                builder.worldBegin();
            } else if (cmd == "WorldEnd") {
                // ignored in pbrt-v4
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd << "\n";
            }
            break;

        default:
            // Could be a number or other token that shouldn't appear at top level
            // This can happen with malformed files; just skip
            if (std::isdigit((unsigned char)cmd[0]) || cmd[0] == '-' || cmd[0] == '+') {
                // Stray number -- ignore
            } else {
                std::cerr << "[PBRTLoader] Unknown directive: " << cmd
                          << " at " << tok->file << ":" << tok->line << "\n";
            }
            break;
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

bool PBRTLoader::load(const std::string& pbrtPath, Scene& scene) {
    if (!fs::exists(pbrtPath)) {
        std::cerr << "[PBRTLoader] File not found: " << pbrtPath << "\n";
        return false;
    }

    std::cout << "[PBRTLoader] Loading PBRT scene: " << pbrtPath << "\n";
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string baseDir = fs::path(pbrtPath).parent_path().string();
    std::string content = readFileToString(pbrtPath);
    if (content.empty()) {
        std::cerr << "[PBRTLoader] Failed to read file: " << pbrtPath << "\n";
        return false;
    }

    auto readTime = std::chrono::high_resolution_clock::now();
    double readMs = std::chrono::duration<double, std::milli>(readTime - startTime).count();
    std::cout << "[PBRTLoader] Read " << (content.size() / 1024) << " KB in "
              << (int)readMs << " ms\n";

    scene.clear();

    // Add default material
    {
        Material mat;
        mat.name = "default";
        scene.addMaterial(mat);
    }

    // Default camera
    {
        CameraInfo cam;
        cam.eye = glm::vec3(0, 1, 5);
        cam.target = glm::vec3(0, 0, 0);
        cam.up = glm::vec3(0, 1, 0);
        cam.fovy = 45.f;
        scene.setCamera(cam);
    }

    SceneBuilder builder(scene, baseDir);

    std::unordered_set<std::string> visitedFiles;
    visitedFiles.insert(fs::absolute(pbrtPath).lexically_normal().string());

    auto tok = std::make_unique<PBRTTokenizer>(content, pbrtPath);
    parse(builder, std::move(tok), baseDir, visitedFiles);

    // Fallback: default cube if nothing loaded
    if (scene.meshes().empty()) {
        std::cout << "[PBRTLoader] No meshes found, creating default cube\n";
        Mesh mesh;
        mesh.name = "default_cube";
        mesh.materialIndex = 0;
        mesh.vertices.resize(8);
        mesh.vertices[0].position = {-1,-1,-1}; mesh.vertices[1].position = { 1,-1,-1};
        mesh.vertices[2].position = { 1, 1,-1}; mesh.vertices[3].position = {-1, 1,-1};
        mesh.vertices[4].position = {-1,-1, 1}; mesh.vertices[5].position = { 1,-1, 1};
        mesh.vertices[6].position = { 1, 1, 1}; mesh.vertices[7].position = {-1, 1, 1};
        for (auto& v : mesh.vertices) { v.normal = {0,1,0}; v.texCoord = {0,0}; }
        mesh.indices = {0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1, 2,6,7, 2,7,3, 0,3,7, 0,7,4, 1,5,6, 1,6,2};
        scene.addMesh(mesh);
    }

    if (scene.materials().empty()) {
        Material mat;
        mat.name = "default";
        scene.addMaterial(mat);
    }

    if (scene.lights().empty()) {
        Light l;
        l.type = LightType::Directional;
        l.direction = glm::normalize(glm::vec3(1, 1, 1));
        l.intensity = glm::vec3(500);
        scene.addLight(l);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    std::cout << "[PBRTLoader] Loaded: " << scene.meshes().size() << " meshes, "
              << scene.materials().size() << " materials, " << scene.lights().size()
              << " lights (" << (int)totalMs << " ms)\n";

    return true;
}
