#include "Renderer.h"
#include "OptixRenderer.h"
#include "Camera.h"
#include "Scene.h"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// Minimal GLSL shaders for blitting the path-traced image to screen
// ---------------------------------------------------------------------------

static const char* kVertSrc = R"glsl(
#version 450 core
out vec2 vUV;
void main()
{
    // Full-screen triangle trick
    vec2 pos = vec2((gl_VertexID & 1) * 2.0 - 1.0,
                    (gl_VertexID & 2) * 1.0 - 1.0);
    vUV = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)glsl";

static const char* kFragSrc = R"glsl(
#version 450 core
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uTexture;
uniform float     uExposure;

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

vec3 linearToSRGB(vec3 c)
{
    bvec3 cutoff = lessThan(c, vec3(0.0031308));
    vec3  higher = vec3(1.055) * pow(c, vec3(1.0/2.4)) - vec3(0.055);
    vec3  lower  = c * vec3(12.92);
    return mix(higher, lower, cutoff);
}

void main()
{
    vec3 hdr  = texture(uTexture, vUV).rgb * uExposure;
    vec3 tone = ACESFilm(hdr);
    fragColor = vec4(linearToSRGB(tone), 1.0);
}
)glsl";

// ---------------------------------------------------------------------------

static unsigned int compileShader(const char* vert, const char* frag)
{
    auto compile = [](GLenum type, const char* src) {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char buf[512];
            glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
            throw std::runtime_error(std::string("Shader compile error: ") + buf);
        }
        return s;
    };

    unsigned int vs = compile(GL_VERTEX_SHADER,   vert);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, frag);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    int ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char buf[512];
        glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        throw std::runtime_error(std::string("Program link error: ") + buf);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

Renderer::Renderer(int width, int height)
    : m_width(width), m_height(height)
{
    initGL();
    m_optix = std::make_unique<OptixRenderer>(width, height);
}

Renderer::~Renderer()
{
    destroyGL();
}

void Renderer::initGL()
{
    // Full-screen quad VAO (we draw a single triangle-strip with 3 vertices)
    glGenVertexArrays(1, &m_quadVAO);

    // Texture for the path-traced image
    glGenTextures(1, &m_displayTex);
    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
                 m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);

    // Pixel buffer object for CUDA-OpenGL interop
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 static_cast<GLsizeiptr>(m_width * m_height * 4 * sizeof(float)),
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    m_displayShader = compileShader(kVertSrc, kFragSrc);
}

void Renderer::destroyGL()
{
    glDeleteVertexArrays(1, &m_quadVAO);
    glDeleteTextures(1, &m_displayTex);
    glDeleteBuffers(1, &m_pbo);
    glDeleteProgram(m_displayShader);
}

void Renderer::resize(int width, int height)
{
    m_width  = width;
    m_height = height;

    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
                 width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 static_cast<GLsizeiptr>(width * height * 4 * sizeof(float)),
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    m_optix->resize(width, height);
    resetAccumulation();
}

void Renderer::resetAccumulation()
{
    m_optix->resetAccumulation();
}

void Renderer::setSamplesPerPixel(int spp)  { m_spp = spp; }
void Renderer::setMaxBounces(int bounces)   { m_maxBounces = bounces; }
void Renderer::setExposure(float exposure)  { m_exposure = exposure; }
void Renderer::setScene(const Scene* scene) { m_optix->setScene(scene); }

uint32_t Renderer::accumulatedSamples() const
{
    return m_optix->accumulatedSamples();
}

void Renderer::render(const Camera& camera)
{
    // Let OptiX (or CPU fallback) render into the PBO
    m_optix->render(camera, m_pbo, m_spp, m_maxBounces);

    // Upload PBO → texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    m_width, m_height, GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Blit to screen
    glUseProgram(m_displayShader);
    glUniform1i(glGetUniformLocation(m_displayShader, "uTexture"), 0);
    glUniform1f(glGetUniformLocation(m_displayShader, "uExposure"), m_exposure);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}
