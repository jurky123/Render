#include "Renderer.h"

#include "IRenderBackend.h"
#include "RenderSettings.h"
#include "optix/OptixBackend.h"

#include <glad/glad.h>

#include <stdexcept>

static const char* kVertSrc = R"glsl(
#version 450 core
out vec2 vUV;
void main()
{
    vec2 pos = vec2((gl_VertexID & 1) != 0 ? 3.0 : -1.0,
                    (gl_VertexID & 2) != 0 ? 3.0 : -1.0);
    vUV = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)glsl";

static const char* kFragSrc = R"glsl(
#version 450 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float uExposure;
vec3 ACESFilm(vec3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
vec3 linearToSRGB(vec3 c)
{
    bvec3 cutoff = lessThan(c, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(c, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = c * vec3(12.92);
    return mix(higher, lower, cutoff);
}
void main()
{
    vec3 hdr = texture(uTexture, vUV).rgb * uExposure;
    vec3 tone = ACESFilm(hdr);
    fragColor = vec4(linearToSRGB(tone), 1.0);
}
)glsl";

static unsigned int compileShader(const char* vert, const char* frag)
{
    auto compile = [](GLenum type, const char* src) {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char buf[512];
            glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
            throw std::runtime_error(std::string("Shader compile error: ") + buf);
        }
        return s;
    };

    unsigned int vs = compile(GL_VERTEX_SHADER, vert);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, frag);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    int ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
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

Renderer::Renderer(int width, int height)
    : m_width(width), m_height(height)
{
    initGL();
    m_backend = std::make_unique<OptixBackend>(width, height);
}

Renderer::~Renderer()
{
    destroyGL();
}

void Renderer::initGL()
{
    glGenVertexArrays(1, &m_quadVAO);

    glGenTextures(1, &m_displayTex);
    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);

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

void Renderer::setScene(const Scene* scene)
{
    m_backend->setScene(scene);
}

void Renderer::resize(int width, int height)
{
    m_width = width;
    m_height = height;

    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 static_cast<GLsizeiptr>(width * height * 4 * sizeof(float)),
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    m_backend->resize(width, height);
}

void Renderer::resetAccumulation()
{
    m_backend->resetAccumulation();
}

uint32_t Renderer::accumulatedSamples() const
{
    return m_backend->accumulatedSamples();
}

void Renderer::render(const Camera& camera, const RenderSettings& settings)
{
    m_backend->render(camera, m_pbo, settings);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glUseProgram(m_displayShader);
    glUniform1i(glGetUniformLocation(m_displayShader, "uTexture"), 0);
    glUniform1f(glGetUniformLocation(m_displayShader, "uExposure"), settings.exposure);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_displayTex);
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}
