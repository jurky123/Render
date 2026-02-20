#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

Camera::Camera(const glm::vec3& position, float yawDeg, float pitchDeg, float fovDeg)
    : m_position(position)
    , m_yaw(yawDeg)
    , m_pitch(pitchDeg)
    , m_fov(fovDeg)
{
    updateVectors();
}

// ---------------------------------------------------------------------------
// Movement
// ---------------------------------------------------------------------------

void Camera::moveForward(float amount)
{
    m_position += m_front * amount;
}

void Camera::moveRight(float amount)
{
    m_position += m_right * amount;
}

void Camera::moveUp(float amount)
{
    m_position += glm::vec3(0.f, 1.f, 0.f) * amount;
}

void Camera::rotate(float deltaYaw, float deltaPitch)
{
    m_yaw   += deltaYaw;
    m_pitch  = std::clamp(m_pitch + deltaPitch, -kMaxPitch, kMaxPitch);
    updateVectors();
}

void Camera::setFovDegrees(float f)
{
    m_fov = std::clamp(f, 1.f, 179.f);
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

glm::mat4 Camera::viewMatrix() const
{
    return glm::lookAt(m_position, m_position + m_front, glm::vec3(0.f, 1.f, 0.f));
}

glm::mat4 Camera::projectionMatrix(float aspectRatio) const
{
    return glm::perspective(glm::radians(m_fov), aspectRatio, 0.01f, 10000.f);
}

// ---------------------------------------------------------------------------
// Ray-generation helpers
// ---------------------------------------------------------------------------

glm::vec3 Camera::vertical() const
{
    const float halfH = std::tan(glm::radians(m_fov) * 0.5f);
    return m_up * (2.f * halfH);
}

glm::vec3 Camera::horizontal(float aspectRatio) const
{
    return m_right * (aspectRatio * glm::length(vertical()));
}

glm::vec3 Camera::lowerLeftCorner(float aspectRatio) const
{
    return m_position + m_front
         - horizontal(aspectRatio) * 0.5f
         - vertical()              * 0.5f;
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

void Camera::updateVectors()
{
    const float yawRad   = glm::radians(m_yaw);
    const float pitchRad = glm::radians(m_pitch);

    glm::vec3 front;
    front.x = std::cos(pitchRad) * std::cos(yawRad);
    front.y = std::sin(pitchRad);
    front.z = std::cos(pitchRad) * std::sin(yawRad);

    m_front = glm::normalize(front);
    m_right = glm::normalize(glm::cross(m_front, glm::vec3(0.f, 1.f, 0.f)));
    m_up    = glm::normalize(glm::cross(m_right, m_front));
}
