#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * @brief First-person-style camera for the path tracer.
 *
 * Position, yaw and pitch are stored in world space.  The camera exposes
 * helpers used by both the keyboard input handler and the ImGui panel.
 */
class Camera
{
public:
    explicit Camera(const glm::vec3& position = glm::vec3(0.f, 0.f, 5.f),
                    float yawDeg   = -90.f,
                    float pitchDeg =   0.f,
                    float fovDeg   =  60.f);

    /* ---- movement ---- */
    void moveForward(float amount);
    void moveRight  (float amount);
    void moveUp     (float amount);

    /** Rotate by delta yaw / pitch (both in degrees). */
    void rotate(float deltaYaw, float deltaPitch);

    /* ---- matrices ---- */
    glm::mat4 viewMatrix()       const;
    glm::mat4 projectionMatrix(float aspectRatio) const;

    /* ---- ray generation helpers (used by the renderer) ---- */
    /** Lower-left corner of the image plane in world space. */
    glm::vec3 lowerLeftCorner(float aspectRatio) const;
    glm::vec3 horizontal     (float aspectRatio) const;
    glm::vec3 vertical       ()                  const;
    glm::vec3 forward()  const { return m_front; }
    glm::vec3 right()    const { return m_right; }
    glm::vec3 up()       const { return m_up;    }

    /* ---- accessors ---- */
    glm::vec3 position()   const { return m_position; }
    float     yaw()        const { return m_yaw;  }
    float     pitch()      const { return m_pitch; }
    float     fovDegrees() const { return m_fov;  }
    void      setFovDegrees(float f);

private:
    void updateVectors();

    glm::vec3 m_position;
    float     m_yaw;     ///< degrees
    float     m_pitch;   ///< degrees
    float     m_fov;     ///< vertical field of view, degrees

    /* derived, recomputed when orientation changes */
    glm::vec3 m_front;
    glm::vec3 m_right;
    glm::vec3 m_up;

    static constexpr float kMaxPitch = 89.0f;
};
