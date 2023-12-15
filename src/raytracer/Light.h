#pragma once
#include <vec3.hpp>

#include "Global.h"

class Light
{
public:
    __device__ Light(const glm::vec3& p, const glm::vec3& i) : position(p), intensity(i) {}
    __device__ virtual ~Light() = default;
    glm::vec3 position;
    glm::vec3 intensity;
};

class AreaLight : public Light
{
public:
    __device__ AreaLight(const glm::vec3& p, const glm::vec3& i) : Light(p, i)
    {
        normal = glm::vec3(0, -1, 0);
        u = glm::vec3(1, 0, 0);
        v = glm::vec3(0, 0, 1);
        length = 100;
    }

    __device__ glm::vec3 SamplePoint() const
    {
        auto random_u = get_random_float();
        auto random_v = get_random_float();
        return position + random_u * u + random_v * v;
    }

    float length;
    glm::vec3 normal;
    glm::vec3 u;
    glm::vec3 v;
};