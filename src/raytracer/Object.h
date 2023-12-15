#pragma once
#include "Intersection.h"


class Object
{
public:
	__device__ virtual bool intersect(const Ray& ray) = 0;
    __device__ virtual bool intersect(const Ray& ray, float&, uint32_t&) const = 0;
    __device__ virtual Intersection getIntersection(Ray _ray) = 0;
    __device__ virtual glm::vec3 evalDiffuseColor(const glm::vec2&) const = 0;
    __device__ virtual AABB getBounds() const = 0;
    __device__ virtual float getArea() const = 0;
    __device__ virtual void Sample(Intersection& pos, float& pdf) const = 0;
    __device__ virtual bool hasEmit() const = 0;
};