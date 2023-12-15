#pragma once

#include <array>
#include <thrust/device_vector.h>
#include <glm.hpp>

#include "BVHNode.h"
#include "Global.h"
#include "Intersection.h"
#include "Object.h"
#include "OBJ_Loader.hpp"

// need three vertex and the light direction with the
__device__ inline bool rayTriangleIntersect(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& orig, const glm::vec3& dir, float& tnear, float& u, float& v)
{
	// Using Moller
	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 s = orig - v0;

	glm::vec3 s1 = glm::cross(dir, e2);
	glm::vec3 s2 = glm::cross(s, e1);

	float demon = 1.0f / glm::dot(s1, e1);
	float t = demon * glm::dot(s2, e2);
	float b1 = demon * glm::dot(s, s1);
	float b2 = demon * glm::dot(dir, s2);

	if (t > 0.f && b1 > 0.f && b2 > 0.f && 1.0f - b1 - b2 > 0.f)
	{
		return true;
	}
	return false;

}

class Triangle : public Object
{
public:
	glm::vec3 v0, v1, v2;
	glm::vec3 e1, e2;
	glm::vec3 normal;
	float area;
	Material* m;

	__device__ __host__ Triangle(const glm::vec3& _v0, const glm::vec3& _v1, const glm::vec3& _v2)
		: v0(_v0), v1(_v1), v2(_v2), e1(v1 - v0), e2(v2 - v0),
		normal(glm::normalize(glm::cross(e1, e2))),
		area(glm::length(glm::cross(e1, e2)) * 0.5f) {}

	__device__ void init(Material* _m)
	{
		this->m = _m;
	}

	__device__ bool intersect(const Ray& ray);
	__device__ AABB getBounds() const;
	__device__ Intersection getIntersection(Ray ray);
	__device__ bool intersect(const Ray& ray, float& tnear, uint32_t& index) const;

	__device__ void Sample(Intersection& pos, float& pdf) const {
		float x = sqrtf(get_random_float()), y = get_random_float();
		pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
		pos.normal = this->normal;
		pdf = 1.0f / area;
	}

	__device__ float getArea() const {
		return area;
	}

	__device__ bool hasEmit() const {
		return m && m->hasEmission();
	}

	__device__ glm::vec3 evalDiffuseColor(const glm::vec2& st) const {
		// Provide implementation or remove this method
		return glm::vec3(0.5, 0.5, 0.5); // Example placeholder
	}
};

class MeshTriangle : public Object
{

public:
	// host level
	uint32_t numTriangles;
	Triangle* triangles;
	glm::vec3 min_vert;
	glm::vec3 max_vert;
	float area;

	// device level
	AABB bounding_box;
	glm::vec3* vertices;
	uint32_t* vertexIndex;
	glm::vec2* stCoordinates;
	BVHAccel* bvh;
	Material* m;


	// using CPU to init
	__host__ MeshTriangle(std::string& filename): numTriangles(0)
	{
		objl::Loader loader;
		loader.LoadFile(filename);
		area = 0;
		assert(loader.LoadedMeshes.size() == 1);
		auto mesh = loader.LoadedMeshes[0];

		min_vert = glm::vec3{ std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::infinity() };
		max_vert = glm::vec3{
			-std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity() };

		numTriangles = mesh.Vertices.size();
		for (int i = 0; i < mesh.Vertices.size(); i += 3) {
			std::array<glm::vec3, 3> face_vertices;

			for (int j = 0; j < 3; j++) {
				auto vert = glm::vec3(mesh.Vertices[i + j].Position.X,
					mesh.Vertices[i + j].Position.Y,
					mesh.Vertices[i + j].Position.Z);
				face_vertices[j] = vert;

				min_vert = glm::vec3(min(min_vert.x, vert.x),
					min(min_vert.y, vert.y),
					min(min_vert.z, vert.z));
				max_vert = glm::vec3(max(max_vert.x, vert.x),
					max(max_vert.y, vert.y),
					max(max_vert.z, vert.z));
			}

			triangles[i] = Triangle(face_vertices[0], face_vertices[1], face_vertices[2]);
		}

		area = 0;
		for (size_t i = 0; i < numTriangles; i++)
		{
			area += triangles[i].area;
		}

	}

	__device__ MeshTriangle() = default;

	__device__ void init(Material* _m)
	{
		bounding_box = AABB(min_vert, max_vert);
		m = _m;
		Object** ptrs = new Object*[numTriangles];
		for (size_t i = 0; i < numTriangles; i++)
		{
			ptrs[i] = &triangles[i];
		}
		bvh = new BVHAccel(ptrs, numTriangles);
	}

	__device__ bool intersect(const Ray& ray, float& tnear, uint32_t& index) const override
	{
		bool intersect = false;
		for (uint32_t k = 0; k < numTriangles; ++k) {
			const glm::vec3 v0 = vertices[vertexIndex[k * 3]];
			const glm::vec3& v1 = vertices[vertexIndex[k * 3 + 1]];
			const glm::vec3& v2 = vertices[vertexIndex[k * 3 + 2]];
			float t, u, v;
			if (rayTriangleIntersect(v0, v1, v2, ray.origin, ray.direction, t, u, v) && t < tnear) {
				tnear = t;
				index = k;
				intersect = true; // or 'intersect |= true;' if you want to keep the bitwise operation
			}
		}

		return intersect;
	}

	__device__ AABB getBounds() const override { return bounding_box; }

	__device__ void getSurfaceProperties(const glm::vec3& P, const glm::vec3& I,
		const uint32_t& index, const glm::vec2& uv,
		glm::vec3& N, glm::vec2& st) const {
		const glm::vec3& v0 = vertices[vertexIndex[index * 3]];
		const glm::vec3& v1 = vertices[vertexIndex[index * 3 + 1]];
		const glm::vec3& v2 = vertices[vertexIndex[index * 3 + 2]];
		glm::vec3 e0 = glm::normalize(v1 - v0);
		glm::vec3 e1 = glm::normalize(v2 - v1);
		N = glm::normalize(glm::cross(e0, e1));
		const glm::vec2& st0 = stCoordinates[vertexIndex[index * 3]];
		const glm::vec2& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
		const glm::vec2& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
		st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
	}

	__device__ glm::vec3 evalDiffuseColor(const glm::vec2& st) const  override {
		float scale = 5;
		float pattern = (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
		return glm::mix(glm::vec3(0.815, 0.235, 0.031),
			glm::vec3(0.937, 0.937, 0.231), pattern);
	}

	__device__ Intersection getIntersection(Ray ray) override {
		Intersection intersec;
		if (bvh) {
			intersec = bvh->Intersect(ray);
		}
		return intersec;
	}

	__device__ void Sample(Intersection& pos, float& pdf) const override
	{
		bvh->Sample(pos, pdf);
		pos.emit = m->getEmission();
	}

	__device__ float getArea() const override {
		return area;
	}

	__device__ bool hasEmit() const override{
		return m->hasEmission();
	}

	__device__ bool intersect(const Ray& ray) override { return true; }


};

__device__ inline bool Triangle::intersect(const Ray& ray) { return true; }

__device__ inline AABB Triangle::getBounds() const { return Union(AABB(v0, v1), v2); }


__device__ inline Intersection Triangle::getIntersection(Ray ray)
{
	Intersection inter;

	// hit the back side of the triangle
	if (glm::dot(ray.direction, normal) > 0) return inter;
	double u, v, t_tmp;
	// moller algorithm
	glm::vec3 p_vec = glm::cross(ray.direction, e2);
	double det = glm::dot(e1, p_vec);
	if (fabs(det) < 0)
	{
		return inter;
	}
	double demon = 1.0 / det;
	auto s = ray.origin - v0;
	glm::vec3 s2 = glm::cross(s, e1);
	t_tmp = glm::dot(s2, e2) * demon;
	u = glm::dot(p_vec, s) * demon;
	v = glm::dot(s2, ray.direction) * demon;
	if (u > -EPSILON && v > -EPSILON && 1 - u - v > -EPSILON && t_tmp > -EPSILON)
	{
		inter.normal = normal;
		inter.obj = this;
		// distance will be using t presented
		inter.distance = t_tmp;
		inter.coords = ray.origin + ray.direction * static_cast<float>(t_tmp);
		inter.m = m;
		inter.happened = true;
	}
	return inter;

}


