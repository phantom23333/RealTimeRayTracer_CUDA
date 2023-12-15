#pragma once

#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <windows.h>

struct HitRecord;

#include "Ray.h"
#include "Hittable.h"
#include "Texture.h"

#define RANDVEC3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

#define SAMPLECOUNT 1024

__device__ inline glm::vec3 random_in_unit_sphere(curandState* local_rand_state) {
	glm::vec3 p;
	float len_squared;
	do {
		// Generate each component of the vector independently
		float x = curand_uniform(local_rand_state) * 2.0f - 1.0f;
		float y = curand_uniform(local_rand_state) * 2.0f - 1.0f;
		float z = curand_uniform(local_rand_state) * 2.0f - 1.0f;
		p = glm::vec3(x, y, z);

		// Compute the length squared in a single step
		len_squared = x * x + y * y + z * z;
	} while (len_squared >= 1.0f);
	return p;
}

__device__ inline glm::vec3 random_in_hemisphere(curandState* local_rand_state, const glm::vec3 normal) {
	glm::vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);

	if (glm::dot(in_unit_sphere, normal) > 0.0) {
		return in_unit_sphere;
	}
	return -in_unit_sphere;
}


__device__ inline glm::vec2 Hammersley(curandState* state, uint32_t N = SAMPLECOUNT) { // 0-1

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Generate a random float in [0, 1)
	float randValue = curand_uniform(&state[id]);

	// Scale to [0, 1023] and convert to int
	int i = static_cast<int>(randValue * 1023.0f);

	uint32_t bits = (i << 16u) | (i >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	float rdi = float(bits) * 2.3283064365386963e-10;
	return { float(i) / float(N), rdi };
}

__device__ inline glm::vec3 random_in_cosin_hemisphere(curandState* local_rand_state, float& pdf)
{
	glm::vec2 Xi = Hammersley(local_rand_state);
	float phi = 2.0f * M_PI * Xi.x;
	float cosTheta = sqrt(Xi.y);
	float sinTheta = sqrt(1 - Xi.y);

	pdf = cosTheta / M_PI;

	return glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Tom Duff algorithim from tangent space to world space
__device__ inline void RevisedONB(const glm::vec3& n, glm::vec3& b1, glm::vec3& b2) {
	if (n.z < 0.) {
		const float a = 1.0f / (1.0f - n.z);
		const float b = n.x * n.y * a;
		b1 = glm::vec3(1.0f - n.x * n.x * a, -b, n.x);
		b2 = glm::vec3(b, n.y * n.y * a - 1.0f, -n.y);
	}
	else {
		const float a = 1.0f / (1.0f + n.z);
		const float b = -n.x * n.y * a;
		b1 = glm::vec3(1.0f - n.x * n.x * a, b, -n.x);
		b2 = glm::vec3(b, -n.y * n.y * a + 1.0f, -n.y);
	}
}

__device__ inline glm::vec3 ImportanceSampleGGX(curandState* local_rand_state, glm::vec3 N, float roughness) {
	glm::vec2 Xi = Hammersley(local_rand_state);

	float a = roughness * roughness;
	//in spherical space
	float theta = atan(a * sqrt(Xi.x) / sqrt(1.0f - Xi.x));
	float phi = 2.0f * M_PI * Xi.y;

	// from spherical space to cartesian space
	float sinTheta = sin(theta);
	float consTheta = cos(theta);
	glm::vec3 H = glm::vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, consTheta);

	glm::vec3 b1, b2;
	// tangent coordinates
	RevisedONB(N, b1, b2);

	// transform H to tangent space
	glm::vec3 sampleVec = b1 * H.x + b2 * H.y + N * H.z;
	return glm::normalize(sampleVec);
}

__device__ inline float GeometrySchlickGGX(float NdotV, float roughness)
{
	float k = roughness * roughness;
	float demon = NdotV * (1.0f - k) + k;
	return NdotV / demon;
}

__device__ inline float GeometrySmith(float roughness, float NoV, float NoL)
{
	float ggx2 = GeometrySchlickGGX(NoV, roughness);
	float ggx1 = GeometrySchlickGGX(NoL, roughness);
	return ggx1 * ggx2;
}

__device__ inline float SchlickGGX(float roughness, float NoH)
{
	float a2 = roughness * roughness;
	float demon = (NoH * NoH * (a2 - 1) + 1);
	float demon2 = M_PI * demon * demon;
	return a2 / demon2;
}


__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
	return v - 2.0f * glm::dot(v, n) * n;
}

__device__ inline bool near_zero(const glm::vec3& v) {
	float theta = 1e-8;
	return (fabs(v[0]) < theta) && (fabs(v[1]) < theta) && (fabs(v[2]) < theta);
}

__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
	float cos_theta = fminf(glm::dot(-uv, n), 1.0f);
	glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	glm::vec3 r_out_parallel = -sqrtf(fabsf(1.0 - glm::dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

__device__ inline glm::vec3 fresnel(float NdotV, glm::vec3 R0)
{
	return R0 + (glm::vec3(1.0f) - R0) * powf(1 - NdotV, 5.0f);
}

class Material {
public:
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Diffuse : public Material {
	Texture* albedo;
public:
	__device__ Diffuse(const glm::vec3& a): albedo(new SolidColor(a)) {}
	__device__ Diffuse(Texture* a) : albedo(a) {}
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		glm::vec3 scatter_direction = random_in_hemisphere(local_rand_state, rec.normal);

		if (near_zero(scatter_direction)) {
			scatter_direction = rec.normal;
		}

		scattered = Ray(rec.p, scatter_direction);
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	__device__ ~Diffuse() {
		delete albedo;
	}
};

class Glossy : public Material {
	glm::vec3 albedo;
	float fuzz;
public:
	__device__ Glossy(const glm::vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		glm::vec3 reflected = reflect(glm::normalize(r_in.direction), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
		attenuation = albedo * glm::vec3(1.25f, 1.25f, 1.25f);
		return (glm::dot(scattered.direction, rec.normal) > 0.0f);
	}
};

class Dielectric : public Material {
public:
	float ir; // index of refraction

	__device__ Dielectric(float index_of_refraction): ir(index_of_refraction) {}

	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
		glm::vec3 unit_direction = glm::normalize(r_in.direction);

		float cos_theta = fminf(glm::dot(-unit_direction, rec.normal), 1.0);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;

		glm::vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = Ray(rec.p, direction);
		return true;
	}
private:
	__device__ static float reflectance(float cosine, float ref_idx) {
		// Use Schlick's approximation for reflectance.
		float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * powf((1.0f - cosine), 5);
	}
};

class MicroFacet : public Material
{
public:
	float roughness;
	float ir;
	Texture* albedo;
	glm::vec3 R0;
	__device__ MicroFacet(float roughness, float ir, SolidColor a): roughness(roughness), ir(ir), albedo(new SolidColor(a))
	{
		R0 = glm::vec3(powf((1.0f - ir) / (1.0f + ir), 2));
	}

	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
	{
		// Yes, that is the light
		attenuation = glm::vec3(1.0f, 1.0f, 1.0f);

		auto V = glm::normalize(-r_in.direction);
		auto N = rec.normal;
		auto H = ImportanceSampleGGX(local_rand_state, N, roughness);
		auto L = glm::normalize(2.0f * H * glm::dot(V, H) - V);

		float NoL = max(glm::dot(N, L), 0.0f);
		float NoH = max(glm::dot(N, H), 0.0f);
		float VoH = max(glm::dot(V, H), 0.0f);
		float NoV = max(glm::dot(N, V), 0.0f);

		// G
		float G = GeometrySmith(roughness, NoV, NoL);

		// F
		glm::vec3 F = fresnel(NoV, R0);

		// D
		float D = SchlickGGX(roughness, NoH);

		// demon
		float demon = 4.0f * NoL * NoV;

		// BRDF
		glm::vec3 brdf = G * D / demon * F;
		float pdf = D * NoH;

		// Diffuse
		glm::vec3 diffuse = albedo->value(0.0f, 0.0f, rec.p) / M_PI;

		// cos(theta) / pdf
		attenuation = brdf * NoL / pdf + diffuse;

		scattered = Ray(rec.p, L);

		return NoL > 0.0f;
	}

};

