#pragma once

#include "Material.h"

class Object;

struct Intersection
{
	__device__ Intersection()
	{
		happened = false;
		coords = glm::vec3();
		normal = glm::vec3();
		distance = FLT_MAX;
		obj = nullptr;
		m = nullptr;
	}

	bool happened;
	glm::vec3 coords;
	glm::vec3 tcoords;
	glm::vec3 normal;
	glm::vec3 emit;
	double distance;
	Object* obj;
	Material* m;
};
