#pragma once

#include <cmath>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include <iostream>
#include <random>

#define EPSILON 0.000001f

__device__ inline bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr == 0)
        x0 = x1 = -0.5 * b / a;
    else
    {
        float q = (b > 0) ? -0.5 * (b + sqrt(discr)) : -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1)
	    thrust::swap(x0, x1);
    return true;
}

enum MaterialType
{
    DIFFUSE,
	MICROFACET,
    REFLECTION_AND_REFRACTION,
    REFLECTION
};

__device__ inline float get_random_float(unsigned int seed = 0) {
    curandState state;
    curand_init(seed, 0, 0, &state);  // Initialize curand
    return curand_uniform(&state);    // Generate a random float between 0.0 and 1.0
}

