/*
* Header file that stores:
* 1) useful constants - pi, infinity.
* 2) useful functions - degrees_to_radians, random_float, clamp.
* 4) common structures - hit_record
* 3) common headers - ray.h, vec3.h
*/

#pragma once

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
__device__ const float infinity = std::numeric_limits<float>::infinity();
#ifndef M_PI
#define M_PI 3.1415926535897932385f
#endif // !M_PI

// Utility functions
__host__ __device__ inline float degrees_to_radians(float degrees) {
	return M_PI * degrees / 180.0f;
}
// random float in [0, 1)
__device__ inline float random_float(curandState* rand_state) {
	float result = curand_uniform(rand_state);

	while (result == 1.0f) {
		result = curand_uniform(rand_state);
	}
	return result;
}
// random float in [min,max)
__device__ inline float random_float(float min, float max, curandState* rand_state) {
	return min + random_float(rand_state) * (max - min);
}
// clamps the value x to the range [min,max]
__host__ __device__ inline float clamp(float x, float min, float max) {
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}

// Common headers
#include "ray.h"
#include "vec3.h"
#include "color.h"

#endif

