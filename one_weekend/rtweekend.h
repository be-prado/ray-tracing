
/*
* Header file that stores:
* 1) useful constants - pi, infinity.
* 2) useful functions - degrees_to_radians, random_double, clamp.
* 3) common headers - ray.h, vec3.h
*/

#pragma once

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>


// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility functions
inline double degrees_to_radians(double degrees) {
	return pi * degrees / 180.0;
}
// random double in [0, 1)
inline double random_double() {
	return rand() / (RAND_MAX + 1.0); // RAND_MAX = 32767
}
// random double in [min,max)
inline double random_double(double min, double max) {
	return min + random_double() * (max - min);
}
// clamps the value x to the range [min,max]
inline double clamp(double x, double min, double max) {
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}

// Common headers
#include "ray.h"
#include "vec3.h"

#endif

