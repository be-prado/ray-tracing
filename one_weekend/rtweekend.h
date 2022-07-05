
/*
* Header file that stores:
* 1) useful constants - pi, infinity.
* 2) useful functions - degrees_to_radians.
* 3) common headers - ray.h, vec3.h
*/

#pragma once

#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
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

// Common headers
#include "ray.h"
#include "vec3.h"

#endif

