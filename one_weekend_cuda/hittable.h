/*
* This file defines the blueprint for a hittable class. The class has a function that
* determines if a ray hits the object and stores the hit information in a hit record.
*/

#pragma once

#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"

class material;

// This stores information from when a ray hits an object
struct hit_record {
	point3 p;
	vec3 normal;
	float t;
	material* mat_ptr;
	bool front_face;

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif