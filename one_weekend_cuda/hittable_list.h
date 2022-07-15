
/*
List of hittable objects, a child class of hittable.
A hittable object can be added to this list, the list can be cleared and
one may check if a ray hits any of the hittable objects in the list.
*/

#pragma once

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using namespace std;

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {

public:
	__device__ hittable_list() {}
	__device__ hittable_list(hittable** object_list, int size) : objects(object_list), list_size(size) {}

	// check if the ray r hits a hittable object in the list in the time constraint
	// [t_min, t_max]
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
	// vector of hittable objects
	hittable** objects;
	int list_size;
};


__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	hit_record temp_rec;

	bool hit_anything = false;
	auto closest_so_far = t_max;

	// find closest hittable hit by the ray
	for (int i = 0; i < list_size; i++) {
		// check if  hittable object is hit by the ray in closer distance than other objects
		if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

#endif