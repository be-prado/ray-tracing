#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {

public:
	sphere() {}
	sphere(point3 cen, double r) : center(cen), radius(r) {}

	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;

public:
	point3 center;
	double radius;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {

	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(r.direction(), oc);
	auto c = oc.length_squared() - radius * radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) 
		return false;

	auto sqrtd = sqrt(discriminant);

	// find nearest root that is between t_min and t_max
	auto root = (-half_b - sqrtd) / a;
	if (root < t_min || root > t_max) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max)
			return false;
	}

	rec.t = root;
	rec.p = r.at(root);
	vec3 outward_normal = unit_vector(rec.p - center);
	rec.set_face_normal(r, outward_normal);

	return true;
}

#endif

