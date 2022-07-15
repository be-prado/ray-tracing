/*
* This file stores material classes that dictate how a ray scatters when bouncing with a 
* material.
* The file contains classes for Lambertian materials (e.g. rubber), reflective materials (e.g. metal)
* and dielectric materials (e.g. glass).
*/

#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"
#include "hittable.h"

class material {

public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, 
									ray& scattered, curandState* rand_state)
		const = 0;
};

// Lambertian scattering material (diffusive material)
class lambertian : public material {

public:
	__device__ lambertian(const color& a) : albedo(a) {}

	__device__ bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation,
		ray& scattered, curandState* rand_state)
		const override {

		//auto scatter_direction = rec.normal + random_unit_vector();
		auto scatter_direction = random_in_hemisphere(rec.normal, rand_state);
		// catch degenerate scatter direction
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
	}

public:
	color albedo;
};


// metal material (reflective)
class metal : public material {

public:
	__device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__device__ virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, 
		ray& scattered, curandState* rand_state)
		const override {

		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

public:
	color albedo;
	float fuzz;
};

// dielectric material
class dielectric : public material {

public:
	__device__ dielectric(float refractive_index) : ri(refractive_index) {}

	__device__ virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, 
		ray& scattered, curandState* rand_state)
		const override {

		attenuation = color(1.0, 1.0, 1.0);
		float refraction_ratio = rec.front_face ? (1.0f / ri) : ri;

		vec3 out_direction;
		vec3 unit_direction = unit_vector(r_in.direction());
		float cos_theta = fminf(dot(unit_direction, -rec.normal), 1.0f);
		float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

		// check if there is a solution to Snell's law for refraction and use
		// Schlick's approximation for reflectance to see if ray is reflected
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(rand_state))
			// relfect
			out_direction = reflect(unit_direction, rec.normal);
		else
			// refract
			out_direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, out_direction);
		return true;
	}

public:
	float ri; // refraction index

private:
	__device__ static float reflectance(float cosine, float refraction_index) {
		// Schlick's approximation of reflectance
		float r0 = (1 - refraction_index) / (1 + refraction_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow(1 - cosine, 5);
	}
};

#endif // !MATERIAL_H

