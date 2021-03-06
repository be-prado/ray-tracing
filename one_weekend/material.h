#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"

struct hit_record;

class material {

public:
	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
		const = 0;
};

// Lambertian scattering material (diffusive material)
class lambertian : public material {
	
public:
	lambertian(const color& a) : albedo(a) {}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
		const override {
		//auto scatter_direction = rec.normal + random_unit_vector();
		auto scatter_direction = random_in_hemisphere(rec.normal);

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
	metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
		const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

public:
	color albedo;
	double fuzz;
};

// dielectric material
class dielectric : public material {

public:
	dielectric(double refractive_index) : ri(refractive_index) {}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
		const override {

		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ri) : ri;

		vec3 out_direction;
		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(unit_direction, -rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
		bool cannot_refract = refraction_ratio * sin_theta > 1.0;

		// check if there is a solution to Snell's law for refraction and use
		// Schlick's approximation for reflectance to see if ray is reflected
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
			// relfect
			out_direction = reflect(unit_direction, rec.normal);
		else
			// refract
			out_direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, out_direction);
		return true;
	}

public:
	double ri; // refraction index

private:
	static double reflectance(double cosine, double refraction_index) {
		// Schlick's approximation of reflectance
		auto r0 = (1 - refraction_index) / (1 + refraction_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow(1 - cosine, 5);
	}
};

#endif // !MATERIAL_H

