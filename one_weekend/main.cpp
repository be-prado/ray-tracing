
/* A renderer that outputs ppm format text and traces rays from a camera.
* To create the image file from this, find the executable file of the project and
* type <path>/one_weekend.exe > <image_name>.ppm
*/


#include "rtweekend.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"

#include <iostream>

using namespace std;


color ray_color(const ray& r, const hittable& world, int depth) {
	hit_record rec;

	// if the ray has exceeded the bounce limit, no light is gathered
	if (depth <= 0)
		return color(0, 0, 0);

	if (world.hit(r, 0.001, infinity, rec)) {
		ray scattered;
		color attenuation;

		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_color(scattered, world, depth - 1);

		return color(0, 0, 0);
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {

	// Image
	const auto aspect_ratio = 16.0 / 9.0;
	const int image_width = 400;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int samples_per_pixel = 100;
	const int max_depth = 50;

	// World
	hittable_list world;

	auto material_ground = make_shared<lambertian>(color(42.0/256.0, 118.0/256.0, 45.0/256.0));
	auto material_center = make_shared<lambertian>(color(0.7, 0.3, 0.3));
	auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
	auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

	world.add(make_shared<sphere>(point3(0, -100.5, -1), 100.0, material_ground));
	world.add(make_shared<sphere>(point3(0, 0, -1), 0.5, material_center));
	world.add(make_shared<sphere>(point3(-1, 0, -1), 0.5, material_left));
	world.add(make_shared<sphere>(point3(1, 0, -1), 0.5, material_right));


	// Camera
	camera cam;


	cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	// Render
	// Loop through each pixel of the image from left to right and bottom to top
	// and render the image by shooting multiple rays per pixel
	for (int j = image_height - 1; j >= 0; --j) {
		cerr << "\rScanning lines remaning: " << j << ' ' << flush;
		for (int i = 0; i <= image_width - 1; ++i) {
			color pixel_color(0, 0, 0);
			// sample selected pixel
			for (int k = 0; k < samples_per_pixel; ++k) {
				// compute random variation of horizontal and vertical position of current 
				// pixel relative to the left and bottom of the image, respectively
				auto u = (i + random_double()) / (image_width - 1);
				auto v = (j + random_double()) / (image_height - 1);
				// create ray from the camera to the pixel
				ray r = cam.get_ray(u, v);
				// update pixel color
				pixel_color += ray_color(r, world, max_depth);

			}
			// write down normalized pixel_color
			write_color(cout, pixel_color, samples_per_pixel);
		}
	}

	cerr << "\nDone\n";
}