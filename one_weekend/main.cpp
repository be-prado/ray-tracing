
/* A renderer that outputs ppm format text and traces rays from a camera.
* To create the image file from this, find the executable file of the project and
* type <path>/one_weekend.exe > <image_name>.ppm
*/


#include "rtweekend.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"

#include <iostream>

using namespace std;


color ray_color(const ray& r, const hittable& world) {
	hit_record rec;
	
	if (world.hit(r, 0, infinity, rec)) {
		return 0.5 * (rec.normal + color(1, 1, 1));
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

	// World
	hittable_list world;
	world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
	world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

	// Camera
	camera cam;


	cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	// Render
	// Loop through each pixel of the image from left to right and bottom to top
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
				pixel_color += ray_color(r, world);

			}
			// write down normalized pixel_color
			write_color(cout, pixel_color, samples_per_pixel);
		}
	}

	cerr << "\nDone\n";
}