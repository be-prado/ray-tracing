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
#include <omp.h>

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		std::cerr << "Error string: " << cudaGetErrorString(result) << "\n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ color ray_color(const ray& r, hittable **d_world, int depth) {
	hit_record rec;
	ray current_ray = r;
	ray scattered;
	color current_attenuation = vec3(1, 1, 1);
	color attenuation;

	color pixel_color(1, 1, 1);

	for (int i = depth; i > 0; i--) {
		
		if ((*d_world)->hit(current_ray, 0.001f, infinity, rec)) {
			if (rec.mat_ptr->scatter(current_ray, rec, attenuation, scattered)) {
				current_ray = scattered;
				current_attenuation = current_attenuation * attenuation;
			}
			else
				return color(0, 0, 0);
		}
		else {
			vec3 unit_direction = unit_vector(current_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			color sky = (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
			return sky * current_attenuation;
		}
	}
	// if the ray has exceeded the bounce limit, no light is gathered
	return color(0, 0, 0);
}

/*
hittable_list random_scene() {

	hittable_list world;

	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0.0, -1000.0, 0.0), 1000.0f, ground_material));


	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9f * random_double(), 0.2f, b + 0.9f * random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9f) {
				shared_ptr<material> sphere_material;

				if (choose_mat < 0.8f) {
					// difuse
					auto albedo = color::random() * color::random();
					sphere_material = make_shared<lambertian>(albedo);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95f) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0.0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
				else {
					// glass
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	return world;
}
*/

__global__ void create_camera(
				camera** d_cam, point3 lookfrom, point3 lookat, vec3 vup,
				float aspect_ratio, float aperture, float focus_distance) {

	*d_cam = new camera(lookfrom, lookat, vup, 20.0f, aspect_ratio, aperture, focus_distance);

}

__global__ void create_world(hittable** d_list, hittable** d_world) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		metal material3 = metal(color(0.7, 0.6, 0.5), 0.0f);
		*(d_list) = new sphere(vec3(0, 0, -1), 0.5f, &material3);
		*(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100, &material3);
		*d_world = new hittable_list(d_list, 2);

		delete &material3;
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
	delete* (d_list);
	delete* (d_list + 1);
	delete* d_world;
}

__global__ void render_init(int max_x, int max_y, curandState* d_rand_state) {
	// get current position in the image
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	// leave the function if we are outside of the image
	if (i >= max_x || j >= max_y)
		return;
	// get pixel index
	int pixel = j * max_y + i;
	// each pixel gets a different seed
	curand_init(1984 + pixel, 0, 0, &d_rand_state[pixel]);
}


__global__ void render(
	color* fb, const int max_x, const int max_y, const int samples_per_pixel,
	camera** d_cam, hittable** d_world, int max_depth) {
	// get current position in the image
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	// leave the function if we are outside of the image
	if (i >= max_x || j >= max_y)
		return;

	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);

	// store color values in frame buffer array
	int pixel = j * max_y + i;
	color pixel_color = color(0, 0, 0);

	for (int k = 0; k < samples_per_pixel; k++) {
		ray r = (*d_cam)->get_ray(u, v);
		pixel_color += ray_color(r, d_world, max_depth);
	}

	fb[pixel] = pixel_color;

}

int main() {

	// Image
	const auto aspect_ratio = 3.0f / 2.0f;
	const int image_width = 1200;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int num_of_pixels = image_width * image_height;
	const int samples_per_pixel = 500;
	const int max_depth = 50;

	// CUDA image variables
	// size of image size frame buffer in memory
	size_t fb_size = num_of_pixels * sizeof(color);
	// allocate frame buffer
	color* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
	// block dimensions
	int block_x = 8;
	int block_y = 8;

	// World
	//auto world = random_scene();
	hittable** d_list;
	int num_of_objects = 2;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_of_objects * sizeof(hittable*)));
	hittable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
	create_world<<<1, 1>>>(d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// Camera
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	auto aperture = 0.1f;
	auto focus_distance = 10.0f;

	camera** d_cam;
	checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));
	create_camera << <1, 1 >> > (d_cam, lookfrom, lookat, vup, aspect_ratio, aperture, focus_distance);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render
	dim3 blocks(image_width / block_x + 1, image_height / block_y + 1);
	dim3 threads(block_x, block_y);
	// initialize render random seeds
	// random state for the pixels
	curandState* d_random_state;
	checkCudaErrors(cudaMalloc((void**)&d_random_state, num_of_pixels * sizeof(curandState)));
	render_init << <blocks, threads >> > (image_width, image_height, d_random_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// render image
	render << <blocks, threads >> > (fb, image_width, image_height, samples_per_pixel,
									 d_cam, d_world, max_depth);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	cudaGetErrorString(cudaGetLastError());


	// Output FB as Image
	// write pixel values from frame buffer in ppm format
	std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
	for (int j = image_height - 1; j >= 0; j--) {
		for (int i = 0; i < image_width; i++) {
			size_t pixel_index = j * image_width + i;
			write_color(cout, fb[pixel_index], samples_per_pixel);
		}
	}

	// clean up memory
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(fb));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_cam));
	cudaDeviceReset();

	cerr << "\nDone\n";
}