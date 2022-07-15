
/* A parallelized renderer that outputs ppm format text and traces rays from a camera.
* The render was written in C++ with CUDA and performs 12x faster than the non-parallelized
* version!
* To create the image file from this, find the executable file of the project and
* type <path>/one_weekend.exe > <image_name>.ppm
*/


#include <stdio.h>
#include <iostream>

#include "rtweekend.h"
#include "sphere.h"
#include "material.h"
#include "hittable_list.h"
#include "camera.h"

//Function to check for errors in CUDA.
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        std::cerr << "CUDA error string = " << cudaGetErrorString(result) << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// This computes the color of a ray as it bounces on world objects with different materials.
__device__ vec3 color_ray(const ray& r, hittable** world, int depth, curandState* rand_state) {

    ray current_ray = r;
    color current_attenuation = color(1, 1, 1);
    color attenuation; // keeps track of the attenuation in each ray bounce
    ray scattered; // keeps track of the scattered ray after each ray bounce
    hit_record rec; // keeps track of the hit record in each bounce

    // get ray color after the ray bounces *depth* amount of times
    for (int i = 0; i < depth; i++) {
        // if the ray hits something, compute its color and scattering
        if ((*world)->hit(current_ray, 0.001f, FLT_MAX, rec)) {
            // if the ray scatters, update the current ray and current atenuation
            if (rec.mat_ptr->scatter(current_ray, 
                                     rec, attenuation,
                                     scattered, 
                                     rand_state)) 
            {
                current_attenuation *= attenuation;
                current_ray = scattered;
            }
            // if the ray doesn't scatter, the material absorbs the ray
            // and the ray is given the color black
            else {
                return color(0, 0, 0);
            }  
        }
        // if the ray doesn't hit anything, color it with the sky color
        else {
            vec3 unit_direction = unit_vector(r.direction());
            auto t = 0.5f * (unit_direction.y() + 1.0f);
            color sky_color = (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
            return current_attenuation * sky_color;
        }
    }
    // if ray exceeded depth color it with the color black
    return color(0, 0, 0);
}

// Initialize CUDA random state for each pixel
__global__ void render_init(const int width, const int height, curandState* rand_state) {

    // get pixel indexes from the current block and thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    int pixel_index = j * width + i;
    // each thread is is given a different random seed with zero offset and 0 subsequence
    curand_init(1999 + pixel_index, 0, 0, &rand_state[pixel_index]);  // 1999 because we love Prince
}

// Render the image using CUDA
__global__ void render(color* fb, int width, int height, 
                       int samples_per_pixel, int depth,
                       camera** d_cam, hittable** world, 
                       curandState* rand_state
                      ) {

    // get pixel indexes from the current block and thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    int pixel_index = j * width + i;

    // get pixel's random state
    curandState local_rand_state = rand_state[pixel_index];

    // color each pixel by averaging samples from random rays
    color pixel_color = color(0, 0, 0);
    for (int k = 0; k < samples_per_pixel; k++) {
        float u = float(i) / float(width);
        float v = float(j) / float(height);

        ray r = (*d_cam)->get_ray(u, v, &local_rand_state);
        pixel_color += color_ray(r, world, depth, &local_rand_state);
    }
    // store pixel color
    fb[pixel_index] = pixel_color;
}

// Initialize the camera
__global__ void create_camera(camera** cam, point3 lookfrom, 
                              point3 lookat, vec3 vup, float vfov,
                              float aspect_ratio, float aperture,
                              float focus_distance
                             ) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *cam = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_distance);
}

// Initialize the world objects with their materials
__global__ void create_world(hittable** d_list, hittable** d_world, curandState* rand_state) {
    
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // initialize world objects random state
    curand_init(1999, 0, 0, rand_state);

    int i = 0; // current d_list index

    d_list[i++] = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(color(0.5f, 0.5f, 0.5f)));
    
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {

            float choose_mat = random_float(rand_state);
            point3 center = point3(float(a) + 0.9f * random_float(rand_state),
                                   0.2f, 
                                   float(b) + 0.9f * random_float(rand_state));

            if (choose_mat < 0.8f) {
                // diffuse
                color albedo = color::random(rand_state) * color::random(rand_state);
                d_list[i++] = new sphere(center, 0.2f, new lambertian(albedo));

            }
            else if (choose_mat < 0.95f) {
                // metal
                d_list[i++] = new sphere(center, 0.2f, new metal(color(0.8, 0.8, 0.8), 0.1f));
            }
            else {
                // glass
                d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5f));
            }
        }
    }

    d_list[i++] = new sphere(point3(0, 1, 0), 1.0f, new dielectric(1.5f));
    d_list[i++] = new sphere(point3(-4, 1, 0), 1.0f, new lambertian(color(0.4, 0.2, 0.1)));
    d_list[i++] = new sphere(point3(4, 1, 0), 1.0f, new metal(color(0.7, 0.6, 0.5), 0.0f));

    *d_world = new hittable_list(d_list, 1 + 22 * 22 + 3);
}

// Free memory by deleting pointers
__global__ void free_world(camera** d_cam, hittable** d_list, hittable** d_world) {
    for (int i = 0; i < 22 * 22 + 3 + 1; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* (d_cam);
    delete* d_world;
}


int main()
{
    // Image
    const float aspect_ratio = 3.0f / 2.0f;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int num_of_pixels = image_width * image_height;
    const int samples_per_pixel = 10;
    const int depth = 50;
    // specify CUDA block dimensions
    const int block_x = 16;
    const int block_y = 16;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << block_x << "x" << block_y << " blocks.\n";

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    float aperture = 0.1f;
    float focus_distance = 10.0f;
    float vfov = 20.0f;
    camera** d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));
    create_camera << <1, 1 >> > (d_cam, lookfrom, lookat, vup, vfov, aspect_ratio,
                                 aperture, focus_distance);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Create world

    cerr << "Creating the world..." << "\n";

    // initialize random state for the world creation
    curandState* d_rand_state_world;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_world, sizeof(curandState)));
    // allocate memory for world objects lists
    hittable** d_list;
    int num_hittables = 22 * 22 + 3 + 1; // number of world objects
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    create_world << <1, 1 >> > (d_list, d_world, d_rand_state_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    // allocate frame buffer
    size_t fb_size = num_of_pixels * sizeof(color);
    color* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    // determine number of blocks and threads per block
    dim3 blocks(image_width / block_x + 1, image_height / block_y + 1);
    dim3 threads(block_x, block_y);

    // set image rendering start time
    clock_t start, stop;
    start = clock();
    // initialize random states for each pixel
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_of_pixels * sizeof(curandState)));
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cerr << "Rendering the image..." << "\n";

    render << <blocks, threads >> > (fb, image_width, image_height, 
                                     samples_per_pixel, depth,
                                     d_cam, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // compute image rendering time
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Write image file

    cerr << "Writing image file..." << "\n";

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            write_color(std::cout, fb[pixel_index], samples_per_pixel);
        }
    }

    // Free memory
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_cam, d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state_world));
    checkCudaErrors(cudaFree(d_cam));
    delete fb;
    delete d_rand_state;
    delete d_rand_state_world;
}
