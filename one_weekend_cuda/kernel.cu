#include <stdio.h>
#include <iostream>

#include "rtweekend.h"
#include "sphere.h"
#include "material.h"
#include "hittable_list.h"
#include "camera.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(hittable** d_list, hittable** d_world) {

    lambertian mat1 = lambertian(color(1, 0, 0));
    *(d_list) = new sphere(point3(0, 0, -1), 0.5, &mat1);
    *(d_list + 1) = new sphere(point3(0, -100.5, -1), 100, &mat1);
    *(d_world) = new hittable_list(d_list, 2);
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete* (d_list);
    delete* (d_list+1);
    delete* d_world;
    delete d_list;
    delete d_world;
}

__device__ vec3 color_ray(const ray& r, hittable** world) {

    hit_record rec;

    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
        return 0.5f * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render_init(const int width, const int height, curandState* rand_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height)
        return;

    int pixel_index = j * width + i;
    // each thread is is given a different random seed with zero offset and 0 subsequence
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(color* fb, int width, int height, vec3 llc, vec3 hor, 
                        vec3 vert, vec3 origin, hittable** world, curandState* rand_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= width || j >= height)
        return;

    int pixel_index = j * width + i;
    // get thread's random state
    curandState local_rand_state = rand_state[pixel_index];

    float u = float(i) / float(width);
    float v = float(j) / float(height);

    ray r(origin, llc + u * hor + v * vert - origin);

    fb[pixel_index] = color_ray(r, world);
        
}


int main()
{
    // Image
    const float aspect_ratio = 16.0f / 9.0f;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    const int num_of_pixels = image_width * image_height;

    size_t fb_size = num_of_pixels * sizeof(color);

    // Camera
    auto viewport_height = 2.0f;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0f;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    camera** d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));

    // allocate FB
    color* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Render
    const int block_x = 8;
    const int block_y = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << block_x << "x" << block_y << " blocks.\n";

    // set image rendering start time
    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width / block_x + 1, image_height / block_y + 1);
    dim3 threads(block_x, block_y);

    // create world
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // initialize random states for each pixel
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_of_pixels * sizeof(curandState)));
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);

    // render image
    render << <blocks, threads >> > (fb, image_width, image_height, lower_left_corner,
                                     horizontal, vertical, origin, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // compute image rendering time
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    // write image file
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            
            size_t pixel_index = j * image_width + i;

            write_color(std::cout, fb[pixel_index], 1);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));
    delete fb;
    delete d_rand_state;
}
