# Ray Tracing

This repository implements Peter Shirley's Ray Tracing in One Weekend, which can be found in the folder "./one_weekend/". Further, I parallelized the ray tracer from the book using CUDA. For the CUDA code, I used the blog post Accelerated Ray Tracing in One Weekend in CUDA by Roger Allen as a reference. The parallelized code can be found in the folder "./one_weekend_cuda/". The tracer with CUDA feature rendered images 12 times faster than the original tracer from the book! Here are some images from the tracer: 

![Regular image](./one_weekend/images/image.ppm)

References: 
- Ray Tracing in One Weekend, https://raytracing.github.io/books/RayTracingInOneWeekend.html
- Accelerated Ray Tracing in One Weekend in CUDA, https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
