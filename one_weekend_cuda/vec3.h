#pragma once

#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include "cuda_runtime.h"


class vec3 {
public:
    __host__ __device__ vec3() : e{ 0, 0, 0 } {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(const float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    // Return true if vector length is less than a small epsilon
    __host__ __device__ bool near_zero() const {
        const auto epsilon = 1e-8;
        return length_squared() < epsilon * epsilon;
    }
    // compute random vector with components in [0,1)
    __device__ inline static vec3 random(curandState* rand_state) {
        return vec3(random_float(rand_state), random_float(rand_state), random_float(rand_state));
    }
    // compute random vector with components in [min,max)
    __device__ inline static vec3 random(float min, float max, curandState* rand_state) {
        return vec3(random_float(min, max, rand_state), 
                    random_float(min, max, rand_state), 
                    random_float(min, max, rand_state));
    }

public:
    float e[3];
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

#endif

// vec3 Utility Functions

inline std::istream& operator>>(std::istream& in, vec3& v) {
    in >> v.e[0] >> v.e[1] >> v.e[2];
    return in;
}

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1.0f / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

/*___________________RANDOM VECTOR GENERATORS_________________________*/

// compute random vector in unit sphere
__device__ inline vec3 random_in_unit_sphere(curandState* rand_state) {
    while (true) {
        auto p = vec3::random(-1, 1, rand_state);
        // check if p is in the unit sphere
        if (p.length_squared() >= 1) continue;
        return p;
    }
}
// compute random unit vector
__device__ inline vec3 random_unit_vector(curandState* rand_state) {
    return unit_vector(random_in_unit_sphere(rand_state));
}
// compute random vectorlying on a hemisphere of the unit ball specified 
// by a normal vector
__device__ inline vec3 random_in_hemisphere(const vec3& normal, curandState* rand_state) {
    vec3 p = random_in_unit_sphere(rand_state);
    // check if p is in the hemisphere specified by the normal direction
    if (dot(p, normal) < 0.0)
        p = -p;
    return p;
}
// compute random vector in the 2D x,y unit disl
__device__ inline vec3 random_in_unit_disk(curandState* rand_state) {
    while (true) {
        vec3 p = vec3(random_float(-1, 1, rand_state), random_float(-1, 1, rand_state), 0);
        if (p.length_squared() >= 1.0) continue;
        return p;
    }
}
/*___________________END OF RANDOM VECTOR GENERATORS_______________________*/



// compute the reflected ray from a reflected material
__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}
// compute refracted ray using Snell's law and the refractive index ratio of
// two materials
__host__ __device__ inline vec3 refract(const vec3& uv, const vec3 n, float refractive_ratio) {
    auto cos_theta_in = fminf(dot(uv, -n), 1.0f);
    vec3 r_out_perp = refractive_ratio * (uv + cos_theta_in * n);
    vec3 r_out_parallel = -sqrt(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}