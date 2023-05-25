#pragma once

#include <cmath>

struct Point {
    float x, y, z;

    __host__ __device__ inline Point() {this->x=0; this->y=0; this->z=0;};
    __host__ __device__ inline Point(float x) {this->x=x; this->y=x; this->z=x;};
    __host__ __device__ inline Point(float x, float y, float z) {this->x=x; this->y=y; this->z=z;};

    __host__ __device__ inline Point& operator+=(const Point& b) {x+=b.x; y+=b.y; z+=b.z; return *this;};
    __host__ __device__ inline Point& operator-=(const Point& b) {x-=b.x; y-=b.y; z-=b.z; return *this;};
    __host__ __device__ inline Point& operator*=(const Point& b) {x*=b.x; y*=b.y; z*=b.z; return *this;};
    __host__ __device__ inline Point& operator/=(const Point& b) {x/=b.x; y/=b.y; z/=b.z; return *this;};
    __host__ __device__ inline Point& operator*=(float b) {x*=b; y*=b; z*=b; return *this;};
    __host__ __device__ inline Point& operator/=(float b) {x/=b; y/=b; z/=b; return *this;};
};

struct Ray {
    Point origin;
    Point dir;

    __host__ __device__ Ray() {this->origin=Point(); this->dir=Point();};
    __host__ __device__ Ray(const Point& origin, const Point& dir) {this->origin=origin; this->dir=dir;};
};

__host__ __device__ 
inline Point operator+(const Point& a, const Point& b) {
    Point c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

__host__ __device__ 
inline Point operator-(const Point& a, const Point& b) {
    Point c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}

__host__ __device__
inline Point operator*(const Point& a, const Point& b) {
    Point c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    return c;
}

__host__ __device__
inline Point operator*(const Point& a, float x) {
    Point c;
    c.x = a.x * x;
    c.y = a.y * x;
    c.z = a.z * x;
    return c;
}

__host__ __device__
inline Point operator*(float x, const Point& a) {
    Point c;
    c.x = a.x * x;
    c.y = a.y * x;
    c.z = a.z * x;
    return c;
}

__host__ __device__
inline Point operator/(const Point& a, float x) {
    Point c;
    c.x = a.x / x;
    c.y = a.y / x;
    c.z = a.z / x;
    return c;
}

__host__ __device__ 
inline float dot(const Point& a, const Point& b) {
    float sum = 0.0f;
    sum += a.x * b.x;
    sum += a.y * b.y;
    sum += a.z * b.z;
    return sum;
}

__host__ __device__
inline Point cross(const Point& V, const Point& W) {
    Point N;
    N.x = V.y * W.z - V.z * W.y;
    N.y = V.z * W.x - V.x * W.z;
    N.z = V.x * W.y - V.y * W.x;
    return N;
}

__host__ __device__
inline float magnitude(const Point& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline Point normalize(const Point& a) {
    float mag = magnitude(a);
    return a / mag;
}

__host__ __device__
inline Point reflect(const Point& incident, const Point& normal) {
    Point reflected = incident - normal * (2 * dot(incident, normal));
    return normalize(reflected);
}

__host__ __device__
inline Point min(const Point& a, const Point& b) {
    return Point(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

__host__ __device__
inline Point max(const Point& a, const Point& b) {
    return Point(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__host__ __device__
inline Point inverse(const Point &a)
{
    return Point(1/a.x,1/a.y,1/a.z);
}