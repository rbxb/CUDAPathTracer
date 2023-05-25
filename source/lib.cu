#include <cuda.h>
#include <cuda_runtime_api.h>

#include "constants.h"
#include "point.h"

void boundingSphere(Point* points, int numPoints, Point& center, float& radius) {
    // First find the point with the smallest x-coordinate
	int minIndex = 0;
	for (int i = 1; i < numPoints; i++) {
		if (points[i].x < points[minIndex].x) {
			minIndex = i;
		}
	}

	// Find the point with the largest distance from the first point
	int maxIndex = 0;
	float maxDist = 0.0f;
	for (int i = 0; i < numPoints; i++) {
		float dist = magnitude(points[i] - points[minIndex]);
		if (dist > maxDist) {
			maxIndex = i;
			maxDist = dist;
		}
	}

	// Find the point with the largest distance from the previous point
	int furthestIndex = 0;
	float furthestDist = 0.0f;
	for (int i = 0; i < numPoints; i++) {
		float dist = magnitude(points[i] - points[maxIndex]);
		if (dist > furthestDist) {
			furthestIndex = i;
			furthestDist = dist;
		}
	}

	// Use the points with the smallest x-coordinate and the largest and furthest distances
	// to create a sphere that encloses all of the points
	Point A = points[minIndex];
	Point B = points[maxIndex];
	Point C = points[furthestIndex];

	// Find the center and radius of the sphere
	center = (A + B + C) * (1.0f / 3.0f);
	radius = max(maxDist, furthestDist);
}

void raySphereIntersection(const Ray* rays, const int numRays, const Point& sphereCenter, float sphereRadius, bool* out) {
    for (int i = 0; i < numRays; i++) {
        out[i] = false;
        Point L = sphereCenter - rays[i].origin;
        float tca = dot(L, rays[i].dir);
        float d2 = dot(L, L) - tca * tca;
        if (d2 > sphereRadius * sphereRadius) {
            // The ray doesn't intersect the sphere
            continue;
        }
        float thc = sqrt(sphereRadius * sphereRadius - d2);
        float t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        if (t0 < 0) {
            t0 = t1;
            if (t0 < 0) {
                // The sphere is behind the ray
                continue;
            }
        }

        out[i] = true
    }
}

void simpleIntersect(const Ray* rays, const int* indices, const int numIndices, const Point* points, const int numFaces, bool* out) {
    for (int i = 0; i < numIndices; i++) {
        Ray& ray = rays[indices[i]];

        for (int k = 0; k < numFaces; k++) {
            Point& A = points[k * 3];
            Point& B = points[k * 3 + 1];
            Point& C = points[k * 3 + 2];
        }
    }
}