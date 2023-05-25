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


bool raySphereIntersection(const Ray& ray, const Point& sphereCenter, float sphereRadius) {
    Point L = sphereCenter - ray.origin;
    float tca = dot(L, ray.dir);
    float d2 = dot(L, L) - tca * tca;
    if (d2 > sphereRadius * sphereRadius) {
        // The ray doesn't intersect the sphere
        return false;
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
            return false;
        }
    }
    // The ray intersects the sphere
    return true;
}
