#include <stdio.h>

#include "constants.h"
#include "test_prefix_sum.cu"
#include "rotate.h"

int main(int argc, char* argv[]) {
    init(true);
    printf("\n");

    int passed = 0;
    int total = 0;

    printf("Test prefix sum: %s\n", testPrefixSum() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test rotation: %s\n", testRodriguesRotation() ? "OK" + (passed++ * 0) : "FAILED"); total++;

    printf("\nPassed %d / %d\n", passed, total);
}