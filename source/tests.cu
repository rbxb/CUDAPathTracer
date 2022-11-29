#include <stdio.h>

#include "constants.h"
#include "test_prefix_sum.cu"
#include "test_bucket.cu"
#include "test_obj.cpp"

int main(int argc, char* argv[]) {
    verifyCUDA(true);
    printf("\n");

    int passed = 0;
    int total = 0;

    printf("Test prefix sum: %s\n", testPrefixSum() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test bucket 1: %s\n", testBucket1() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test bucket 2: %s\n", testBucket2() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test bucket 3: %s\n", testBucket3() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test bucket 4: %s\n", testBucket4() ? "OK" + (passed++ * 0) : "FAILED"); total++;
    printf("Test read OBJ: %s\n", testObj() ? "OK" + (passed++ * 0) : "FAILED"); total++;

    printf("\nPassed %d / %d\n", passed, total);
}