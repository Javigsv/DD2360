#include <stdio.h>
#define B 1
#define TPB 256


__device__ uint whoami() {
    return blockIdx.x*blockDim.x+threadIdx.x;
}

__global__ void greetings() {
    uint id = whoami();
    printf("Hello world! My threadId is %d\n", id);
}

int main() {
    greetings<<<B, TPB>>>();
    cudaDeviceSynchronize();
}