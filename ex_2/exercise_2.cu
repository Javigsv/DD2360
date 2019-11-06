#include <stdio.h>
#include <sys/time.h>
#define ARRAY_SIZE 1000000
#define TPB 256

float* saxpy(float* x, float* y, float a) {
    int i = 0;
    float* res = (float *)malloc(ARRAY_SIZE*sizeof(float)); 
    for(i = 0; i < ARRAY_SIZE; i++) {
        res[i] = a*x[i]+y[i];
    }
    return res;
}

__global__ void saxpy_gpu(float* res, float* x, float* y, float a) {
    const int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < ARRAY_SIZE) {
        res[id] = a*x[id]+y[id];
    }
}

int main() {
    float* res;
    float* res2 = (float *)malloc(ARRAY_SIZE*sizeof(float));
    float* x = (float *)malloc(ARRAY_SIZE*sizeof(float));
    float* y = (float *)malloc(ARRAY_SIZE*sizeof(float)); 
    float* res_gpu;
    float* d_x;
    float* d_y;
    float a = 10;
    struct timeval t1;
    struct timeval t2;
    suseconds_t timeCPU;
    suseconds_t timeGPU;

    printf("Filling arrays...\n");
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = i;
        y[i] = 2*i;
    }
    printf("Done!\n");

    printf("Computing in CPU...\n");
    gettimeofday(&t1, NULL);
    res = saxpy(x, y, a);
    gettimeofday(&t2, NULL);
    timeCPU = t2.tv_usec - t1.tv_usec;
    printf("Done! %ld\n", timeCPU);

    
    printf("Computing in GPU...\n");
    cudaMalloc(&res_gpu, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_x, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_y, ARRAY_SIZE*sizeof(float));
    cudaMemcpy(d_x, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    gettimeofday(&t1, NULL);
    saxpy_gpu<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(res_gpu, d_x, d_y, a);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    timeGPU = t2.tv_usec - t1.tv_usec;
    printf("Done! %ld\n", timeGPU);

    cudaMemcpy(res2, res_gpu, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(res_gpu);

    for(int i = 0; i<ARRAY_SIZE; i++) {
        //printf("%d -> %f \t %f\n", i, res[i], res2[i]);
        if(res[i] != res2[i]) {
            printf("This is bad, %d\n", i);
            exit(0);
        }
    }
    printf("Hurray!\n");
}