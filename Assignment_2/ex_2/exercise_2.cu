#include <stdio.h>
#include <sys/time.h>
#define ARRAY_SIZE 1000000
#define TPB 256
#define MARGIN 1e-6

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

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
    double t1;
    double t2;
    double timeCPU;
    double timeGPU;

    printf("Filling arrays...\n");
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = i;
        y[i] = 2*i;
    }
    printf("Done!\n");

    printf("Computing in CPU...\n");
    t1 = cpuSecond();
    res = saxpy(x, y, a);
    t2 = cpuSecond();
    timeCPU = t2 - t1;
    printf("Done! %f s\n", timeCPU);

    
    printf("Computing in GPU...\n");
    cudaMalloc(&res_gpu, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_x, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_y, ARRAY_SIZE*sizeof(float));
    cudaMemcpy(d_x, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    t1 = cpuSecond();
    saxpy_gpu<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(res_gpu, d_x, d_y, a);
    cudaDeviceSynchronize();
    t2 = cpuSecond();
    timeGPU = t2 - t1;
    printf("Done! %f s\n", timeGPU);

    cudaMemcpy(res2, res_gpu, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(res_gpu);

    for(int i = 0; i<ARRAY_SIZE; i++) {
        //printf("%d -> %f \t %f\n", i, res[i], res2[i]);
        if(res[i] - res2[i] > MARGIN) {
            printf("This is bad, %d\n", i);
            exit(0);
        }
    }
    printf("Hurray!\n");
}