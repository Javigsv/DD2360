#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#define NUM_PARTICLES 1000000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 64
#define SEED 787

typedef struct Particle {
    float3 position;
    float3 velocity;
} Particle;

float floatRand(float max) {
    return ((float)rand()/RAND_MAX)*max;
}

void initParticle(Particle* p) {
    float x = floatRand(5.0);
    float y = floatRand(5.0);
    float z = floatRand(5.0);
    float vx = floatRand(5.0);
    float vy = floatRand(5.0);
    float vz = floatRand(5.0);
    p->position = make_float3(x, y, z);
    p->velocity = make_float3(vx, vy, vz);
}

__host__ __device__ float gen_random(int seed, int p, int i) {
    return (seed*p+i) % NUM_PARTICLES;
}

void timeStepCPU(Particle* p, float dt, int iter) {
    float x;
    float y;
    float z;
    float3 pos;
    float3 vel;
    float randomV;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        pos = p[i].position;
        x = pos.x;
        y = pos.y;
        z = pos.z;
        randomV = gen_random(SEED, i, iter);
        p[i].velocity.x = randomV*0.2;
        p[i].velocity.y = randomV*0.5;
        p[i].velocity.z = randomV*0.3;
        vel = p[i].velocity;
        p[i].position = make_float3(x+vel.x*dt, y+vel.y*dt, z+vel.z*dt);
    }
}

__device__ void timeStepGPU(Particle* p, float dt, int iter) {
    float x;
    float y;
    float z;
    float3 pos;
    float3 vel;
    float randomV;
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    pos = p[id].position;
    x = pos.x;
    y = pos.y;
    z = pos.z;
    randomV = gen_random(SEED, id, iter);
    p[id].velocity.x = randomV*0.2;
    p[id].velocity.y = randomV*0.5;
    p[id].velocity.z = randomV*0.3;
    vel = p[id].velocity;
    p[id].position = make_float3(x+vel.x*dt, y+vel.y*dt, z+vel.z*dt);
}

__global__ void executeGPU(Particle* p, float dt) {
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        timeStepGPU(p, dt, i);
    }
}

int main(int argc, char* argv[]){
    Particle* particles = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    int i = 0;
    Particle* particlesGPU;
    Particle* solutionGPU = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));

    struct timeval t1;
    struct timeval t2;
    suseconds_t timeCPU;
    suseconds_t timeGPU;

    srand((unsigned) time(NULL)); 

    gettimeofday(&t1, NULL);
    cudaMalloc(&particlesGPU, NUM_PARTICLES*sizeof(Particle));
    cudaMemcpy(particlesGPU, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
    gettimeofday(&t2, NULL);
    timeGPU = t2.tv_usec - t1.tv_usec;

    printf("Init: (%f, %f, %f)\n", particles[0].position.x, particles[0].position.y, particles[0].position.z);
    //CPU
    printf("Computing in the CPU...\n");
    gettimeofday(&t1, NULL);
    for(i = 0; i < NUM_ITERATIONS; i++) {
        timeStepCPU(particles, 1, i);
    }
    gettimeofday(&t2, NULL);
    timeCPU = t2.tv_usec - t1.tv_usec;
    //printf("%d: (%f, %f, %f)\n", i, particles[0].position.x, particles[0].position.y, particles[0].position.z);
    printf("Done! %ld\n", timeCPU);

    //GPU
    printf("Computing in the GPU...\n");
    gettimeofday(&t1, NULL);
    executeGPU<<<(NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(particlesGPU, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(solutionGPU, particlesGPU, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
    gettimeofday(&t2, NULL);
    timeGPU += t2.tv_usec - t1.tv_usec;
    //printf("%d: (%f, %f, %f)\n", i, solutionGPU[0].position.x, solutionGPU[0].position.y, solutionGPU[0].position.z);
    printf("Done! %ld\n", timeGPU);

    printf("Comparing results...\n");
    for(i = 0; i < NUM_PARTICLES; i++) {
        float xCPU = particles[i].position.x;
        float yCPU = particles[i].position.y;
        float zCPU = particles[i].position.z;
        float xGPU = solutionGPU[i].position.x;
        float yGPU = solutionGPU[i].position.y;
        float zGPU = solutionGPU[i].position.z;
        if(xCPU != xGPU | yCPU != yGPU | zCPU != zGPU) {
            printf("CPU: (%f, %f, %f)\n", xCPU, yCPU, zCPU);
            printf("GPU: (%f, %f, %f)\n", xGPU, yGPU, zGPU);
            printf("Something is bad %d\n", i);
            exit(0);
        }
    }
    printf("Everything seems fine\n");
}