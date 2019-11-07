#include <stdio.h>
#include <sys/time.h>
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 256

typedef struct Particle {
    float3 position;
    float3 velocity;
} Particle;

void timeStepCPU(Particle* p, float dt) {
    float x;
    float y;
    float z;
    float3 pos;
    float3 vel;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        pos = p[i].position;
        x = pos.x;
        y = pos.y;
        z = pos.z;
        vel = p[i].velocity;
        p[i].position = make_float3(x+vel.x*dt, y+vel.y*dt, z+vel.z*dt);
    }
}

__global__ void timeStepGPU(Particle* p, float dt) {
    float x;
    float y;
    float z;
    float3 pos;
    float3 vel;
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    pos = p[id].position;
    x = pos.x;
    y = pos.y;
    z = pos.z;
    vel = p[id].velocity;
    p[id].position = make_float3(x+vel.x*dt, y+vel.y*dt, z+vel.z*dt);
}

int main(){
    Particle* particles = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    
}