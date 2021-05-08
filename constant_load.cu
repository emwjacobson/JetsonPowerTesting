#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <random>

// 1    128     -> 10%

__global__ void testKernel(float* A, float* B, float* C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float t_a;
    float t_b;
    for(int j=0; j<10; j++) {
        for(int i=0; i<10; i++) {
            t_a = 1.01 * A[idx];
        }
        for(int i=0; i<10; i++) {
            t_b = 0.99 * B[idx];
        }
    }

    C[idx] = 0.1 * A[idx] + 0.55 * B[idx] + t_a + t_b;

    for(int i=0; i<10; i++)
        C[idx] = 0.999 * C[idx];
}

void print_usage(int argc, char* argv[]) {
    printf("Usage: %s [# blocks] [# threads/block] [seconds]\n", argv[0]);
}

int main(int argc, char* argv[]) {
    // TB with 1000 threads vs 100
    // Differing blocks 1 vs #SMs
    if (argc != 4) {
        print_usage(argc, argv);
        return 0;
    }

    int blocks = std::stoi(argv[1]);
    int threads = std::stoi(argv[2]);
    int runtime = std::stoi(argv[3]); // In seconds

    if (blocks <= 0 || threads <= 0) {
        print_usage(argc, argv);
        return 0;
    }

    float* hA = new float[threads*blocks];
    float* hB = new float[threads*blocks];
    float* hC = new float[threads*blocks];
    for(int i = 0; i < threads*blocks; i++) {
        hA[i] = float(std::rand())/float((RAND_MAX));
        hB[i] = float(std::rand())/float((RAND_MAX));
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float) * threads * blocks);
    cudaMalloc(&dB, sizeof(float) * threads * blocks);
    cudaMalloc(&dC, sizeof(float) * threads * blocks);
    cudaMemcpy(dA, hA, sizeof(float) * threads * blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * threads * blocks, cudaMemcpyHostToDevice);

    printf("Using %i blocks, %i threads/block for %i seconds.\n", blocks, threads, runtime);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(runtime);
    while (std::chrono::system_clock::now() < end) {
        testKernel<<<blocks, threads>>>(dA, dB, dC);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(hC, dC, sizeof(float) * threads * blocks, cudaMemcpyDeviceToHost);

    // for(int i=0; i < threads*blocks; i++) printf("%f\n", hC[i]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete hA, hB, hC;
}