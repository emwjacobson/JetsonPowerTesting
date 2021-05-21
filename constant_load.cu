#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <chrono>
#include <random>

// Flops = (iterations * num_ops_per_item * blocks * threads) / time_seconds
// (14586 * 3 * 128 * 1024 ) / 5 = 1.14x10^9 FLOPS

__global__ void testKernel(float* A, float* B, float* C, unsigned long long int *timers) {
    // Add timer here
    // https://github.com/zchee/cuda-sample/blob/master/0_Simple/clock/clock.cu

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long int start = clock();
    while ((clock() - start) <= 1000000) {
        C[idx] = .997 * A[idx] + .998 * B[idx]; // 3 Floating Point Operations
    }
    // timers[idx] += clock() - start;
    timers[idx]++;
}

void print_usage(int argc, char* argv[]) {
    printf("Usage: %s [# blocks] [# threads/block] [seconds]\n", argv[0]);
}

int main(int argc, char* argv[]) {
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

    // Allocate memory on host and fill with random numbers
    float* hA = new float[threads*blocks];
    float* hB = new float[threads*blocks];
    float* hC = new float[threads*blocks];
    for(int i = 0; i < threads*blocks; i++) {
        hA[i] = float(std::rand())/float((RAND_MAX));
        hB[i] = float(std::rand())/float((RAND_MAX));
    }
    unsigned long long int* hTimers = new unsigned long long int[threads*blocks];

    // Allocate memory on GPU and copy data
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float) * threads * blocks);
    cudaMalloc(&dB, sizeof(float) * threads * blocks);
    cudaMalloc(&dC, sizeof(float) * threads * blocks);
    cudaMemcpy(dA, hA, sizeof(float) * threads * blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * threads * blocks, cudaMemcpyHostToDevice);
    unsigned long long int* dTimers;
    cudaMalloc(&dTimers, sizeof(unsigned long long int) * threads * blocks);
    cudaMemcpy(dTimers, hTimers, sizeof(unsigned long long int) * threads * blocks, cudaMemcpyHostToDevice);

    // Initialize timing variables and start timer
    float time_ms;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    // Run computations
    int i = 0;
    printf("Using %i blocks with %i threads/block for %i seconds.\n", blocks, threads, runtime);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(runtime);
    while (std::chrono::system_clock::now() < end) {
        testKernel<<<blocks, threads>>>(dA, dB, dC, dTimers);
        cudaDeviceSynchronize();
        i++;
    }

    // Calculate runtime
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&time_ms, gpu_start, gpu_stop);
    printf("Actual time: %fms over %i iterations\n", time_ms, i);

    cudaMemcpy(hC, dC, sizeof(float) * threads * blocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(hTimers, dTimers, sizeof(unsigned long long int) * threads * blocks, cudaMemcpyDeviceToHost);

    for(int i=0; i < 1; i++) printf("%llu\n", hTimers[i]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete hA, hB, hC;
}