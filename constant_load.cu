#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <chrono>
#include <random>
#include <iostream>

void print_usage(int argc, char* argv[]) {
    printf("Usage: %s [dim] [runtime]\n", argv[0]);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argc, argv);
        return 0;
    }

    int matrix_dim = std::stoi(argv[1]);
    int runtime = std::stoi(argv[2]);

    // Allocate memory on host and fill with random numbers
    float *A, *B, *C;
    cudaMallocManaged(&A, sizeof(float) * matrix_dim * matrix_dim);
    cudaMallocManaged(&B, sizeof(float) * matrix_dim * matrix_dim);
    cudaMallocManaged(&C, sizeof(float) * matrix_dim * matrix_dim);
    for(int i = 0; i<matrix_dim; i++) {
        A[i] = float(std::rand())/float((RAND_MAX));
        B[i] = float(std::rand())/float((RAND_MAX));
        C[i] = float(std::rand())/float((RAND_MAX));
    }

    // Initialize timing variables and start timer
    float time_ms;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Run computations
    int i = 0;
    float alpha = 1, beta = 0;
    printf("Using a %i x %i matrix for %i seconds.\n", matrix_dim, matrix_dim, runtime);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(runtime);

    cudaEventRecord(gpu_start);
    while (std::chrono::system_clock::now() < end) {
        cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                matrix_dim,
                matrix_dim,
                matrix_dim,
                &alpha, A, matrix_dim,
                B, matrix_dim,
                &beta, C, matrix_dim);
        cudaDeviceSynchronize();
        i++;
    }
    cudaEventRecord(gpu_stop);

    // Calculate runtime
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&time_ms, gpu_start, gpu_stop);
    printf("Actual time: %fms over %i iterations\n", time_ms, i);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}