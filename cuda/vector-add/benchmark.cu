#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vectorAdd(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    float* h_a = new float[N];
    float* h_b = new float[N];

    float* h_c_cpu = new float[N];
    float* h_c_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    auto cpu_start = std::chrono::steady_clock::now();

    for (int i = 0; i < N; i++) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    
    auto cpu_end = std::chrono::steady_clock::now();

    auto cpu_time =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    auto gpu_total_start = std::chrono::steady_clock::now();
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(start_event);
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float gpu_kernel_time = 0.0f;

    cudaEventElapsedTime(
        &gpu_kernel_time,
        start_event,
        stop_event
    );
    
    std::cout << "GPU kernel time: "
              << gpu_kernel_time
              << " ms\n";

    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    auto gpu_total_end = std::chrono::steady_clock::now();

    auto gpu_total_time =
        std::chrono::duration<double, std::milli>(
            gpu_total_end - gpu_total_start
        ).count();
    
    std::cout << "GPU total time: "
              << gpu_total_time
              << " ms\n";

    std::cout << "GPU result: " << h_c_gpu[0] << std::endl;
    std::cout << "CPU result: " << h_c_cpu[0] << std::endl;

    bool correct = true;

    for (int i = 0; i < N; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "Results match: "
              << (correct ? "YES" : "NO")
              << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;

    delete[] h_c_cpu;
    delete[] h_c_gpu;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    return 0;
}