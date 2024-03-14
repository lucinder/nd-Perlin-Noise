
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 2
#define M 10
__device__ int hash[256] = { 96, 131, 210, 231, 40, 179, 184, 56, 133, 209, 188, 207, 176, 245, 218, 230, 185, 70, 76, 105, 214, 182, 174, 72, 146, 159, 162, 14, 227, 160, 82, 212, 192, 191, 172, 74, 157, 236, 39, 26, 226, 201, 250, 211, 81, 254, 244, 219, 107, 161, 24, 53, 5, 154, 253, 34, 145, 197, 112, 233, 23, 68, 87, 49, 8, 156, 196, 248, 27, 149, 111, 19, 4, 62, 203, 190, 25, 132, 140, 202, 65, 16, 141, 118, 104, 153, 113, 144, 67, 175, 37, 216, 114, 60, 243, 189, 101, 150, 220, 217, 12, 252, 206, 124, 71, 103, 69, 171, 38, 126, 152, 167, 121, 178, 78, 106, 86, 15, 194, 66, 10, 237, 99, 55, 77, 13, 57, 205, 30, 44, 89, 138, 88, 41, 187, 73, 241, 221, 92, 215, 125, 168, 1, 46, 29, 239, 193, 52, 143, 251, 128, 61, 129, 45, 242, 64, 63, 213, 0, 123, 238, 43, 35, 208, 22, 33, 169, 222, 50, 51, 59, 32, 83, 20, 180, 183, 17, 108, 198, 177, 18, 80, 199, 94, 3, 9, 2, 170, 130, 186, 95, 165, 247, 204, 142, 28, 229, 102, 195, 116, 224, 163, 164, 97, 47, 36, 31, 223, 151, 225, 100, 122, 135, 136, 109, 84, 166, 249, 119, 7, 246, 155, 120, 235, 200, 181, 255, 127, 147, 21, 137, 58, 42, 75, 228, 54, 90, 232, 85, 93, 6, 79, 117, 98, 134, 173, 91, 148, 234, 240, 158, 11, 110, 48, 139, 115};

__global__ void perlin(float* dnoise) // TODO - Replace srand/rand with cuRAND
{
    // printf("Called function\n");
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current point
    if (i > pow(M, N)) {
        // printf("Noise size: %d\n", sizeof(*dnoise));
        // printf("Returning from OOB index %d\n",i);
        return;
    }
    int idx = 0; // forward declare to save register
    dnoise[i] = 0.0f;
    float coords[N]; // coordinate array
    curandState state;
    curand_init(i, 0, 0, &state); // initiate using current thread
    for (int j = 0; j < N; j++) {
        coords[j] = 0.0f;
        coords[j] = curand_uniform(&state); // fill with random values
        // printf("%f", coords[j]);
    }
    // float* distances = (float*)malloc(sizeof(float)*pow(2, N)*N); // distance vector
    float* gradients = (float*)malloc(sizeof(float) * pow(2, N)); // gradient vector
    // printf("Syncing threads!\n");
    // __syncthreads(); // sync threads to avoid srand conflict
    int seed = hash[i % 256]; // reseed with pseudorandomness
    curand_init(seed, 0, 0, &state); // reseed with hashing
    for (int j = 0; j < pow(2, N); j++) {
        float euclid_dist = 0;
        float curgrad[N];
        for (int k = 0; k < N; k++) {
            curgrad[k] = 0; // initialize to 0 for safety
            curgrad[k] = curand_uniform(&state) * 2.0f - 1.0f; // load pseudorandom gradient
            // printf("%f ", curgrad[k]);
            euclid_dist += pow(curgrad[k], 2); // add to sum of squares
        }
        // printf("\n");
        // printf("%f ", gradients[j * N]);
        euclid_dist = sqrt(euclid_dist); // get euclidean distance
        int bin = ((j & (1 << (- 1))) >> (- 1));
        float dist = coords[0] - (float)bin;
        // printf("%f ", dist);
        gradients[j] = 0.0f;
        // printf("%f ", gradients[j * N]);
        for (int k = 1; k < N; k++) {
            bin = ((j & (1 << (k - 1))) >> (k - 1));
            dist = coords[k] - (float)bin;
            // printf("%d ", bin);
            gradients[j] += (curgrad[k]/euclid_dist) * dist; // calculate distance + dot product, add to dot product indices
        }
        // printf("%f ", gradients[j * N]);
    }
    // linear interpolation
    int step = 1;
    int dim = 0; // what dimension were looking at
    while (step < pow(2, N)) {
        for (int j = 0; j <= pow(2, N); j += 2 * step) {
            float f = coords[dim];
            f = f * f * f * (f * (f * 6 - 15) + 10); // apply fade function
            gradients[j] = gradients[j] + f * ((gradients[j + step]) - (gradients[j])); // interpolate along dim'th dimension
        }
        dim++;
        step *= 2;
    }
    // in theory gradients[0] should have our final value
    // printf("%f\n", gradients[0]);
    dnoise[i] = gradients[0];
    free(gradients);
    
    if (dnoise[i] == 0.0 || dnoise[i] < -1.0) {
        printf("Dangerous value detected at thread %d\n", i);
    }
    
    // printf("%f\n", noise[i]);
    // printf("Completed noisegen at index %d\n", i);
}

int main()
{
    printf("Matrix size: %d^%d\n", M, N);
    if (pow(INT_MAX, 1.0 / (N)) < M) {
        printf("ERROR: matrix size out of bounds!\n");
        return;
    }
    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEvent_t start2, stop2;
    float elapsedTime, elapsedTimeWithMem;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    const int mSize = pow(M, N);
    const int mBytes = mSize * sizeof(float);
    // printf("Bytes: %d\n", mBytes);

    cudaEventRecord(start, 0);
    float* noise = (float*)malloc(mBytes);
    float* dev_noise;
    if (cudaMalloc((void**)&dev_noise, mBytes) != cudaSuccess) {
        printf("Failed to allocate device memory!\n");
    }

    int n_thr = min(512, M); // if we can do our whole matrix in 1 block, do it, otherwise use 1024 thr/block
    int n_blk = (mSize / n_thr);
    if ((pow(M, N) / (float)n_thr) != (int)(pow(M, N) / (float)n_thr)) {
        printf("Adding kernel space for non-32-multiple matrix\n");
        n_blk++;
    }
    printf("Kernel configuration: %d blocks, %d threads.\n", n_blk, n_thr);
    
    cudaEventRecord(start2, 0);
    perlin<<<n_blk, n_thr>>>(dev_noise);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime, start2, stop2);
    printf("Runtime (noise generation only): %.7f seconds\n", elapsedTime/1000.0);
    
    cudaMemcpy(noise, dev_noise, mBytes, cudaMemcpyDeviceToHost);
    /*
    for (int i = 0; i < pow(m, N); i++) {
        printf("%f\n", noise[i]);
    }
    */
    // printf("Noise generation complete. Loading noise to file.\n");

    FILE* fptr;
    fptr = fopen("perlin_out.txt", "w+");
    fprintf(fptr, "%d,%d\n",M,N);
    for (int i = 0; i < mSize; i++) {
        if ((noise[i] <= -1.0 || noise[i] == 0.0)) {
            printf("WARNING: Noise values may be incorrect.\n");
        }
        fprintf(fptr,"%f\n",noise[i]);
    }
    cudaFree(dev_noise);
    // printf("Device memory freed.\n");
    // free(host_gradients);
    free(noise);
    // printf("Host memory freed.\n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeWithMem, start, stop);
    printf("Runtime with memory + filewrite operations: %.7f seconds\n", elapsedTimeWithMem/1000.0);
    return 0;
}