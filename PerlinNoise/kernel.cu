
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 2
#define atoa(x) #x
__device__ int hash[256] = { 96, 131, 210, 231, 40, 179, 184, 56, 133, 209, 188, 207, 176, 245, 218, 230, 185, 70, 76, 105, 214, 182, 174, 72, 146, 159, 162, 14, 227, 160, 82, 212, 192, 191, 172, 74, 157, 236, 39, 26, 226, 201, 250, 211, 81, 254, 244, 219, 107, 161, 24, 53, 5, 154, 253, 34, 145, 197, 112, 233, 23, 68, 87, 49, 8, 156, 196, 248, 27, 149, 111, 19, 4, 62, 203, 190, 25, 132, 140, 202, 65, 16, 141, 118, 104, 153, 113, 144, 67, 175, 37, 216, 114, 60, 243, 189, 101, 150, 220, 217, 12, 252, 206, 124, 71, 103, 69, 171, 38, 126, 152, 167, 121, 178, 78, 106, 86, 15, 194, 66, 10, 237, 99, 55, 77, 13, 57, 205, 30, 44, 89, 138, 88, 41, 187, 73, 241, 221, 92, 215, 125, 168, 1, 46, 29, 239, 193, 52, 143, 251, 128, 61, 129, 45, 242, 64, 63, 213, 0, 123, 238, 43, 35, 208, 22, 33, 169, 222, 50, 51, 59, 32, 83, 20, 180, 183, 17, 108, 198, 177, 18, 80, 199, 94, 3, 9, 2, 170, 130, 186, 95, 165, 247, 204, 142, 28, 229, 102, 195, 116, 224, 163, 164, 97, 47, 36, 31, 223, 151, 225, 100, 122, 135, 136, 109, 84, 166, 249, 119, 7, 246, 155, 120, 235, 200, 181, 255, 127, 147, 21, 137, 58, 42, 75, 228, 54, 90, 232, 85, 93, 6, 79, 117, 98, 134, 173, 91, 148, 234, 240, 158, 11, 110, 48, 139, 115};

__global__ void perlin(float* noise, int* gradients)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int gradI = N*(hash[i % 256] % (sizeof(gradients)/N));
    // printf("Gradient index: %d\n", gradI);
    int currentGrad[N];
    for (int j = 0; j < N; j++) {
        currentGrad[j] = gradients[gradI + j]; // load gradients into current grad arr
        // printf("%d ", currentGrad[j]);
    }
    // printf("\n");
}

int* loadGradients() {
    int lineCount = 0;
    char pre[] = "gradients/gradient-";
    char suf[] = ".csv";
    char dim[2];
    sprintf(dim, "%d", N);
    char fileName[25];
    sprintf(fileName, "%s%s%s", pre, dim, suf);
    FILE* grad = fopen(fileName, "r");
    if (grad == NULL) {
        printf("File not found: %s\n",fileName);
        return 0;
    }
    else {
        printf("Loaded file %s\n", fileName);
    }
    char buffer[80];

    while (fgets(buffer, sizeof(buffer), grad) != NULL) { // count lines in gradient file
        lineCount++;
    }
    
    int* grads = (int*)malloc(lineCount * N * sizeof(int));

    int i = 0;
    while (fgets(buffer, 80, grad)) {
        char* token = strtok(buffer, ","); // read start of line
        while (token) {
            int n = atoi(token);
            // printf("%d\n", n);
            grads[i] = n;
            token = strtok(NULL, ",");
            i++;
        }
    }
    return grads;
}

int main()
{
    const int m = 10;
    float* noise = (float*)malloc(pow(m,N)*sizeof(float));
    float* dev_noise = 0;
    int* gradients = 0;
    int* host_gradients = loadGradients();
    cudaMalloc((void**)&dev_noise, pow(m,N) * sizeof(float));
    cudaMalloc((void**)&gradients, sizeof(host_gradients));

    cudaMemcpy(gradients, host_gradients, sizeof(host_gradients), cudaMemcpyHostToDevice);

    int n_thr = min(1024, m + (32 - (m % 32))); // round to nearest multiple of 32
    int n_blk = pow(m, N) / n_thr;
    perlin<<<n_blk, n_thr>>> (dev_noise, gradients);

    cudaFree(dev_noise);
    // free(host_gradients);
    free(noise);
    return 0;
}