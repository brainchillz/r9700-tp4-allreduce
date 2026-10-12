// gpu-context-keeper: hold an idle HIP context on the R9700 forever.
// On this host (R9700, gfx1201) a vLLM/Radiance process that is the ONLY HIP context on the card
// decodes at ~50 ms/step; with any other context resident it runs at ~32 ms/step (measured
// 2026-10-08, measured on a single-card box). This process is that other context.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <unistd.h>
__global__ void touch(int *p) { p[threadIdx.x] = threadIdx.x; }
int main() {
    int *d = nullptr;
    if (hipMalloc(&d, 256 * sizeof(int)) != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
    touch<<<1, 256>>>(d);
    if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
    int dev = -1; hipGetDevice(&dev);
    hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
    printf("gpu-context-keeper: holding context on device %d (%s, %s)\n", dev, p.name, p.gcnArchName);
    fflush(stdout);
    for (;;) pause();
}
