// pending_wait_repro.hip: tiny-kernel slowdown when other HIP streams hold pending waits.
//
// A main stream runs N tiny kernels, then one large memory-bound kernel. It
// times both sections with events. Afterwards, K other streams each wait on an
// event recorded at the end of the main stream's work and then run one tiny
// kernel. A spin kernel at the head of the main stream keeps the GPU behind the
// host, so every wait is already queued (pending) while the timed kernels run.
//
// Build:  hipcc -O2 --offload-arch=gfx1201 --offload-arch=gfx1030 pending_wait_repro.hip -o pending_wait_repro
// Run:    HIP_VISIBLE_DEVICES=0 ./pending_wait_repro [--mode graph|eager|both] [--max-side 4]
//         [--kernels 200] [--reps 200] [--spin-us 4000] [--big-mib 1024]
// Compare with GPU_MAX_HW_QUEUES=1/2/8 in the environment.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    hipError_t e_ = (x);                                                               \
    if (e_ != hipSuccess) {                                                            \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,               \
                   hipGetErrorString(e_));                                             \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

__global__ void tiny_add(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 0.001f;
}

// Busy-waits on the constant-rate wall clock so the GPU lags the host.
__global__ void spin(unsigned long long ticks) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long start = wall_clock64();
    while (wall_clock64() - start < ticks) {
    }
  }
}

// Memory-bound read of the whole buffer (grid-stride), one partial sum per block.
__global__ void big_read(const float4* __restrict__ src, size_t n4, float* partial) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  __shared__ float s[256];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if ((int)threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

static double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[(size_t)(p * (v.size() - 1))];
}

int main(int argc, char** argv) {
  std::string mode = "both";
  int max_side = 4, kernels = 200, reps = 200, spin_us = 4000, big_mib = 1024;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--mode") mode = argv[i + 1];
    else if (a == "--max-side") max_side = std::atoi(argv[i + 1]);
    else if (a == "--kernels") kernels = std::atoi(argv[i + 1]);
    else if (a == "--reps") reps = std::atoi(argv[i + 1]);
    else if (a == "--spin-us") spin_us = std::atoi(argv[i + 1]);
    else if (a == "--big-mib") big_mib = std::atoi(argv[i + 1]);
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }

  int dev = 0;
  CK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, dev));
  int drv = 0, rt = 0, wall_khz = 0;
  CK(hipDriverGetVersion(&drv));
  CK(hipRuntimeGetVersion(&rt));
  CK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  const char* hwq = std::getenv("GPU_MAX_HW_QUEUES");
  std::printf("device=\"%s\" arch=%s pci=%04x:%02x:%02x driver=%d runtime=%d GPU_MAX_HW_QUEUES=%s\n",
              prop.name, prop.gcnArchName, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID,
              drv, rt, hwq ? hwq : "(unset)");

  const int n_tiny = 3 * 4096;
  float* tiny = nullptr;
  CK(hipMalloc(&tiny, n_tiny * sizeof(float)));
  CK(hipMemset(tiny, 0, n_tiny * sizeof(float)));
  std::vector<float*> side_buf(max_side, nullptr);
  for (auto& b : side_buf) {
    CK(hipMalloc(&b, n_tiny * sizeof(float)));
    CK(hipMemset(b, 0, n_tiny * sizeof(float)));
  }
  size_t big_bytes = (size_t)big_mib << 20;
  size_t n4 = big_bytes / sizeof(float4);
  float4* big = nullptr;
  float* partial = nullptr;
  const int big_blocks = 4096;
  CK(hipMalloc(&big, big_bytes));
  CK(hipMemset(big, 0, big_bytes));
  CK(hipMalloc(&partial, big_blocks * sizeof(float)));

  hipStream_t main_s;
  CK(hipStreamCreateWithFlags(&main_s, hipStreamNonBlocking));
  std::vector<hipStream_t> side(max_side);
  for (auto& s : side) CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  const dim3 tb(256), tg((n_tiny + 255) / 256);
  hipGraph_t graph;
  hipGraphExec_t graph_exec;
  CK(hipStreamBeginCapture(main_s, hipStreamCaptureModeGlobal));
  for (int k = 0; k < kernels; ++k) hipLaunchKernelGGL(tiny_add, tg, tb, 0, main_s, tiny, n_tiny);
  CK(hipStreamEndCapture(main_s, &graph));
  CK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  hipEvent_t t0, t1, b1, done;
  CK(hipEventCreate(&t0));
  CK(hipEventCreate(&t1));
  CK(hipEventCreate(&b1));
  CK(hipEventCreateWithFlags(&done, hipEventDisableTiming));
  const unsigned long long spin_ticks = (unsigned long long)spin_us * wall_khz / 1000;

  std::vector<std::string> modes;
  if (mode == "both" || mode == "graph") modes.push_back("graph");
  if (mode == "both" || mode == "eager") modes.push_back("eager");

  std::printf("%-6s %5s %12s %12s %12s %12s %12s %12s\n", "mode", "waits", "tiny_p10_us",
              "tiny_p50_us", "tiny_p90_us", "big_p50_ms", "big_p90_ms", "enqueue_ms");
  for (const auto& m : modes) {
    for (int ns = 0; ns <= max_side; ++ns) {
      std::vector<double> tiny_us, big_ms;
      double max_enqueue_ms = 0;
      for (int r = 0; r < reps + 10; ++r) {
        hipLaunchKernelGGL(spin, dim3(1), dim3(64), 0, main_s, spin_ticks);
        auto h0 = std::chrono::steady_clock::now();
        CK(hipEventRecord(t0, main_s));
        if (m == "graph") {
          CK(hipGraphLaunch(graph_exec, main_s));
        } else {
          for (int k = 0; k < kernels; ++k)
            hipLaunchKernelGGL(tiny_add, tg, tb, 0, main_s, tiny, n_tiny);
        }
        CK(hipEventRecord(t1, main_s));
        hipLaunchKernelGGL(big_read, dim3(big_blocks), dim3(256), 0, main_s, big, n4, partial);
        CK(hipEventRecord(b1, main_s));
        CK(hipEventRecord(done, main_s));
        for (int s = 0; s < ns; ++s) {
          CK(hipStreamWaitEvent(side[s], done, 0));
          hipLaunchKernelGGL(tiny_add, tg, tb, 0, side[s], side_buf[s], n_tiny);
        }
        double enq = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - h0).count();
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        if (r < 10) continue;
        float a = 0, b = 0;
        CK(hipEventElapsedTime(&a, t0, t1));
        CK(hipEventElapsedTime(&b, t1, b1));
        tiny_us.push_back(1000.0 * a / kernels);
        big_ms.push_back(b);
        max_enqueue_ms = std::max(max_enqueue_ms, enq);
      }
      std::printf("%-6s %5d %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f%s\n", m.c_str(), ns,
                  pct(tiny_us, .1), pct(tiny_us, .5), pct(tiny_us, .9), pct(big_ms, .5),
                  pct(big_ms, .9), max_enqueue_ms,
                  max_enqueue_ms * 1000 > spin_us ? "  (WARN: enqueue outlasted spin)" : "");
      std::fflush(stdout);
    }
  }
  return 0;
}
