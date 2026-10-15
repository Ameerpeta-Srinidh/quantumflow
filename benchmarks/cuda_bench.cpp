// CUDA Monte Carlo benchmark (only built when nvcc is available).
//   * throughput vs block size (32 ... 1024 threads per block)
//   * memory layout experiment: SoA vs AoS path storage
//   * latency for 10,000 paths (the README headline number)
//   * variance reduction on the GPU
// Writes CSVs to results/. Run under Nsight Systems / Compute for profiles:
//   nsys profile -o profiles/qf_timeline ./build/cuda_bench
//   ncu --set full -o profiles/qf_kernel ./build/cuda_bench
#include <cstdio>
#include <fstream>

#include "../src/common/heston_analytic.hpp"
#include "../src/cuda/heston_mc.cuh"
#include "bench_util.hpp"

using namespace qf;
using namespace qf::cuda;

int main(int argc, char** argv) {
    const std::string out = bench::results_dir(argc, argv);
    if (!device_available()) {
        std::printf("no CUDA device found\n");
        return 1;
    }
    std::printf("device: %s\n", device_name());
    const HestonParams p;
    std::printf("analytic reference: %.6f\n", heston_call_analytic(p));

    McConfig cfg;
    cfg.n_paths = 1 << 22;
    cfg.n_steps = 256;
    heston_mc_cuda(p, cfg);  // warm-up (context creation, module load)

    std::ofstream fb(out + "/cuda_block_sweep.csv");
    fb << "block_size,grid,kernel_ms,paths_per_sec\n";
    for (int bs = 32; bs <= 1024; bs *= 2) {
        KernelStats st;
        const McResult r = heston_mc_cuda(p, cfg, bs, PathLayout::kNone, &st);
        std::printf("block %4d  grid %5d  kernel %8.3f ms  %8.1f M paths/s  price %.5f +- %.5f\n", bs, st.grid_size,
                    st.kernel_ms, st.paths_per_sec / 1e6, r.price, r.std_error);
        fb << bs << "," << st.grid_size << "," << st.kernel_ms << "," << st.paths_per_sec << "\n";
    }

    // Precision experiment: float vs double paths (sums are double either way).
    std::ofstream fp(out + "/cuda_precision.csv");
    fp << "precision,kernel_ms,paths_per_sec,price,std_error\n";
    for (int sp = 1; sp >= 0; --sp) {
        McConfig pc = cfg;
        pc.gpu_single_precision = sp;
        KernelStats st;
        const McResult r = heston_mc_cuda(p, pc, 256, PathLayout::kNone, &st);
        std::printf("%-6s kernel %8.3f ms  %8.1f M paths/s  price %.5f +- %.5f\n", sp ? "float" : "double",
                    st.kernel_ms, st.paths_per_sec / 1e6, r.price, r.std_error);
        fp << (sp ? "float" : "double") << "," << st.kernel_ms << "," << st.paths_per_sec << "," << r.price << ","
           << r.std_error << "\n";
    }

    // Layout experiment: writing every step of every path to global memory.
    std::ofstream fl(out + "/cuda_layout.csv");
    fl << "layout,kernel_ms,effective_GBps\n";
    McConfig lc = cfg;
    lc.n_paths = 1 << 18;
    lc.n_steps = 256;
    const char* names[] = {"registers_only", "SoA", "AoS"};
    for (int l = 0; l < 3; ++l) {
        KernelStats st;
        heston_mc_cuda(p, lc, 256, PathLayout(l), &st);
        const double bytes = (lc.n_paths / 2.0) * lc.n_steps * sizeof(float);
        const double gbps = l == 0 ? 0.0 : bytes / (st.kernel_ms * 1e-3) / 1e9;
        std::printf("layout %-15s kernel %8.3f ms   write bandwidth %.1f GB/s\n", names[l], st.kernel_ms, gbps);
        fl << names[l] << "," << st.kernel_ms << "," << gbps << "\n";
    }

    // Headline: 10,000 paths end to end.
    McConfig small = cfg;
    small.n_paths = 10000;
    KernelStats st;
    heston_mc_cuda(p, small, 256, PathLayout::kNone, &st);
    std::printf("10,000 paths x 256 steps: kernel %.3f ms, end-to-end %.3f ms\n", st.kernel_ms, st.total_ms);

    // Variance reduction on the GPU.
    std::ofstream fv(out + "/cuda_variance_reduction.csv");
    fv << "method,price,std_error\n";
    for (int a = 0; a < 2; ++a)
        for (int c = 0; c < 2; ++c) {
            McConfig v = cfg;
            v.antithetic = a;
            v.control_variate = c;
            const McResult r = heston_mc_cuda(p, v);
            std::printf("antithetic=%d cv=%d  price %.5f  SE %.6f\n", a, c, r.price, r.std_error);
            fv << (a ? "anti" : "plain") << (c ? "+cv" : "") << "," << r.price << "," << r.std_error << "\n";
        }
    return 0;
}
