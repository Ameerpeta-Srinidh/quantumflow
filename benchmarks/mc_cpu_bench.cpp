// CPU Monte Carlo benchmark: throughput vs threads, variance reduction, and
// the cost of bump-and-reprice Greeks. Writes CSVs to results/.
#include <cstdio>
#include <fstream>
#include <thread>

#include "bench_util.hpp"
#include "common/heston_analytic.hpp"
#include "common/mc_greeks.hpp"
#include "cpu/heston_mc_cpu.hpp"

using namespace qf;

int main(int argc, char** argv) {
    const std::string out = bench::results_dir(argc, argv);
    const HestonParams p;  // defaults: ATM 1y call, kappa 2, theta 0.04, xi 0.5, rho -0.7
    const double exact = heston_call_analytic(p);
    std::printf("reference analytic price: %.6f\n", exact);
    const int max_threads = (int)std::thread::hardware_concurrency();

    // ---- throughput vs threads ----
    std::ofstream ft(out + "/mc_cpu_threads.csv");
    ft << "threads,paths_per_sec,ms\n";
    McConfig cfg;
    cfg.n_paths = 1 << 20;
    cfg.n_steps = 256;
    for (int t = 1; t <= max_threads; t *= 2) {
        heston_mc_cpu(p, cfg, t);  // warm-up
        const McResult r = heston_mc_cpu(p, cfg, t);
        const double pps = cfg.n_paths / (r.elapsed_ms * 1e-3);
        std::printf("threads %2d: %8.1f ms  %.2f M paths/s  (256 steps)\n", t, r.elapsed_ms, pps / 1e6);
        ft << t << "," << pps << "," << r.elapsed_ms << "\n";
    }

    // ---- variance reduction at an equal number of simulated paths ----
    std::ofstream fv(out + "/mc_variance_reduction.csv");
    fv << "method,n_paths,price,std_error,variance_ratio_vs_plain,ms\n";
    cfg.n_paths = 1 << 20;
    cfg.n_steps = 128;
    struct M { const char* name; bool a, c; } methods[] = {
        {"plain", false, false}, {"antithetic", true, false}, {"control_variate", false, true}, {"antithetic+cv", true, true}};
    double var_plain = 0;
    std::printf("\nvariance reduction (%llu paths, 128 steps):\n", (unsigned long long)cfg.n_paths);
    for (const auto& m : methods) {
        cfg.antithetic = m.a;
        cfg.control_variate = m.c;
        const McResult r = heston_mc_cpu(p, cfg, max_threads);
        const double var = r.std_error * r.std_error;
        if (var_plain == 0) var_plain = var;
        std::printf("  %-16s price %.5f  SE %.5f  variance reduction %5.2fx  (%.0f ms)\n", m.name, r.price,
                    r.std_error, var_plain / var, r.elapsed_ms);
        fv << m.name << "," << cfg.n_paths << "," << r.price << "," << r.std_error << "," << var_plain / var << ","
           << r.elapsed_ms << "\n";
    }

    // ---- convergence: error vs number of paths ----
    std::ofstream fc(out + "/mc_convergence.csv");
    fc << "n_paths,method,price,std_error,abs_error\n";
    for (std::uint64_t n = 1 << 12; n <= (1u << 22); n <<= 2)
        for (const auto& m : {methods[0], methods[3]}) {
            cfg.n_paths = n;
            cfg.antithetic = m.a;
            cfg.control_variate = m.c;
            const McResult r = heston_mc_cpu(p, cfg, max_threads);
            fc << n << "," << m.name << "," << r.price << "," << r.std_error << "," << std::fabs(r.price - exact) << "\n";
        }

    // ---- Greeks: bump-and-reprice cost ----
    cfg.n_paths = 1 << 16;
    cfg.n_steps = 128;
    cfg.antithetic = cfg.control_variate = true;
    const McGreeks g = mc_bump_greeks(p, cfg, [&](const HestonParams& q, const McConfig& c) { return heston_mc_cpu(q, c, max_threads); });
    std::printf("\nMC bump Greeks (%d runs x %llu paths): %.1f ms on %d threads\n", g.n_runs,
                (unsigned long long)cfg.n_paths, g.elapsed_ms, max_threads);
    std::ofstream fg(out + "/mc_greeks_cost.csv");
    fg << "n_paths,n_steps,threads,ms\n" << cfg.n_paths << "," << cfg.n_steps << "," << max_threads << "," << g.elapsed_ms << "\n";
    return 0;
}
