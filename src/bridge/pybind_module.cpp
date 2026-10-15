// QuantumFlow - C++ <-> Python bridge (pybind11).
//
// Exposes the pricing engines and the order book to Python as the module
// `qf_core`. Large arrays cross the boundary without copying:
//  * inputs are taken as py::array_t<double> views straight onto NumPy's buffer
//    (the buffer protocol); a copy only happens if the caller passes a
//    non-contiguous or non-float64 array
//  * outputs are allocated in C++ and handed to NumPy together with a
//    "capsule" that frees them, so NumPy (and torch.from_numpy) use the C++
//    memory directly
// Long computations release the GIL so other Python threads keep running.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../common/heston_analytic.hpp"
#include "../common/mc_greeks.hpp"
#include "../cpu/heston_mc_cpu.hpp"
#include "../orderbook/feed_simulator.hpp"
#include "../orderbook/order_book.hpp"
#ifdef QF_WITH_CUDA
#include "../cuda/heston_mc.cuh"
#endif

namespace py = pybind11;
using namespace qf;

namespace {

bool cuda_available() {
#ifdef QF_WITH_CUDA
    return qf::cuda::device_available();
#else
    return false;
#endif
}

McResult run_mc(const HestonParams& p, const McConfig& cfg, const std::string& backend, int block_size) {
    if (backend == "cpu") return heston_mc_cpu(p, cfg);
    if (backend == "cuda") {
#ifdef QF_WITH_CUDA
        if (!qf::cuda::device_available()) throw std::runtime_error("backend='cuda' but no CUDA device was found");
        return qf::cuda::heston_mc_cuda(p, cfg, block_size);
#else
        throw std::runtime_error("qf_core was built without CUDA; rebuild with an nvcc toolchain or use backend='cpu'");
#endif
    }
    if (backend == "auto") return cuda_available() ? run_mc(p, cfg, "cuda", block_size) : heston_mc_cpu(p, cfg);
    throw std::invalid_argument("backend must be 'cpu', 'cuda' or 'auto'");
}

McConfig make_cfg(std::uint64_t n_paths, int n_steps, std::uint64_t seed, bool antithetic, bool cv) {
    McConfig c;
    c.n_paths = n_paths;
    c.n_steps = n_steps;
    c.seed = seed;
    c.antithetic = antithetic;
    c.control_variate = cv;
    return c;
}

// Wrap a heap buffer as a NumPy array that owns it (zero copy).
py::array_t<double> owning_array(double* data, py::ssize_t rows, py::ssize_t cols) {
    py::capsule owner(data, [](void* ptr) { delete[] static_cast<double*>(ptr); });
    return py::array_t<double>({rows, cols}, {cols * (py::ssize_t)sizeof(double), (py::ssize_t)sizeof(double)}, data,
                               owner);
}

HestonParams row_to_params(const double* r) {
    HestonParams p;
    p.S0 = r[0]; p.K = r[1]; p.T = r[2]; p.r = r[3]; p.v0 = r[4];
    p.kappa = r[5]; p.theta = r[6]; p.xi = r[7]; p.rho = r[8];
    return p;
}

using InArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check_param_matrix(const InArray& a) {
    if (a.ndim() != 2 || a.shape(1) != 9)
        throw std::invalid_argument("expected an (N, 9) array with columns S0, K, T, r, v0, kappa, theta, xi, rho");
}

}  // namespace

PYBIND11_MODULE(qf_core, m) {
    m.doc() = "QuantumFlow core: Heston pricing engines (CPU / CUDA) and a low-latency order book";

    py::class_<HestonParams>(m, "HestonParams")
        .def(py::init([](double S0, double K, double T, double r, double v0, double kappa, double theta, double xi,
                         double rho) {
                 return HestonParams{S0, K, T, r, v0, kappa, theta, xi, rho};
             }),
             py::arg("S0") = 100.0, py::arg("K") = 100.0, py::arg("T") = 1.0, py::arg("r") = 0.03,
             py::arg("v0") = 0.04, py::arg("kappa") = 2.0, py::arg("theta") = 0.04, py::arg("xi") = 0.5,
             py::arg("rho") = -0.7)
        .def_readwrite("S0", &HestonParams::S0)
        .def_readwrite("K", &HestonParams::K)
        .def_readwrite("T", &HestonParams::T)
        .def_readwrite("r", &HestonParams::r)
        .def_readwrite("v0", &HestonParams::v0)
        .def_readwrite("kappa", &HestonParams::kappa)
        .def_readwrite("theta", &HestonParams::theta)
        .def_readwrite("xi", &HestonParams::xi)
        .def_readwrite("rho", &HestonParams::rho)
        .def("__repr__", [](const HestonParams& p) {
            return "HestonParams(S0=" + std::to_string(p.S0) + ", K=" + std::to_string(p.K) +
                   ", T=" + std::to_string(p.T) + ", r=" + std::to_string(p.r) + ", v0=" + std::to_string(p.v0) +
                   ", kappa=" + std::to_string(p.kappa) + ", theta=" + std::to_string(p.theta) +
                   ", xi=" + std::to_string(p.xi) + ", rho=" + std::to_string(p.rho) + ")";
        });

    m.def("cuda_available", &cuda_available, "True if built with CUDA and a GPU is present");
    m.def("cuda_device_name", []() -> std::string {
#ifdef QF_WITH_CUDA
        return qf::cuda::device_name();
#else
        return "none (built without CUDA)";
#endif
    });

    m.def("bs_call", &bs_call, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          "Black-Scholes European call price");

    m.def("heston_price", &heston_call_analytic, py::arg("params"),
          "Semi-closed-form Heston European call price (the reference answer)");

    m.def(
        "heston_greeks",
        [](const HestonParams& p) {
            const Greeks g = heston_greeks_analytic(p);
            return py::dict(py::arg("price") = g.price, py::arg("delta") = g.delta, py::arg("gamma") = g.gamma,
                            py::arg("theta") = g.theta, py::arg("vega") = g.vega, py::arg("rho") = g.rho);
        },
        py::arg("params"));

    m.def(
        "heston_greeks_batch",
        [](InArray params) {
            check_param_matrix(params);
            const py::ssize_t n = params.shape(0);
            const double* in = params.data();
            double* out = new double[n * 6];
            {
                py::gil_scoped_release nogil;
#pragma omp parallel for schedule(dynamic, 64)
                for (py::ssize_t i = 0; i < n; ++i) {
                    const Greeks g = heston_greeks_analytic(row_to_params(in + 9 * i));
                    double* o = out + 6 * i;
                    o[0] = g.price; o[1] = g.delta; o[2] = g.gamma; o[3] = g.theta; o[4] = g.vega; o[5] = g.rho;
                }
            }
            return owning_array(out, n, 6);
        },
        py::arg("params"),
        "Analytic price + 5 Greeks for an (N, 9) parameter matrix. Returns (N, 6): price, delta, gamma, theta, "
        "vega, rho.");

    m.def(
        "mc_price",
        [](const HestonParams& p, std::uint64_t n_paths, int n_steps, std::uint64_t seed, bool antithetic,
           bool control_variate, const std::string& backend, int block_size) {
            McResult r;
            {
                py::gil_scoped_release nogil;
                r = run_mc(p, make_cfg(n_paths, n_steps, seed, antithetic, control_variate), backend, block_size);
            }
            return py::dict(py::arg("price") = r.price, py::arg("std_error") = r.std_error,
                            py::arg("beta") = r.beta, py::arg("n_samples") = r.n_samples,
                            py::arg("elapsed_ms") = r.elapsed_ms);
        },
        py::arg("params"), py::arg("n_paths") = 1 << 18, py::arg("n_steps") = 128, py::arg("seed") = 42,
        py::arg("antithetic") = true, py::arg("control_variate") = true, py::arg("backend") = "auto",
        py::arg("block_size") = 256, "Monte Carlo Heston call price");

    m.def(
        "mc_greeks",
        [](const HestonParams& p, std::uint64_t n_paths, int n_steps, std::uint64_t seed, const std::string& backend) {
            McGreeks g;
            {
                py::gil_scoped_release nogil;
                g = mc_bump_greeks(p, make_cfg(n_paths, n_steps, seed, true, true),
                                   [&](const HestonParams& q, const McConfig& c) { return run_mc(q, c, backend, 256); });
            }
            return py::dict(py::arg("price") = g.g.price, py::arg("delta") = g.g.delta,
                            py::arg("gamma") = g.g.gamma, py::arg("theta") = g.g.theta, py::arg("vega") = g.g.vega,
                            py::arg("rho") = g.g.rho, py::arg("elapsed_ms") = g.elapsed_ms,
                            py::arg("n_runs") = g.n_runs);
        },
        py::arg("params"), py::arg("n_paths") = 1 << 16, py::arg("n_steps") = 128, py::arg("seed") = 42,
        py::arg("backend") = "auto", "Greeks by bump-and-reprice Monte Carlo with common random numbers");

    m.def(
        "mc_greeks_batch",
        [](InArray params, std::uint64_t n_paths, int n_steps, std::uint64_t seed, const std::string& backend) {
            check_param_matrix(params);
            const py::ssize_t n = params.shape(0);
            const double* in = params.data();
            double* out = new double[n * 6];
            {
                py::gil_scoped_release nogil;
                for (py::ssize_t i = 0; i < n; ++i) {
                    const McGreeks g = mc_bump_greeks(
                        row_to_params(in + 9 * i), make_cfg(n_paths, n_steps, seed, true, true),
                        [&](const HestonParams& q, const McConfig& c) { return run_mc(q, c, backend, 256); });
                    double* o = out + 6 * i;
                    o[0] = g.g.price; o[1] = g.g.delta; o[2] = g.g.gamma;
                    o[3] = g.g.theta; o[4] = g.g.vega; o[5] = g.g.rho;
                }
            }
            return owning_array(out, n, 6);
        },
        py::arg("params"), py::arg("n_paths") = 1 << 15, py::arg("n_steps") = 64, py::arg("seed") = 42,
        py::arg("backend") = "auto", "Monte Carlo bump Greeks for an (N, 9) matrix. Returns (N, 6).");

    // ---------------- order book ----------------
    py::enum_<ob::Side>(m, "Side").value("BUY", ob::Side::kBuy).value("SELL", ob::Side::kSell);

    py::class_<ob::TopOfBook>(m, "TopOfBook")
        .def_readonly("bid_px", &ob::TopOfBook::bid_px)
        .def_readonly("ask_px", &ob::TopOfBook::ask_px)
        .def_readonly("bid_qty", &ob::TopOfBook::bid_qty)
        .def_readonly("ask_qty", &ob::TopOfBook::ask_qty);

    py::class_<ob::OrderBook>(m, "OrderBook")
        .def(py::init<std::int32_t, double>(), py::arg("n_ticks") = 1 << 14, py::arg("tick_size") = 0.01)
        .def("add", &ob::OrderBook::add, py::arg("id"), py::arg("side"), py::arg("price"), py::arg("qty"),
             "Add a limit order (price in ticks). Returns traded quantity.")
        .def("cancel", &ob::OrderBook::cancel, py::arg("id"))
        .def("execute", &ob::OrderBook::execute, py::arg("id"), py::arg("qty"))
        .def("best_bid", &ob::OrderBook::best_bid)
        .def("best_ask", &ob::OrderBook::best_ask)
        .def("top", &ob::OrderBook::top)
        .def("mid_price", &ob::OrderBook::mid_price)
        .def("spread", &ob::OrderBook::spread)
        .def("depth_at", &ob::OrderBook::depth_at)
        .def("live_orders", &ob::OrderBook::live_orders)
        .def("trades", &ob::OrderBook::trades)
        .def(
            "run_feed",
            [](ob::OrderBook& book, std::uint64_t n_messages, std::uint64_t seed, int mid_every) {
                // Generates a synthetic feed, plays it into the book, and samples the
                // mid price every `mid_every` messages. Returns (mids, total_ns).
                ob::FeedConfig cfg;
                cfg.n_messages = n_messages;
                cfg.seed = seed;
                cfg.center_tick = book.n_ticks() / 2;
                const auto feed = ob::generate_feed(cfg, book.n_ticks());
                std::vector<double> mids;
                mids.reserve(n_messages / (mid_every > 0 ? mid_every : 1) + 1);
                const auto t0 = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < feed.size(); ++i) {
                    book.apply(feed[i]);
                    if (mid_every > 0 && i % mid_every == 0) mids.push_back(book.mid_price());
                }
                const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
                return py::make_tuple(py::array_t<double>(mids.size(), mids.data()), ns);
            },
            py::arg("n_messages") = 1'000'000, py::arg("seed") = 7, py::arg("mid_every") = 1000);
}
