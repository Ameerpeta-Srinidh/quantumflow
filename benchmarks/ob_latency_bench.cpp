// Order book latency benchmark.
//
// Plays a synthetic market-data feed into the book and measures:
//   * per-message latency (p50 / p95 / p99 / p99.9), by message type
//   * throughput (messages per second, no per-message timer)
//   * best bid/ask query cost from a reader's point of view
// Writes results/ob_latency_hist.csv and results/ob_summary.csv.
#include <cstdio>
#include <fstream>

#include "bench_util.hpp"
#include "orderbook/feed_simulator.hpp"
#include "orderbook/order_book.hpp"

using namespace qf;
using namespace qf::ob;

int main(int argc, char** argv) {
    const std::string out = bench::results_dir(argc, argv);
    const std::int32_t n_ticks = 1 << 14;
    FeedConfig cfg;
    cfg.n_messages = 5'000'000;
    cfg.center_tick = n_ticks / 2;
    std::printf("generating %llu messages...\n", (unsigned long long)cfg.n_messages);
    const auto feed = generate_feed(cfg, n_ticks);
    const double npt = bench::ns_per_tick();
    const double overhead = bench::timer_overhead_ticks();
    std::printf("timer: %.3f ns/tick, overhead %.0f ticks (%.1f ns), subtracted from samples\n", npt, overhead,
                overhead * npt);

    // ---- 1) per-message latency ----
    std::vector<double> lat_all, lat_add, lat_cancel, lat_exec;
    lat_all.reserve(feed.size());
    {
        OrderBook book(n_ticks);
        book.reserve_ids(feed.size() + 1);
        const std::size_t warm = 500'000;  // fill the book and warm caches/pools first
        for (std::size_t i = 0; i < warm; ++i) book.apply(feed[i]);
        for (std::size_t i = warm; i < feed.size(); ++i) {
            const auto t0 = bench::ticks();
            book.apply(feed[i]);
            const auto t1 = bench::ticks();
            const double ns = std::max(0.0, (double(t1 - t0) - overhead) * npt);
            lat_all.push_back(ns);
            (feed[i].type == MsgType::kAdd ? lat_add : feed[i].type == MsgType::kCancel ? lat_cancel : lat_exec)
                .push_back(ns);
        }
        std::printf("book after run: %zu live orders, %llu trades, bid %d ask %d\n", book.live_orders(),
                    (unsigned long long)book.trades(), book.best_bid(), book.best_ask());
    }

    // ---- 2) throughput: whole feed, one timer around everything ----
    double msgs_per_sec = 0;
    {
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            OrderBook book(n_ticks);
            book.reserve_ids(feed.size() + 1);
            const auto c0 = std::chrono::steady_clock::now();
            for (const auto& m : feed) book.apply(m);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
            best = std::min(best, s);
        }
        msgs_per_sec = feed.size() / best;
    }

    // ---- 3) top-of-book query cost ----
    double top_ns, bbo_ns;
    {
        OrderBook book(n_ticks);
        for (std::size_t i = 0; i < 500'000; ++i) book.apply(feed[i]);
        const int N = 10'000'000;
        volatile std::int64_t sink = 0;
        auto t0 = bench::ticks();
        for (int i = 0; i < N; ++i) sink = sink + book.best_bid() + book.best_ask();
        auto t1 = bench::ticks();
        bbo_ns = double(t1 - t0) * npt / N;
        t0 = bench::ticks();
        for (int i = 0; i < N; ++i) { const auto t = book.top(); sink = sink + t.bid_qty; }
        t1 = bench::ticks();
        top_ns = double(t1 - t0) * npt / N;
    }

    auto show = [](const char* name, const std::vector<double>& v) {
        const auto p = bench::percentiles(v);
        std::printf("%-8s n=%-8zu p50 %6.1f  p90 %6.1f  p95 %6.1f  p99 %6.1f  p99.9 %7.1f  mean %6.1f ns\n", name,
                    v.size(), p.p50, p.p90, p.p95, p.p99, p.p999, p.mean);
        return p;
    };
    std::printf("\nper-message latency (ns):\n");
    const auto pa = show("all", lat_all);
    const auto pad = show("add", lat_add);
    const auto pc = show("cancel", lat_cancel);
    const auto pe = show("execute", lat_exec);
    std::printf("\nthroughput: %.2f M messages/sec (single thread)\n", msgs_per_sec / 1e6);
    std::printf("best bid+ask read: %.2f ns   seqlock top-of-book snapshot: %.2f ns\n", bbo_ns, top_ns);

    // Histogram, 1 ns bins up to 1000 ns.
    std::vector<long> hist(1001, 0);
    for (double x : lat_all) hist[std::min<std::size_t>(1000, std::size_t(x))]++;
    std::ofstream h(out + "/ob_latency_hist.csv");
    h << "ns,count\n";
    for (std::size_t i = 0; i < hist.size(); ++i) h << i << "," << hist[i] << "\n";
    std::ofstream s(out + "/ob_summary.csv");
    s << "metric,value\n"
      << "p50_ns," << pa.p50 << "\np95_ns," << pa.p95 << "\np99_ns," << pa.p99 << "\np999_ns," << pa.p999
      << "\nadd_p99_ns," << pad.p99 << "\ncancel_p99_ns," << pc.p99 << "\nexecute_p99_ns," << pe.p99
      << "\nthroughput_msgs_per_sec," << msgs_per_sec << "\nbbo_query_ns," << bbo_ns << "\ntop_snapshot_ns,"
      << top_ns << "\n";
    std::printf("wrote %s/ob_latency_hist.csv and ob_summary.csv\n", out.c_str());
    return 0;
}
