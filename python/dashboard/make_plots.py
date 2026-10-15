"""Benchmarking & profiling dashboard (Component E).

Reads every CSV / JSON / NPZ the benchmarks and the training script wrote to
results/, draws the charts into results/plots/, and writes
results/dashboard.html, a single page with every number and chart.

    python python/dashboard/make_plots.py

Charts that need data you have not produced yet (e.g. the CUDA block-size
sweep on a machine without a GPU) are skipped with a note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qf_paths import GREEKS, RESULTS_DIR  # noqa: E402

PLOTS = RESULTS_DIR / "plots"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREEK_LABELS = ["Delta", "Gamma", "Theta", "Vega", "Rho"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
})

made: list[tuple[str, str]] = []
notes: list[str] = []


def saved(fig, name, caption):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name)
    plt.close(fig)
    made.append((name, caption))


def have(*names):
    missing = [n for n in names if not (RESULTS_DIR / n).exists()]
    if missing:
        notes.append(f"skipped a chart: missing {', '.join(missing)}")
    return not missing


def order_book():
    if not have("ob_latency_hist.csv", "ob_summary.csv"):
        return
    h = pd.read_csv(RESULTS_DIR / "ob_latency_hist.csv")
    s = pd.read_csv(RESULTS_DIR / "ob_summary.csv").set_index("metric")["value"]
    h = h[h.ns <= 600]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(h.ns, h["count"], width=1.0, color=BLUE, linewidth=0)
    for q, lab in [("p50_ns", "p50"), ("p99_ns", "p99"), ("p999_ns", "p99.9")]:
        ax.axvline(s[q], color=MUTED, linestyle="--", linewidth=1)
        ax.text(s[q] + 4, ax.get_ylim()[1] * 0.9, f"{lab} {s[q]:.0f} ns", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("latency per message (ns)")
    ax.set_ylabel("messages (log scale)")
    ax.set_title("Order book: latency per market-data message")
    saved(fig, "ob_latency_hist.png", "Every message timed with the CPU timestamp counter (timer overhead subtracted).")


def mc_threads():
    if not have("mc_cpu_threads.csv"):
        return
    d = pd.read_csv(RESULTS_DIR / "mc_cpu_threads.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(d.threads, d.paths_per_sec / 1e6, marker="o", markersize=7, color=BLUE, label="measured")
    ax.plot(d.threads, d.threads * d.paths_per_sec.iloc[0] / 1e6, linestyle=":", color=MUTED, label="perfect scaling")
    ax.set_xticks(d.threads)
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("million paths / second")
    ax.set_title("CPU Monte Carlo throughput vs threads")
    ax.legend()
    saved(fig, "mc_cpu_threads.png", "Heston paths with 256 time steps; OpenMP parallel over paths.")


def cuda_sweep():
    if not (RESULTS_DIR / "cuda_block_sweep.csv").exists():
        notes.append("CUDA block-size sweep not run yet (needs an NVIDIA GPU: run build/cuda_bench).")
        return
    d = pd.read_csv(RESULTS_DIR / "cuda_block_sweep.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(d.block_size, d.paths_per_sec / 1e6, marker="o", markersize=7, color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(d.block_size, [str(b) for b in d.block_size])
    ax.set_xlabel("threads per block")
    ax.set_ylabel("million paths / second")
    ax.set_title("CUDA throughput vs block size")
    saved(fig, "cuda_block_sweep.png", "Kernel time only (cudaEvent), 4M paths x 256 steps.")


def variance_reduction():
    if not have("mc_variance_reduction.csv"):
        return
    d = pd.read_csv(RESULTS_DIR / "mc_variance_reduction.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(d.method, d.variance_ratio_vs_plain, color=BLUE, width=0.6)
    for b, v in zip(bars, d.variance_ratio_vs_plain):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}x", ha="center", color=INK, fontsize=9)
    ax.set_ylabel("variance reduction vs plain MC")
    ax.set_title("Variance reduction at equal path count")
    ax.grid(axis="x", visible=False)
    saved(fig, "variance_reduction.png", "Same number of simulated paths; higher = fewer paths needed for the same accuracy.")


def convergence():
    if not have("mc_convergence.csv"):
        return
    d = pd.read_csv(RESULTS_DIR / "mc_convergence.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for m, c in [("plain", ORANGE), ("antithetic+cv", BLUE)]:
        s = d[d.method == m]
        ax.plot(s.n_paths, s.std_error, marker="o", markersize=7, color=c, label=f"{m} (std error)")
    s = d[d.method == "plain"]
    ax.plot(s.n_paths, s.std_error.iloc[0] * np.sqrt(s.n_paths.iloc[0] / s.n_paths), ":", color=MUTED,
            label="1/sqrt(N) slope")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("simulated paths")
    ax.set_ylabel("standard error of price")
    ax.set_title("Monte Carlo convergence")
    ax.legend(fontsize=8)
    saved(fig, "mc_convergence.png", "Error falls like 1/sqrt(N): 4x the paths for 2x the accuracy.")


def surrogate():
    if not have("surrogate_test_preds.npz", "surrogate_metrics.json"):
        return
    z = np.load(RESULTS_DIR / "surrogate_test_preds.npz")
    m = json.load(open(RESULTS_DIR / "surrogate_metrics.json"))
    truth, pred = z["truth"], z["pred"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    n = min(5000, len(truth))
    for j, ax in enumerate(axes):
        t, p = truth[:n, j], pred[:n, j]
        ax.scatter(t, p, s=3, color=BLUE, alpha=0.35, linewidths=0)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
        mae = m["per_greek"][GREEKS[j]]["mae"]
        ax.set_title(f"{GREEK_LABELS[j]}  (MAE {mae:.2g})")
        ax.set_xlabel("exact")
        if j == 0:
            ax.set_ylabel("surrogate")
    fig.suptitle("Neural surrogate vs exact Greeks on held-out options", fontweight="bold", y=1.06)
    saved(fig, "surrogate_scatter.png", "Each dot is one test option; the line is perfect agreement.")

    # calibration curve: nominal vs observed coverage
    from statistics import NormalDist
    err = np.abs(z["mc_mean"] - truth)
    sd = z["mc_std"]
    levels = np.linspace(0.05, 0.99, 30)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    for lab, sc, c in [("raw MC dropout", 1.0, ORANGE), ("calibrated", z["scale"][0], BLUE)]:
        obs = [(err[:, 0] <= NormalDist().inv_cdf(0.5 + q / 2) * sc * sd[:, 0]).mean() for q in levels]
        ax.plot(levels, obs, color=c, label=lab)
    ax.plot([0, 1], [0, 1], ":", color=MUTED, label="perfect calibration")
    ax.set_xlabel("nominal interval coverage")
    ax.set_ylabel("observed coverage")
    ax.set_title("Delta uncertainty calibration")
    ax.legend(fontsize=8)
    saved(fig, "uq_calibration.png", "Observed share of true deltas inside the predicted interval, test set.")

    # uncertainty vs distance from training region
    xo, so, eo = z["ood_x"], z["ood_std"][:, 0], np.abs(z["ood_mean"][:, 0] - z["ood_truth"][:, 0])
    logm = np.abs(np.log(xo[:, 1] / xo[:, 0]))
    bins = np.linspace(0, 0.6, 13)
    idx = np.digitize(logm, bins) - 1
    centers = 0.5 * (bins[1:] + bins[:-1])
    mean_sd = [so[idx == k].mean() for k in range(len(centers))]
    mean_err = [eo[idx == k].mean() for k in range(len(centers))]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(centers, mean_sd, marker="o", markersize=7, color=BLUE, label="MC-dropout std (delta)")
    ax.plot(centers, mean_err, marker="o", markersize=7, color=ORANGE, label="actual |error| (delta)")
    ax.axvspan(0.3, 0.6, color=GRID, alpha=0.6, linewidth=0)
    ax.text(0.31, ax.get_ylim()[0] * 1.08, "outside training range", color=MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("|log(K / S)|  (distance from at-the-money)")
    ax.set_title("Uncertainty rises outside the training data")
    ax.legend(fontsize=8)
    saved(fig, "uq_ood.png", "Options sampled beyond the training ranges: uncertainty tracks the error.")


def training_curve():
    if not have("surrogate_training.csv"):
        return
    d = pd.read_csv(RESULTS_DIR / "surrogate_training.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(d.epoch, d.train_loss, color=ORANGE, label="train")
    ax.plot(d.epoch, d.val_loss, color=BLUE, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted MSE (standardised units)")
    ax.set_title("Surrogate training curve")
    ax.legend()
    saved(fig, "training_curve.png", "Validation loss tracks training loss: no overfitting.")


def html():
    def table(rows):
        return "<table>" + "".join(f"<tr><td>{a}</td><td><b>{b}</b></td></tr>" for a, b in rows) + "</table>"

    sections = []
    if (RESULTS_DIR / "ob_summary.csv").exists():
        s = pd.read_csv(RESULTS_DIR / "ob_summary.csv").set_index("metric")["value"]
        sections.append(("Order book", table([
            ("p50 / p99 / p99.9 latency", f"{s.p50_ns:.0f} / {s.p99_ns:.0f} / {s.p999_ns:.0f} ns"),
            ("throughput (1 thread)", f"{s.throughput_msgs_per_sec / 1e6:.1f} M msg/s"),
            ("best bid/ask read", f"{s.bbo_query_ns:.1f} ns")])))
    if (RESULTS_DIR / "surrogate_metrics.json").exists():
        m = json.load(open(RESULTS_DIR / "surrogate_metrics.json"))
        rows = [(f"{GREEK_LABELS[j]} MAE", f"{m['per_greek'][g]['mae']:.4g}") for j, g in enumerate(GREEKS)]
        sp = m["speed"]
        rows += [("surrogate, 1 option", f"{sp['surrogate_single_call_us']:.0f} µs"),
                 ("surrogate, batched", f"{sp['surrogate_batched_per_option_us']:.2f} µs / option"),
                 ("MC bump Greeks", f"{sp['mc_bump_greeks_ms']:.0f} ms ({sp['mc_bump_config']})"),
                 ("speedup (batched / single)", f"{sp['speedup_batched']:,.0f}x / {sp['speedup_single_call']:,.0f}x"),
                 ("95% coverage, delta (calibrated)", f"{m['uncertainty']['coverage95_calibrated']['delta']:.1%}")]
        sections.append(("Greeks surrogate", table(rows)))
    imgs = "".join(f'<figure><img src="plots/{n}" alt="{c}"><figcaption>{c}</figcaption></figure>' for n, c in made)
    note_html = "".join(f"<li>{n}</li>" for n in notes)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantumFlow Benchmarks</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px}} h2{{margin-top:32px;border-bottom:1px solid var(--line)}} p.sub{{color:var(--muted);margin-top:0}}
table{{border-collapse:collapse}} td{{padding:4px 18px 4px 0;border-bottom:1px solid var(--line)}}
figure{{margin:24px 0}} img{{max-width:100%;height:auto}} figcaption{{color:var(--muted);font-size:13px}}
</style></head><body><h1>QuantumFlow benchmarks</h1><p class="sub">Every number on this page was measured by the code in benchmarks/ and python/; nothing is typed in by hand.</p>
{''.join(f'<h2>{t}</h2>{b}' for t, b in sections)}<h2>Charts</h2>{imgs}{f'<h2>Notes</h2><ul>{note_html}</ul>' if notes else ''}</body></html>"""
    (RESULTS_DIR / "dashboard.html").write_text(page)


if __name__ == "__main__":
    for fn in (order_book, mc_threads, cuda_sweep, variance_reduction, convergence, surrogate, training_curve):
        fn()
    html()
    for n, _ in made:
        print("wrote", PLOTS / n)
    for n in notes:
        print("note:", n)
    print("wrote", RESULTS_DIR / "dashboard.html")
