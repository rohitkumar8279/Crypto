"""Generate all publication figures from experiment outputs (CSV/JSON only; no hard-coded data).

Figures whose inputs are missing are skipped and reported. Writes results/figures/*.png and
results/figures/figure_sources.json (figure -> source files).

Usage: python experiments/make_figures.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from common import ROOT, dump, out_dir  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = ROOT / "results"
OUT = out_dir("figures")
SOURCES: dict[str, list[str]] = {}
SKIPPED: dict[str, str] = {}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "figure.dpi": 100})
C_BIAS, C_DEG1, C_SPEC = "#7a7a7a", "#e69f00", "#0072b2"


def need(name, *paths):
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        SKIPPED[name] = "missing: " + ", ".join(missing)
        return False
    SOURCES[name] = [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]
    return True


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def fig1_detection_depth():
    name = "fig1_detection_depth"
    p = R / "novel/e2_detection_depth/per_round.csv"
    if not need(name, p):
        return
    df = pd.read_csv(p)
    targets = [("simon3264", "0x400"), ("speck3264", "0x400000")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    thr = math.log10(0.01 / 13)
    for a, (c, d) in zip(ax, targets):
        g = df[(df.cipher == c) & (df.difference == d)]
        for col, lab, colr in [("logp_autond_bias_null", "AutoND bias score (simulated null)", C_BIAS),
                               ("logp_chi2_deg1", "degree-1 χ² (calibrated per-bit)", C_DEG1),
                               ("logp_chi2_deg12", "degree-≤2 spectral χ² (ours)", C_SPEC)]:
            m = g.groupby("round")[col].agg(["min", "max", "median"]) / math.log(10)
            m = m.clip(lower=-400)
            a.plot(m.index, -m["median"], "o-", color=colr, label=lab, ms=4)
            a.fill_between(m.index, -m["min"], -m["max"], color=colr, alpha=0.2)
        a.axhline(-thr, color="k", ls=":", lw=1, label="significance (α=0.01, Bonferroni)")
        a.set_yscale("symlog", linthresh=1)
        a.set_ylim(0, 600)
        a.set_xlabel("rounds"); a.set_ylabel("−log10 p  (clipped at 400)")
        a.set_title(f"{c.upper()} δ={d}, t=10⁵ per round")
    ax[1].legend(fontsize=7, loc="lower left")
    save(fig, name)


def fig2_cost_and_depth():
    name = "fig2_search_cost_depth"
    p = R / "novel/e3_search_with_ablation/runs.csv"
    if not p.exists():
        p = R / "novel/e3_search/runs.csv"
    if not need(name, p):
        return
    df = pd.read_csv(p)
    methods = [m for m in ["autond_bias", "spectral_deg1_ablation", "spectral"] if m in set(df.method)]
    cols = {"autond_bias": C_BIAS, "spectral_deg1_ablation": C_DEG1, "spectral": C_SPEC}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ciphers = ["speck3264", "simon3264"]
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        g = df[df.method == m].groupby("cipher")
        for j, (metric, a, lab) in enumerate([("highest_round", ax[0], "highest round reached"),
                                              ("search_encryptions", ax[1], "search encryptions"),
                                              ("seconds", ax[2], "wall time (s)")]):
            mean = g[metric].mean().reindex(ciphers)
            lo = g[metric].min().reindex(ciphers); hi = g[metric].max().reindex(ciphers)
            x = np.arange(len(ciphers)) + i * w
            a.bar(x, mean, w, color=cols[m], label=m if j == 0 else None,
                  yerr=[mean - lo, hi - mean], capsize=3)
            a.set_xticks(np.arange(len(ciphers)) + w * (len(methods) - 1) / 2, [c.upper() for c in ciphers])
            a.set_ylabel(lab)
    ax[0].legend(fontsize=7)
    ax[0].set_title("Search depth (3 seeds, t=10³)"); ax[1].set_title("Encryption cost"); ax[2].set_title("Runtime")
    save(fig, name)


def fig3_capacity_vs_rounds():
    name = "fig3_capacity_vs_rounds"
    p = R / "novel/e2_detection_depth/per_round.csv"
    if not need(name, p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, c in zip(ax, ["simon3264", "speck3264"]):
        for d, g in df[df.cipher == c].groupby("difference"):
            m = g.groupby("round")[["cap1", "cap2"]].mean()
            a.plot(m.index, m["cap1"].clip(lower=1e-6), "s--", label=f"{d}: degree-1 (per-bit) part", ms=3)
            a.plot(m.index, m["cap2"].clip(lower=1e-6), "o-", label=f"{d}: 2-bit mask part", ms=3)
        a.set_yscale("log"); a.set_xlabel("rounds"); a.set_ylabel("unbiased capacity (clipped at 1e-6)")
        a.set_title(c.upper() + "  (points at 1e-6: unbiased estimate ≤ 0)", fontsize=9); a.legend(fontsize=7)
    save(fig, name)


def _e1_table():
    p = R / "novel/e1_speck3264/e1_table.csv"
    return p, (pd.read_csv(p) if p.exists() else None)


def fig4_dbitnet_vs_scores():
    name = "fig4_dbitnet_vs_scores"
    p, df = _e1_table()
    a_json = R / "novel/e1_speck3264/e1_analysis.json"
    if not need(name, p, a_json):
        return
    an = json.loads(a_json.read_text())
    rounds = sorted(int(c.split("_r")[1]) for c in df.columns if c.startswith("test_acc_r"))
    fig, ax = plt.subplots(len(rounds), 2, figsize=(10, 3.6 * len(rounds)), squeeze=False)
    for i, r in enumerate(rounds):
        y = df[f"test_acc_r{r}"]
        blk = an[f"round_{r}"]
        for j, (col, lab, colr) in enumerate([("autond_weighted", "AutoND weighted bias score", C_BIAS),
                                               (f"cap12_r{r}", f"spectral C≤2 at round {r}", C_SPEC)]):
            a = ax[i, j]
            a.scatter(df[col], y, color=colr, s=22)
            if j == 1:
                a.scatter(df[col], df[f"acc_lower_bound_certified_r{r}"], marker="_", color="r", s=60,
                          label="certified floor (Prop. 3/6)")
                a.set_xscale("symlog", linthresh=1e-3); a.legend(fontsize=7)
            s = blk[col]
            a.set_title(f"Spearman ρ={s['spearman']:.2f}  95% CI [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]", fontsize=9)
            a.set_xlabel(lab); a.set_ylabel(f"DBitNet test accuracy, round {r}")
            a.axhline(0.5, color="k", lw=0.5, ls=":")
    save(fig, name)


def fig5_spectral_vs_bias():
    name = "fig5_spectral_vs_bias"
    p = R / "novel/e1_speck3264/weighted_scores.csv"
    if not need(name, p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(df["wb"], df["cap12_next_round"].clip(lower=1e-4), s=14, color=C_SPEC)
    for _, r in df.nsmallest(3, "rank_bias").iterrows():
        ax.annotate(r["difference"], (r["wb"], max(r["cap12_next_round"], 1e-4)), fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("AutoND weighted bias score (rounds 1..R)")
    ax.set_ylabel("spectral C≤2 at round R+1 (clipped 1e-4)")
    from cgsearch.metrics import spearman
    ax.set_title(f"SPECK32 pool (n={len(df)}): Spearman {spearman(df['wb'], df['cap12_next_round']):.2f}", fontsize=9)
    save(fig, name)


def fig6_correlation_cis():
    name = "fig6_correlation_cis"
    a_json = R / "novel/e1_speck3264/e1_analysis.json"
    if not need(name, a_json):
        return
    an = json.loads(a_json.read_text())
    rounds = sorted(int(k.split("_")[1]) for k in an if k.startswith("round_"))
    proxies = [("autond_weighted", "AutoND weighted bias", C_BIAS), ("autond_bias_r{r}", "AutoND bias @ round r", C_DEG1),
               ("cap12_r{r}", "spectral C≤2 @ round r", C_SPEC), ("max_abs_corr_r{r}", "max|c| @ round r", "#009e73")]
    fig, ax = plt.subplots(figsize=(7, 1.0 + 1.1 * len(rounds) * len(proxies) / 2))
    ypos, labels, y = [], [], 0.0
    for r in rounds:
        blk = an[f"round_{r}"]
        for key, lab, colr in proxies:
            k = key.format(r=r)
            s_ = blk[k]
            lo, hi = s_["ci95"]
            ax.errorbar(s_["spearman"], y, xerr=[[s_["spearman"] - lo], [hi - s_["spearman"]]], fmt="o",
                        color=colr, capsize=3)
            ypos.append(y); labels.append(f"test round {r}: {lab.replace('round r', f'round {r}')}")
            y += 1
        y += 0.6
    ax.set_yticks(ypos, labels, fontsize=7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Spearman ρ with DBitNet test accuracy (95% bootstrap CI, n candidates)")
    ax.set_title(f"E1, SPECK32, n={an['n_candidates']} candidates", fontsize=9)
    ax.invert_yaxis()
    save(fig, name)


def fig7_allocation():
    name = "fig7_adaptive_vs_fixed"
    p_main = R / "phase5/simon32_main/summary.csv"
    p_e5 = R / "novel/e5_compare/summary.csv"
    have_main, have_e5 = p_main.exists(), p_e5.exists()
    if not (have_main or have_e5):
        SKIPPED[name] = "missing phase5 main and e5 summaries"
        return
    SOURCES[name] = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (p_main, p_e5) if p.exists()]
    n = int(have_main) + int(have_e5)
    fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 3.8), squeeze=False)
    i = 0
    cols = {"A_fixed": C_BIAS, "B_adaptive": C_SPEC, "C_uniform_matched": C_DEG1, "C2_successive_halving": "#009e73"}
    if have_main:
        d = pd.read_csv(p_main)
        a = ax[0, i]; i += 1
        for pol, g in d.groupby("policy"):
            g = g.sort_values("encryptions_mean")
            a.plot(g["encryptions_mean"], g["topk_overlap_ref"], "o-", color=cols.get(pol), label=pol)
        a.set_xscale("log"); a.set_ylim(0, 1.05); a.set_xlabel("scoring encryptions")
        a.set_ylabel("top-16 overlap with reference"); a.set_title("SIMON32 main (pre-registered, 5 seeds)")
        a.legend(fontsize=7)
    if have_e5:
        d = pd.read_csv(p_e5)
        a = ax[0, i]
        for stat, mk in [("bias", "s"), ("spectral", "o")]:
            for pol, g in d[d.statistic == stat].groupby("policy"):
                a.errorbar(g["encryptions_mean"], g["regret_mean"], yerr=g["regret_sd"], fmt=mk, color=cols.get(pol),
                           capsize=3, label=f"{stat}: {pol}")
        a.set_xscale("log"); a.set_xlabel("scoring encryptions"); a.set_ylabel("regret vs same-statistic reference")
        a.set_title("SPECK32 pool, K=8 (E5, 3 seeds)"); a.legend(fontsize=6)
    save(fig, name)


def fig8_runtime():
    name = "fig8_runtime"
    srcs = [R / "novel/e5_compare/runs.csv", R / "phase5/simon32_main/runs.csv"]
    srcs = [p for p in srcs if p.exists()]
    if not srcs:
        SKIPPED[name] = "missing e5 and phase5 runs"
        return
    SOURCES[name] = [str(p.relative_to(ROOT)).replace("\\", "/") for p in srcs]
    fig, ax = plt.subplots(1, len(srcs), figsize=(5.5 * len(srcs), 3.6), squeeze=False)
    for a, p in zip(ax[0], srcs):
        d = pd.read_csv(p)
        if "budget_fraction" in d:
            d = d[d.budget_fraction == 1.0]
        key = ["statistic", "policy"] if "statistic" in d else ["policy"]
        g = d.groupby(key)["seconds"].agg(["mean", "min", "max"]).reset_index()
        lab = g[key].astype(str).agg(": ".join, axis=1)
        a.barh(lab, g["mean"], xerr=[g["mean"] - g["min"], g["max"] - g["mean"]], color=C_SPEC, capsize=3)
        a.set_xlabel("wall time per scoring run (s)"); a.set_title(p.parent.name, fontsize=9)
        a.tick_params(axis="y", labelsize=7)
    save(fig, name)


def main():
    for f in (fig1_detection_depth, fig2_cost_and_depth, fig3_capacity_vs_rounds, fig4_dbitnet_vs_scores,
              fig5_spectral_vs_bias, fig6_correlation_cis, fig7_allocation, fig8_runtime):
        try:
            f()
        except Exception as e:  # report, never invent data
            SKIPPED[f.__name__] = f"error: {e!r}"
    dump({"sources": SOURCES, "skipped": SKIPPED}, OUT / "figure_sources.json")
    print("skipped:", json.dumps(SKIPPED, indent=1))


if __name__ == "__main__":
    main()
