"""
Experiments for "How does feature learning change the function space evolution?" (single-index model).

    python experiments.py alignment        # alignment experiment: K0, the network after one step, K_D, K1
    python experiments.py e2e              # end-to-end experiment, original scenario
    python experiments.py stress_align     # alignment stress test     (all scenarios, or --scenarios key ...)
                                           #   add --dims 50 100 200 400 800 1600 3200 for a d-sweep per scenario
    python experiments.py stress_e2e       # end-to-end stress test    (all scenarios, or --scenarios key ...)
    python experiments.py plots            # every figure (curves only) and table from the saved results
    python experiments.py plots --layout group   # one figure per type of variation (links, noise, inputs, d)
    python experiments.py plots --per-scenario   # one figure per stress scenario
    python experiments.py all              # everything above, in this order

Results are saved in results/*.pkl after every scenario, so runs can be interrupted and resumed per scenario.
"""
import os
import sys
import shutil
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator, FormatStrFormatter, NullFormatter
from kernels import *
from two_layer_nn import OneStepNet
from data import Target, InputLaw, sample_data, SCENARIOS, SC, VARIATIONS

def _latex_works():
    """Use LaTeX for text only if it is installed AND can render (matplotlib also needs e.g. cm-super)."""
    if shutil.which("latex") is None:
        return False
    try:
        with plt.rc_context({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath,amssymb}"}):
            fig = plt.figure()
            fig.text(0.5, 0.5, r"$\mathbb{S}^{d-1}$ test")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


USETEX = _latex_works()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (3.5, 2.5),
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "text.usetex": USETEX,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
})
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
# alignment experiment (App. D.3): N = d points; the network's step sizes give B = 5 d^{p/10} in Eq. (3.3)
ALIGN_SEED = 54643
D_LIST = [50, 100, 200, 400, 800, 1600, 3200]
B_VAL = 5
LR_EXPS = [3/10, 5/10, 7/10, 9/10]
LR_EXPS_STR = ["3/10", "5/10", "7/10", "9/10"]
M_WIDTH = 400          # network width of the paper
ALPHA = 2              # first-stage samples n = ALPHA * d

# end-to-end experiment: one GD step (Eq. 2.3) on n = 2d samples, ridge on the second layer (Eq. 2.4)
D_E2E = 300
N_RATIOS = [1/4, 1/2, 1, 2, 4, 8]      # second-stage N = ratio * d  (75 ... 2400 at d = 300)
ZETAS = [5/8, 3/4, 7/8]                # eta = ETA_TILDE * d^zeta
ZETAS_STR = ["5/8", "3/4", "7/8"]
ETA_TILDE = 3.0
M_TEST = 2000
# Network setting of the end-to-end experiment (and its stress test). Default: n = 2d, m = 4d/3 (400 at d = 300).
# Larger values bring the network closer to the theory (K_D -> K1 as n/d grows; infinite width) at almost no
# extra cost: e.g. --alpha-e2e 8 --m-per-d 5.333 gives n = 2400, m = 1600 at d = 300.
# The step size is scaled with the width, eta = 3 d^zeta * m / (4d/3), so that B (Eq. 3.3) does not change.
ALPHA_E2E = 2.0        # first-stage samples n = ALPHA_E2E * d
M_PER_D = 4 / 3        # width m = M_PER_D * d

NUM_TRIALS = {"alignment": 10, "e2e": 10, "stress_align": 10, "stress_e2e": 5}


def save(obj, name):
    with open(f"results/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load(name):
    path = f"results/{name}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def mean_std(records, value, **conds):
    """Mean and standard deviation (over trials) of `value` among the records matching `conds`."""
    v = np.array([r[value] for r in records if all(r.get(k) == c for k, c in conds.items())])
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


# =========================================================
# 1. Alignment experiment: K0, the network after one step, K_D and K1 (A, B of the step, Eq. 3.3)
# =========================================================
def run_experiment_dimension_alignment(num_trials=NUM_TRIALS["alignment"]):
    """
    Squared alignment <v_top, Y2>^2 on N = d Gaussian points (same seed and draws of Z and w* as the original
    code), for K0 and for the network after one GD step (width M_WIDTH, n = 2d samples of the original-scenario
    target). For each level B = 5 d^{p/10}, the step size is solved from Eq. (3.3) so that the idealized kernel of
    the network has this B; we also report K1 with the A and B of that step, and K_D of the same first-stage sample.
    The network uses its own random stream, so K0 is computed on exactly the draws of the original code.
    """
    print("Running Experiment: Alignment vs. Dimension...")
    np.random.seed(ALIGN_SEED)
    T_base = Target("main", 0.3)
    records = []
    for d in D_LIST:
        for trial in range(num_trials):
            print(f"d={d}, trial: {trial+1}")
            N = d
            X = np.random.randn(N, d)
            w_star = np.random.randn(d)
            w_star /= np.linalg.norm(w_star)
            y_2 = y2_feature(X, w_star, center="paper")

            K0 = compute_analytical_kernel(X, X, w_star, A=1, B=0)
            records.append(dict(d=d, trial=trial, model="K0", p=None, align=alignment(K0, y_2)))

            state = np.random.get_state()
            net_seed = 10_000 * (trial + 1) + d
            np.random.seed(net_seed)
            n, m = ALPHA * d, M_WIDTH
            X1, y1, _ = sample_data(InputLaw("gauss", d, w_star), T_base, n)
            h = X1.T @ y1 / n
            h_hat = h / np.linalg.norm(h)
            net = OneStepNet(d, m, seed=net_seed)
            grad = net.first_layer_gradient(X1, y1)
            for e in LR_EXPS:
                B = B_VAL * d ** e
                eta = eta_for_B(d, n, m, B, T_base.c1, T_base.s)
                A_step, _ = idealized_constants(d, n, m, eta, T_base.c1, T_base.s, T_base.Ey2)
                bD = single_dataset_b(d, m, eta, h)
                F = net.features_after_step(X, grad, eta)
                info = dict(d=d, trial=trial, p=e, A=A_step, B=B, bD=bD, eta=eta, cos2=float((h_hat @ w_star) ** 2))
                records.append(dict(info, model="net", align=alignment(F @ F.T, y_2)))
                records.append(dict(info, model="KD", align=alignment(
                    compute_analytical_kernel(X, X, h_hat, A=1.0, B=bD), y_2)))
                records.append(dict(info, model="K1", align=alignment(
                    compute_analytical_kernel(X, X, w_star, A=A_step, B=B), y_2)))
            np.random.set_state(state)
        save(records, "alignment_original")
        print(f"Done d={d}")
    return records


# =========================================================
# 2. End-to-end experiment (exact two-stage protocol of Sec. 2)
# =========================================================
def end_to_end_trial(sc, trial):
    """
    One trial of the two-stage protocol for scenario `sc`:
      stage 1  one full-batch GD step on W (Eq. 2.3) on n = ALPHA_E2E * d samples (default 2d), width
               m = M_PER_D * d (default 4d/3), eta = 3 d^zeta * m / (4d/3)
      stage 2  ridge on the second layer (Eq. 2.4) on N fresh samples, lambda by 5-fold CV
      test     excess risk ||f_hat - f*||^2 / E[f*^2] on M_TEST fresh samples
    Predictors, all fitted on the same stage-2 sample:
      net      features of the trained first layer W_1          rf0     features of W_0 (same width)
      K0       infinite-width kernel at initialization          linear  ridge on raw inputs
      KD       single-dataset kernel k_{I + b_D hhat hhat^T}    K1      idealized kernel k_{A I + B w* w*^T}, Eq. (3.3)
    """
    d = sc["d"]
    n, m = int(round(ALPHA_E2E * d)), int(round(M_PER_D * d))
    N_list = [max(5, int(round(r * d))) for r in N_RATIOS]
    N_max = max(N_list)

    seed = 40_000 + trial
    np.random.seed(seed)
    T = Target(sc["link"], sc["sigma"], sc["noise"])
    w_star = np.random.randn(d)
    w_star /= np.linalg.norm(w_star)
    law = InputLaw(sc["law"], d, w_star)
    X1, y1, _ = sample_data(law, T, n)            # stage-1 data
    X2, y2, _ = sample_data(law, T, N_max)        # stage-2 data (independent)
    Xt, _, ft = sample_data(law, T, M_TEST)       # test data
    Ef2 = law.Ef2(T)
    h = X1.T @ y1 / n
    h_hat = h / np.linalg.norm(h)

    net = OneStepNet(d, m, seed=seed)
    grad = net.first_layer_gradient(X1, y1)

    meta = dict(scenario=sc["key"], trial=trial, d=d, m=m, n=n, Ef2=Ef2, c1=T.c1,
                cos2=float((h_hat @ w_star) ** 2))

    def feature_kernels(eta):
        F2, Ft = net.features_after_step(X2, grad, eta), net.features_after_step(Xt, grad, eta)
        return F2 @ F2.T, Ft @ F2.T

    def gauss_kernels(v, A, B):
        return compute_analytical_kernel(X2, X2, v, A, B), compute_analytical_kernel(Xt, X2, v, A, B)

    builders = [("rf0", lambda: feature_kernels(0.0)),
                ("K0", lambda: gauss_kernels(w_star, 1.0, 0.0)),
                ("linear", lambda: (X2 @ X2.T / d, Xt @ X2.T / d))]
    for zeta in ZETAS:
        eta = ETA_TILDE * d ** zeta * M_PER_D / (4 / 3)      # scaled with the width: same B as at m = 4d/3
        A, B = idealized_constants(d, n, m, eta, T.c1, T.s, T.Ey2)
        bD = single_dataset_b(d, m, eta, h)
        meta.update({f"A_{zeta}": A, f"B_{zeta}": B, f"bD_{zeta}": bD})
        builders += [(f"net_{zeta}", lambda eta=eta: feature_kernels(eta)),
                     (f"KD_{zeta}", lambda bD=bD: gauss_kernels(h_hat, 1.0, bD)),
                     (f"K1_{zeta}", lambda A=A, B=B: gauss_kernels(w_star, A, B))]

    records = []
    for name, build in builders:              # one model at a time keeps memory low
        K_train, K_test = build()
        for N, r in zip(N_list, N_RATIOS):
            risk, lam = krr_excess_risk(K_train[:N, :N], K_test[:, :N], y2[:N], ft, Ef2)
            records.append(dict(meta, model=name, N=N, N_over_d=r, risk=risk, lam=lam))
    return records


def run_experiment_end_to_end(num_trials=NUM_TRIALS["e2e"]):
    print("Running End-to-End Experiment (original scenario)...")
    records = []
    for trial in range(num_trials):
        print(f"trial: {trial+1}")
        records += end_to_end_trial(SC["base"], trial)
        save(records, "end_to_end_original")
    return records


# =========================================================
# 3. Stress tests
# =========================================================
def alignment_trial(sc, trial, d=None):
    """
    Alignment at dimension d (default: the scenario's d = 300; N = d points, squared alignment) for scenario
    `sc`: K0, the network after one step, K_D, and K1 (A, B from Eq. 3.3 for this scenario's link and noise).
    The step size for each level p is the one that gives B = 5 d^{p/10} in the ORIGINAL scenario; it is not
    retuned, so a different link changes B through c1^2 exactly as Eq. (3.3) says.
    Y2 is centred by its sample mean (identical to -1/d up to O(N^{-1/2}) for rotation-invariant inputs, and
    needed for anisotropic inputs, where E<omega, w*>^2 != 1/d).
    """
    d = d or sc["d"]
    N, n, m = d, ALPHA * d, M_WIDTH
    seed = 50_000 + 10_000 * trial + d
    np.random.seed(seed)
    T, T_base = Target(sc["link"], sc["sigma"], sc["noise"]), Target("main", 0.3)
    w_star = np.random.randn(d)
    w_star /= np.linalg.norm(w_star)
    law = InputLaw(sc["law"], d, w_star)
    X = law.sample(N)
    y_2 = y2_feature(X, w_star, center="empirical")
    X1, y1, _ = sample_data(law, T, n)
    h = X1.T @ y1 / n
    h_hat = h / np.linalg.norm(h)
    net = OneStepNet(d, m, seed=seed)
    grad = net.first_layer_gradient(X1, y1)

    meta = dict(scenario=sc["key"], trial=trial, d=d, c1=T.c1, cos2=float((h_hat @ w_star) ** 2))
    records = [dict(meta, model="K0", p=None,
                    align=alignment(compute_analytical_kernel(X, X, w_star, A=1, B=0), y_2))]
    for e in LR_EXPS:
        B_p = B_VAL * d ** e
        eta = eta_for_B(d, n, m, B_p, T_base.c1, T_base.s)
        A, B = idealized_constants(d, n, m, eta, T.c1, T.s, T.Ey2)
        bD = single_dataset_b(d, m, eta, h)
        F = net.features_after_step(X, grad, eta)
        info = dict(meta, p=e, A=A, B=B, bD=bD, eta=eta)
        records += [
            dict(info, model="net", align=alignment(F @ F.T, y_2)),
            dict(info, model="KD", align=alignment(compute_analytical_kernel(X, X, h_hat, A=1.0, B=bD), y_2)),
            dict(info, model="K1", align=alignment(compute_analytical_kernel(X, X, w_star, A=A, B=B), y_2)),
        ]
    return records


def run_stress_alignment(keys=None, num_trials=NUM_TRIALS["stress_align"], dims=None):
    """dims=None: every scenario at its own d (300).
    dims=[50, ..., 3200]: a dimension sweep per scenario, so each scenario gets a figure like the alignment figure."""
    keys = keys or [s["key"] for s in SCENARIOS if s["group"] != "dimension"]
    for key in keys:
        print(f"Stress test (alignment): {key}")
        records = []
        for d in (dims or [SC[key]["d"]]):
            for trial in range(num_trials):
                print(f"  d={d}, trial: {trial+1}")
                records += alignment_trial(SC[key], trial, d)
            save(records, f"stress_align_{key}")


def run_stress_end_to_end(keys=None, num_trials=NUM_TRIALS["stress_e2e"]):
    keys = keys or [s["key"] for s in SCENARIOS]
    for key in keys:
        print(f"Stress test (end-to-end): {key}")
        if key == "base":                     # same seeds as the original run: reuse its first trials
            original = load("end_to_end_original")
            same_setting = original is not None and original[0]["n"] == int(round(ALPHA_E2E * 300)) \
                and original[0]["m"] == int(round(M_PER_D * 300))
            if same_setting and len({r["trial"] for r in original}) >= num_trials:
                save([r for r in original if r["trial"] < num_trials], "stress_e2e_base")
                continue
        records = []
        for trial in range(num_trials):
            print(f"  trial: {trial+1}")
            records += end_to_end_trial(SC[key], trial)
            save(records, f"stress_e2e_{key}")


# =========================================================
# 4. Plots and tables
# =========================================================
COL = {"net": "#2CA02C", "K0": "#0072B2", "K1": "#D55E00", "KD": "#CC79A7", "linear": "#E69F00",
       "rf0": "#7f7f7f"}
STY = {"net": ("-.", "D", False), "K0": ("--", "o", False), "K1": ("-", "s", False), "KD": ("-", "^", False),
       "linear": (":", "+", False), "rf0": ("--", "v", True)}
LAB = {"net": "Neural Net (one step)", "K0": "$K_0$", "K1": "$K_1$ ($A,B$ from Eq. 3.3)",
       "KD": "$K_D$ (single dataset)", "linear": "linear ridge", "rf0": "random features at init"}


def _log_y(ax):
    """Log-scale risk axis with plain tick labels (0.05, 0.1, 0.2, 0.5, 1)."""
    ax.set_yscale('log', base=10)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def band(ax, x, mean, std, key, label=True, lw=1.5, mkrsize=3):
    ls, mk, open_ = STY[key]
    ax.plot(x, mean, ls=ls, marker=mk, markersize=mkrsize, color=COL[key], lw=lw,
            mfc="white" if open_ else COL[key], label=LAB[key] if label else None)
    ax.fill_between(x, np.maximum(mean - std, 1e-12), mean + std, color=COL[key], alpha=0.15, edgecolor="none")


def _alignment_curves(ax, records, p=None, lw=1.5, label=True):
    """Alignment curves of one experiment/scenario: against d at level p when several dimensions were run,
    otherwise against the B level (step sizes giving B = 5 d^{p/10} in the original scenario)."""
    dims = sorted({r["d"] for r in records})
    models = ["K1", "KD", "net"]
    if len(dims) > 1:
        m0 = np.array([mean_std(records, "align", model="K0", d=d) for d in dims])
        band(ax, dims, m0[:, 0], m0[:, 1], "K0", label=label, lw=lw)
        for key in models:
            mm = np.array([mean_std(records, "align", model=key, p=p, d=d) for d in dims])
            band(ax, dims, mm[:, 0], mm[:, 1], key, label=label, lw=lw)
        ax.set_xscale('log', base=2)
        ax.set_xticks([d for d in dims if d in (50, 200, 800, 3200)] if len(dims) > 4 else dims)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Dimension ($d$)")
    else:
        x = np.arange(len(LR_EXPS))
        m0 = mean_std(records, "align", model="K0")
        band(ax, x, np.full(len(x), m0[0]), np.full(len(x), m0[1]), "K0", label=label, lw=lw)
        for key in models:
            mm = np.array([mean_std(records, "align", model=key, p=e) for e in LR_EXPS])
            band(ax, x, mm[:, 0], mm[:, 1], key, label=label, lw=lw)
        ax.set_xticks(x)
        ax.set_xticklabels([fr"$5d^{{{s}}}$" for s in LR_EXPS_STR])
        ax.set_xlabel(fr"$B$ in the original scenario ($d={dims[0]}$)")
    ax.grid(True, which='major', linestyle='--', alpha=0.6)


def plot_alignment_network(records, name="alignment_network", title=None):
    """One panel per B level, against d: K0, K1 with the step's A and B, K_D and the network."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for ax, e, e_str in zip(axes, LR_EXPS, LR_EXPS_STR):
        _alignment_curves(ax, records, p=e, label=(e == LR_EXPS[0]))
        ax.set_title(fr"$B = 5d^{{{e_str}}}$")
    axes[0].set_ylabel(r"Alignment with feature $Y_2$")
    if title:
        fig.suptitle(title, y=1.04)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1),
               frameon=False)
    plt.tight_layout()
    plt.savefig(f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_end_to_end(records, name="end_to_end", c1=None, Ef2=None, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    Ns = sorted({r["N"] for r in records})
    for ax, zeta, z_str in zip(axes, ZETAS, ZETAS_STR):
        for key in ["rf0", "K0", "linear", "net", "KD", "K1"]:
            model = key if key in ("rf0", "K0", "linear") else f"{key}_{zeta}"
            mm = np.array([mean_std(records, "risk", model=model, N=N) for N in Ns])
            band(ax, Ns, mm[:, 0], mm[:, 1], key, label=(zeta == ZETAS[0]))
        if c1 is not None:
            ax.axhline(1 - c1 ** 2 / Ef2, color="black", lw=0.8, ls=":")
        B = np.mean([r[f"B_{zeta}"] for r in records])
        ax.set_title(fr"$\zeta = {z_str}$")
        ax.set_xscale('log', base=2)
        _log_y(ax)
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("Number of Training Samples ($N$)")
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
    axes[0].set_ylabel(r"Excess risk $\|\hat f - f^*\|^2 / \|f^*\|^2$")
    if title:
        fig.suptitle(title, y=1.04)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1),
               frameon=False)
    plt.tight_layout()
    plt.savefig(f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_stress(prefix):
    records = []
    for s in SCENARIOS:
        r = load(f"{prefix}_{s['key']}")
        if r is not None:
            records += r
    return records


def _present(records):
    """Scenario keys present in `records`, in the order of data.SCENARIOS."""
    keys = {r["scenario"] for r in records}
    return [s["key"] for s in SCENARIOS if s["key"] in keys]


def _grid(n_panels, ncol, width, height):
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height * nrow), squeeze=False)
    for ax in list(axes.flat)[n_panels:]:
        ax.axis("off")
    return fig, axes


def _best_linear(sc):
    """(c1, E[g^2]) for the best-linear-predictor line 1 - c1^2/E[g^2]; only drawn when <w*, x> is Gaussian."""
    if sc["law"] not in ("gauss", "aniso"):
        return None, None
    T = Target(sc["link"], sc["sigma"], sc["noise"])
    return T.c1, T.Ef2


def plot_stress_end_to_end(records, per_scenario=False, zeta=ZETAS[-1]):
    """
    per_scenario=True : figures/stress_e2e/<scenario>.pdf, each laid out like the end-to-end figure
                        (risk against N, one panel per step size).
    per_scenario=False: figures/stress_end_to_end.pdf, all scenarios in one image, one panel per scenario
                        (risk against N/d, step size eta = 3 d^zeta).
    """
    keys = _present(records)
    if per_scenario:
        os.makedirs("figures/stress_e2e", exist_ok=True)
        for key in keys:
            c1, Ef2 = _best_linear(SC[key])
            plot_end_to_end([r for r in records if r["scenario"] == key], name=f"stress_e2e/{key}",
                            c1=c1, Ef2=Ef2, title=SC[key]["label"])
        return
    fig, axes = _grid(len(keys), ncol=6, width=12, height=2.1)
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        for model in ["rf0", "K0", "linear", "net", "KD", "K1"]:
            name = model if model in ("rf0", "K0", "linear") else f"{model}_{zeta}"
            mm = np.array([mean_std(records, "risk", model=name, scenario=key, N_over_d=r) for r in N_RATIOS])
            band(ax, N_RATIOS, mm[:, 0], mm[:, 1], model, label=(i == 0), lw=1.0)
        ax.set_xscale('log', base=2)
        _log_y(ax)
        ax.set_xticks([1/4, 1, 4])
        ax.set_xticklabels(["1/4", "1", "4"])
        ax.set_title(SC[key]["label"], fontsize=8)
        ax.set_xlabel("$N/d$", fontsize=8)
        if i % axes.shape[1] == 0:
            ax.set_ylabel("Excess risk")
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    plt.savefig("figures/stress_end_to_end.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_stress_alignment(records, per_scenario=False, p=LR_EXPS[-1]):
    """
    per_scenario=True : figures/stress_align/<scenario>.pdf, alignment against B at d = 300; or, if the stress
                        test was run with --dims, the layout of the alignment figure (against d, one panel per B).
    per_scenario=False: figures/stress_alignment.pdf, all scenarios in one image, one panel per scenario
                        (against B; or against d at level p when several dimensions were run).
    """
    keys = _present(records)
    if per_scenario:
        os.makedirs("figures/stress_align", exist_ok=True)
        for key in keys:
            sub = [r for r in records if r["scenario"] == key]
            if len({r["d"] for r in sub}) > 1:
                plot_alignment_network(sub, name=f"stress_align/{key}", title=SC[key]["label"])
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.set_box_aspect(0.8)
            _alignment_curves(ax, sub)
            ax.set_ylabel(r"Alignment with feature $Y_2$")
            ax.set_title(SC[key]["label"])
            ax.legend(loc="upper left", fontsize="small", frameon=True, framealpha=0.8)
            plt.tight_layout()
            plt.savefig(f"figures/stress_align/{key}.pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        return
    fig, axes = _grid(len(keys), ncol=5, width=12, height=2.5)
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        _alignment_curves(ax, [r for r in records if r["scenario"] == key], p=p, lw=1.0, label=(i == 0))
        ax.set_title(SC[key]["label"], fontsize=8)
        ax.xaxis.label.set_size(8)
        if i % axes.shape[1] == 0:
            ax.set_ylabel("Alignment")
    plt.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    plt.savefig("figures/stress_alignment.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _variation_panels(records, variation):
    """Panels of one grouped figure: the original scenario first, then every scenario of this type of
    variation (ordered by d for the dimension figure), with their panel titles."""
    groups, base_title = VARIATIONS[variation]
    present = {r["scenario"] for r in records}
    keys = [s["key"] for s in SCENARIOS if s["group"] in groups and s["key"] in present]
    if not keys:
        return [], {}
    titles = {k: SC[k]["label"] + (f" ({SC[k]['group'].split()[0]})" if variation == "link" else "") for k in keys}
    if "base" in present:
        keys, titles["base"] = ["base"] + keys, base_title
    if variation == "dimension":
        keys = sorted(keys, key=lambda k: SC[k]["d"])
    return keys, titles


def _panel_grid(n):
    ncol = n if n <= 5 else int(np.ceil(n / 2))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.5 * nrow), sharey=True, squeeze=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig, axes


def plot_variation_end_to_end(records, variation, zeta=ZETAS[-1]):
    """figures/stress_e2e_<variation>.pdf: excess risk against N, one panel per scenario of this variation
    (plus the original scenario), step size eta = 3 d^zeta. Against N/d for the dimension figure."""
    keys, titles = _variation_panels(records, variation)
    if not keys:
        return
    one_d = len({SC[k]["d"] for k in keys}) == 1
    xkey = "N" if one_d else "N_over_d"
    fig, axes = _panel_grid(len(keys))
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        sub = [r for r in records if r["scenario"] == key]
        xs = sorted({r[xkey] for r in sub})
        for model in ["rf0", "K0", "linear", "net", "KD", "K1"]:
            name = model if model in ("rf0", "K0", "linear") else f"{model}_{zeta}"
            mm = np.array([mean_std(sub, "risk", model=name, **{xkey: x}) for x in xs])
            band(ax, xs, mm[:, 0], mm[:, 1], model, label=(i == 0), lw=1.2)
        c1, Ef2 = _best_linear(SC[key])
        if c1 is not None:
            ax.axhline(1 - c1 ** 2 / Ef2, color="black", lw=0.8, ls=":")
        ax.set_xscale('log', base=2)
        _log_y(ax)
        if one_d:
            ax.set_xticks(xs[::2])
            ax.set_xticklabels([str(x) for x in xs[::2]])
            ax.set_xlabel("Number of Training Samples ($N$)", fontsize=8)
        else:
            ax.set_xticks([1/4, 1, 4])
            ax.set_xticklabels(["1/4", "1", "4"])
            ax.set_xlabel("$N/d$", fontsize=8)
        ax.set_title(titles[key], fontsize=8)
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Excess risk")
    plt.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.06 / axes.shape[0]), frameon=False)
    plt.savefig(f"figures/stress_e2e_{variation}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_variation_alignment(records, variation, p=LR_EXPS[-1]):
    """figures/stress_align_<variation>.pdf: alignment curves, one panel per scenario of this variation
    (plus the original scenario); against B at d = 300, or against d at level p if run with --dims."""
    keys, titles = _variation_panels(records, variation)
    if not keys:
        return
    fig, axes = _panel_grid(len(keys))
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        _alignment_curves(ax, [r for r in records if r["scenario"] == key], p=p, lw=1.2, label=(i == 0))
        ax.set_title(titles[key], fontsize=8)
        ax.xaxis.label.set_size(8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Alignment with feature $Y_2$")
    plt.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06 / axes.shape[0]), frameon=False)
    plt.savefig(f"figures/stress_align_{variation}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _pm(m, s):
    return f"{m:.2f}\\,{{\\scriptsize$\\pm${s:.2f}}}"


def write_stress_tables(rec_e2e, rec_align):
    if rec_e2e:
        zeta, last = ZETAS[-1], max(N_RATIOS)
        final = [r for r in rec_e2e if r["N_over_d"] == last]
        lines, prev = [], None
        for s in SCENARIOS:
            sub = [r for r in final if r["scenario"] == s["key"]]
            if not sub:
                continue
            if prev is not None and s["group"] != prev:
                lines.append("\\midrule")
            prev = s["group"]
            cells = [s["label"], f"{sub[0]['c1']:.2f}", f"{np.mean([r['cos2'] for r in sub]):.2f}",
                     f"{np.mean([r[f'B_{zeta}'] for r in sub]):.0f}"]
            for model in ["rf0", "K0", "linear", f"net_{zeta}", f"KD_{zeta}", f"K1_{zeta}"]:
                cells.append(_pm(*mean_std(sub, "risk", model=model)))
            lines.append(" & ".join(cells) + " \\\\")
        with open("tables/stress_e2e.tex", "w") as f:
            f.write("\\begin{tabular}{lccc|ccc|ccc}\n\\toprule\n"
                    "scenario & $c_1$ & $\\cos^2(\\hat h_D,w^*)$ & $B$ & RF at $W_0$ & $K_0$ & linear & network & $K_D$ & $K_1$ \\\\\n"
                    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    if rec_align:
        lines, prev = [], None
        for s in SCENARIOS:
            sub = [r for r in rec_align if r["scenario"] == s["key"] and r["d"] == s["d"]]
            if not sub:
                continue
            if prev is not None and s["group"] != prev:
                lines.append("\\midrule")
            prev = s["group"]
            cells = [s["label"], f"{np.mean([r['cos2'] for r in sub]):.2f}", _pm(*mean_std(sub, "align", model="K0"))]
            for e in LR_EXPS[2:]:
                for model in ["net", "KD", "K1"]:
                    cells.append(_pm(*mean_std(sub, "align", model=model, p=e)))
            lines.append(" & ".join(cells) + " \\\\")
        with open("tables/stress_align.tex", "w") as f:
            f.write("\\begin{tabular}{lcc|ccc|ccc}\n\\toprule\n"
                    " & & & \\multicolumn{3}{c|}{$B=5d^{7/10}$ (original)} & \\multicolumn{3}{c}{$B=5d^{9/10}$ (original)} \\\\\n"
                    "scenario & $\\cos^2(\\hat h_D,w^*)$ & $K_0$ & network & $K_D$ & $K_1$ & network & $K_D$ & $K_1$ \\\\\n"
                    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def make_all_plots(layout="all", zeta=ZETAS[-1]):
    """layout='all': every stress scenario in one image; 'group': one image per type of variation
    (links, noise, inputs, dimension); 'scenario': one image per stress scenario."""
    rec = load("alignment_original")
    if rec:
        plot_alignment_network(rec)
    rec = load("end_to_end_original")
    if rec:
        T = Target("main", 0.3)
        plot_end_to_end(rec, c1=T.c1, Ef2=T.Ef2)
    rec_e2e, rec_align = load_stress("stress_e2e"), load_stress("stress_align")
    if layout == "group":
        for variation in VARIATIONS:
            if rec_e2e:
                plot_variation_end_to_end(rec_e2e, variation, zeta=zeta)
            if rec_align:
                plot_variation_alignment(rec_align, variation)
    else:
        if rec_e2e:
            plot_stress_end_to_end(rec_e2e, per_scenario=(layout == "scenario"), zeta=zeta)
        if rec_align:
            plot_stress_alignment(rec_align, per_scenario=(layout == "scenario"))
    write_stress_tables(rec_e2e, rec_align)
    for root, _, files in os.walk("figures"):
        print(root, sorted(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="*", default=["all"],
                        help="alignment | e2e | stress_align | stress_e2e | plots | all")
    parser.add_argument("--scenarios", nargs="*", default=None, help="scenario keys for the stress tests (see data.py)")
    parser.add_argument("--layout", choices=["all", "group", "scenario"], default="all",
                        help="plots: all stress scenarios in one image | one image per type of variation | "
                             "one image per scenario")
    parser.add_argument("--per-scenario", action="store_true", help="plots: same as --layout scenario")
    parser.add_argument("--zeta", type=float, default=ZETAS[-1],
                        help="plots: step size eta = 3 d^zeta shown in the one-panel-per-scenario figures "
                             "(0.625, 0.75 or 0.875)")
    parser.add_argument("--dims", nargs="*", type=int, default=None,
                        help="stress_align: dimensions to sweep per scenario (default: d = 300 only)")
    parser.add_argument("--alpha-e2e", type=float, default=ALPHA_E2E,
                        help="e2e/stress_e2e: first-stage samples n = alpha * d (default 2)")
    parser.add_argument("--m-per-d", type=float, default=M_PER_D,
                        help="e2e/stress_e2e: width m = m_per_d * d (default 4/3, i.e. m = 400 at d = 300)")
    args = parser.parse_args()
    ALPHA_E2E, M_PER_D = args.alpha_e2e, args.m_per_d
    what = ["alignment", "e2e", "stress_align", "stress_e2e", "plots"] if "all" in args.what else args.what
    for w in what:
        if w == "alignment":
            run_experiment_dimension_alignment()
        elif w == "e2e":
            run_experiment_end_to_end()
        elif w == "stress_align":
            run_stress_alignment(args.scenarios, dims=args.dims)
        elif w == "stress_e2e":
            run_stress_end_to_end(args.scenarios)
        elif w == "plots":
            make_all_plots(layout="scenario" if args.per_scenario else args.layout, zeta=args.zeta)
        else:
            sys.exit(f"unknown experiment: {w}")
