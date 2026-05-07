"""
Autofocus PID Simulation
 equations used:
  Airy disk radius : x = 1.22 * lambda * f / d          (Eq. 6)
  CCD diameter     : D_CCD = D_beam + d(t) + 0.01*sigma (Hypothesis)
  PID control      : u(t) = -(Kp*e + Ki*integral(e) + Kd*de/dt)  (Eq. 1)

System parameters (from Table in §2.2):
  f_lens  = 100e-3 m   (focal length)
  d_lens  = 80e-3  m   (lens diameter / aperture)
  lambda  = 520e-9 m   (wavelength)
  d_laser = 1e-3   m   (laser beam diameter)

Disturbance at t=0.5 s: CCD position steps from 0.3 m -> 0.25 m
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Airy disk radius as function of CCD distance ───────────────────────────────
def min_beam_diameter(f,wav,aperature):
    """Airy disk radius for a given CCD distance from the lens."""
    return 2*1.22 * wav * f / aperature  # [m]


# ── System parameters ──────────────────────────────────────────────────────────
wav = 520e-9  # wavelength [m]
f_lens = 300e-3  # focal length [m]
d_lens = 80e-3  # lens diameter (aperture) [m]
d_laser = 1e-3  # input beam diameter [m]

W_IN = d_lens / 2  # [m]
W0 = wav * f_lens / (np.pi * W_IN)  # [m]  focused beam waist radius

# Rayleigh range: z_R = π * w0² / λ
Z_R = np.pi * W0 ** 2 / wav  # [m]


# Nominal CCD position and disturbed position
x_initial = 0
x_ccd_nominal = 0.31  # [m]  initial distance laser→CCD
x_ccd_disturbed = 0.29  # [m]  position after disturbance

t_disturb = 0.5  # time of disturbance [s]

dt = 1 / 1000
record_time = 0.5
time = np.arange(0, record_time, dt)

d_set = min_beam_diameter(f_lens, wav, d_laser)
d_max = d_set + d_set * 0.02
tau = 0.005                    # lens response time constant
readout_noise = 0.01           # σ of Gaussian readout noise


def d_at_x(x):
    """
        Gaussian beam radius at CCD position x_ccd (measured from lens).
        Minimum at the focal point (x_ccd = F_LENS), grows on both sides:
          w(z) = w0 * sqrt(1 + (z / z_R)^2)   with z = x_ccd - F_LENS
        """
    z = x - f_lens
    return W0 * np.sqrt(1.0 + (z / Z_R) ** 2)



# Setpoint: beam radius when CCD is at nominal position
r_setpoint = d_at_x(x_ccd_nominal)
print(f"Setpoint beam radius  : {r_setpoint * 1e6:.4f} µm")
print(f"Setpoint beam diameter: {2 * r_setpoint * 1e6:.4f} µm")

def disturbance(t):
    return x_ccd_nominal if t < t_disturb else x_ccd_disturbed

def simulate_run(Kp, Ki, Kd, seed = None):
    """

    """
    if seed is not None:
        np.random.seed(seed)

    errors = []
    d_on_ccd = []
    x_lens = x_initial

    for i, t in enumerate(time):
        x_ccd = disturbance(t)
        x = x_ccd - x_lens
        noise = np.random.normal(0, readout_noise)

        pd_d =d_at_x(x) + noise
        d_on_ccd.append(pd_d)

        error = d_set - pd_d
        errors.append(error)

        P = Kp * error
        I = Ki * np.trapezoid(errors, dx=dt) if i > 1 else 0
        D = Kd * (errors[i] - errors[i - 1]) / dt if i > 0 else 0

        control = P + I + D

        x_lens += (control - x_lens) * dt / tau
        x_lens = np.clip(x_lens, -10, x_ccd)

    return np.array(d_on_ccd)


# =============================================================
# RESEARCH QUESTION 1
# Individual coefficient sweeps (Kp, Ki or Kd from 0 → 20).
# For each value: run 3 repeats, average, compute 1st & 2nd
# moment, then plot both moments vs coefficient value.
# =============================================================

coeff_values = np.arange(0, 1e6, 100)
RAW_DIR  = "raw_runs_rq1"
COMB_DIR = "combined_data_rq1"

def rq1_run_experiment(n_repeats=3):
    os.makedirs(RAW_DIR, exist_ok=True)
    for repeat in range(n_repeats):
        for mode in ["P", "I", "D"]:
            for value in coeff_values:
                Kp = value if mode == "P" else 0
                Ki = value if mode == "I" else 0
                Kd = value if mode == "D" else 0

                data = simulate_run(Kp, Ki, Kd)
                fname = f"{RAW_DIR}/{mode}_{value:.1f}_run{repeat}.csv"
                pd.DataFrame(data, columns=["diameter"]).to_csv(fname, index=False)
    print(f"RQ1: saved {n_repeats} repeats × 3 modes × {len(coeff_values)} values.")


def rq1_combine_runs(n_repeats=3):
    os.makedirs(COMB_DIR, exist_ok=True)
    for mode in ["P", "I", "D"]:
        for value in coeff_values:
            runs = []
            for repeat in range(n_repeats):
                fname = f"{RAW_DIR}/{mode}_{value:.1f}_run{repeat}.csv"
                runs.append(pd.read_csv(fname)["diameter"].values)

            runs = np.array(runs)                         # shape (n_repeats, N)
            mean_trace = np.mean(runs, axis=0)
            std_trace  = np.std(runs,  axis=0, ddof=1)

            pd.DataFrame({"diameter": mean_trace, "stdev": std_trace}).to_csv(
                f"{COMB_DIR}/{mode}_{value:.1f}.csv", index=False
            )


def rq1_load_statistics():
    stats = {m: {"coeff": [], "mean": [], "stdev": [], "mean_err": [], "stded_err": []}
             for m in ["P", "I", "D"]}

    for mode in ["P", "I", "D"]:
        for value in coeff_values:
            df = pd.read_csv(f"{COMB_DIR}/{mode}_{value:.1f}.csv")
            diameter = df["diameter"].values
            stdev   = df["stdev"].values
            N = len(diameter)

            mean  = np.mean(diameter)
            sigma = np.std(diameter, ddof=1)

            sigma_mean  = (1 / N) * np.sqrt(np.sum(stdev ** 2))
            sigma_stdev = sigma / np.sqrt(2 * (N - 1))

            stats[mode]["coeff"].append(value)
            stats[mode]["mean"].append(mean)
            stats[mode]["stdev"].append(sigma)
            stats[mode]["mean_err"].append(sigma_mean)
            stats[mode]["stded_err"].append(sigma_stdev)

    return stats


def rq1_plot(stats):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for mode, label in zip(["P", "I", "D"], ["Kp", "Ki", "Kd"]):
        axes[0].errorbar(stats[mode]["coeff"], stats[mode]["mean"],
                         yerr=stats[mode]["mean_err"], label=label, capsize=3)
        axes[1].errorbar(stats[mode]["coeff"], stats[mode]["stdev"],
                         yerr=stats[mode]["stded_err"], label=label, capsize=3)

    for ax, title, ylabel in zip(
        axes,
        ["1st Moment (Mean) vs PID Coefficient", "2nd Moment (Std Dev) vs PID Coefficient"],
        ["Mean laser diameter (m)", "Standard deviation (m)"]
    ):
        ax.set_xlabel("Coefficient value")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(d_set, color="k", linestyle="--", linewidth=0.8, label="d_set")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig("rq1_moments.pdf")
    plt.show()
    print("RQ1 plot saved to rq1_moments.pdf")


def run_rq1():
    rq1_run_experiment()
    rq1_combine_runs()
    stats = rq1_load_statistics()
    rq1_plot(stats)


# =============================================================
# RESEARCH QUESTION 2
# Time-domain stability analysis.
#
# # Measure time-domain responses for selected (Kp, Ki, Kd)
# # combinations that produce stable, overdamped, oscillatory
# # and unstable behaviour.
# =============================================================



def rq2_plot_case(Kp, Ki, Kd, label, ax):
    d_on_ccd = simulate_run(Kp, Ki, Kd, seed=42)
    classification = classify_response(d_on_ccd)

    color_map = {"stable": "tab:green", "overdamped": "tab:blue",
                 "oscillatory": "tab:orange", "unstable": "tab:red"}
    ax.plot(time, d_on_ccd, color=color_map.get(classification, "k"), linewidth=1.2)
    ax.axhline(d_set, color="red", linestyle="--", linewidth=0.8, label="d_set")
    ax.fill_between(time, d_set * 0.95, d_set * 1.05,
                    color="gray", alpha=0.2, label="5% band")
    ax.axvline(0.05, color="orange", linestyle=":", linewidth=0.8, label="Disturbance on")
    ax.set_title(f"{label}\n[{classification.upper()}]  Kp={Kp}, Ki={Ki}, Kd={Kd}",
                 fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("d_pd (V)", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.4)



# ── Fine 3-D grid search ─────────────────────────────────────────────
# Ranges chosen from physical analysis of the stability boundaries:
#   Kp ∈ [0.1, 2.0]  step 0.1   (20 values)
#   Ki ∈ [0,  50]    step 2     (26 values)
#   Kd ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}  (6 values)
#   Total: 20 × 26 × 6 = 3 120 runs
KP_GRID = np.arange(0, 25, 5)   # 5 values
KI_GRID = np.arange(0,   10,   5)  # 2 values
KD_GRID = np.array([0.0, 0.005, 0.01, 0.05])  # 4 values


def rq2_grid_search():
    """
    Fine 3-D grid search over Kp × Ki × Kd.
    Returns a DataFrame with columns [Kp, Ki, Kd, class].
    Saves rq2_grid_search.csv.
    """
    rows = []
    total = len(KP_GRID) * len(KI_GRID) * len(KD_GRID)
    print(f"RQ2 grid search: {total} combinations…")

    for i, Kp in enumerate(KP_GRID):
        for Ki in KI_GRID:
            for Kd in KD_GRID:
                d_on_ccd = simulate_run(Kp, Ki, Kd, seed=0)
                #add subplot with
                rows.append({"Kp": float(Kp), "Ki": float(Ki),
                              "Kd": float(Kd), "class": cls})
        if (i + 1) % 5 == 0:
            print(f"  Kp {i+1}/{len(KP_GRID)} done")

    df = pd.DataFrame(rows)
    df.to_csv("rq2_grid_search.csv", index=False)

    print("\nClassification counts:")
    print(df["class"].value_counts().to_string())
    print("\nExample stable combinations:")
    print(df[df["class"] == "stable"].head(8).to_string(index=False))
    print("\nExample unstable combinations:")
    print(df[df["class"] == "unstable"].head(8).to_string(index=False))
    return df


def rq2_plot_cases(n_repeats=3):
    """
    Plot the photodiode voltage trace over time for every (Kp, Ki, Kd)
    combination that has been measured, averaging over all available repeats.
    """
    # Build the full list of tested combinations
    combinations = [
        (Kp, Ki, Kd)
        for Kp in KP_GRID
        for Ki in KI_GRID
        for Kd in KD_GRID
    ]

    # Keep only combinations that have at least one saved file
    measured = []
    for Kp, Ki, Kd in combinations:
        runs = []
        for repeat in range(n_repeats):
            fname = _rq2_fname(Kp, Ki, Kd, repeat)
            if os.path.exists(fname):
                runs.append(pd.read_csv(fname)["voltage"].values)
        if runs:
            measured.append((Kp, Ki, Kd, runs))

    if not measured:
        print("rq2_plot_cases: no saved runs found in", RAW_DIR_RQ2)
        return

    n = len(measured)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()

    settling_threshold = 0.02 * V_set

    for idx, (Kp, Ki, Kd, runs) in enumerate(measured):
        mean_voltages = np.mean(runs, axis=0)
        t = np.linspace(0, record_time, len(mean_voltages))

        ax = axes[idx]
        ax.plot(t, mean_voltages, linewidth=1.2, label=f"mean ({len(runs)} rep)")
        ax.axhline(V_set, color="red", linestyle="--", linewidth=0.8, label="V_set")
        ax.fill_between(t,
                        V_set - settling_threshold, V_set + settling_threshold,
                        color="gray", alpha=0.2, label="2 % band")
        ax.axvline(0.2, color="orange", linestyle=":", linewidth=0.8, label="Disturbance on")
        ax.set_title(f"Kp={Kp}, Ki={Ki}, Kd={Kd}", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("V_pd (V)", fontsize=7)
        ax.set_ylim(max(0, V_set - 0.05), V_set + 0.05)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.4)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("RQ2 – Combinations tested over time (measured)", fontsize=12)
    plt.tight_layout()
    plt.savefig("rq2_measured.pdf")
    plt.show()

def run_rq2():
    df = rq2_grid_search()
    rq2_plot_stability_maps(df)


# =============================================================
# Entry point
# =============================================================
if __name__ == "__main__":
    print("=== Research Question 1: Coefficient sweeps ===")
    run_rq1()

    print("\n=== Research Question 2: Stability analysis ===")
    #run_rq2()
