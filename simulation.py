import numpy as np
import matplotlib.pyplot as plt
from astropy.units import beam_angular_area


# =============================================================
# PHYSICAL MODEL (AIRY DISK + DEFOCUS)
# =============================================================

def airy_disk_diameter(lam, f_number):
    """Diffraction-limited Airy disk diameter"""
    return 2.44 * lam * f_number


def spot_size_with_defocus(z, z_focus, lam, D):
    """
    Spot size increases with defocus.
    Combines diffraction + defocus blur.
    """
    f_number = z_focus / D
    d_diff = airy_disk_diameter(lam, f_number)

    # Defocus blur (quadratic growth)
    dz = z - z_focus
    d_defocus = np.abs(dz) * (D / z_focus)

    return np.sqrt(d_diff**2 + d_defocus**2)


def focus_error_signal(z, z_focus):
    """
    Signed focus error (this is the key fix!)
    Mimics astigmatic autofocus systems.
    """
    return (z - z_focus)


# =============================================================
# SYSTEM PARAMETERS
# =============================================================

lam = 520e-9
f_lens = 0.30
D = 0.08

z_focus = f_lens

# disturbance
z_ccd_nominal = 0.30
z_ccd_disturbed = 0.303   # small  shift
t_disturb = 0.5

# simulation
dt = 0.001
T = 1.5
time = np.arange(0, T, dt)

# actuator
tau = 0.02
v = 0

# PID gains (reasonable starting point)
Kp = 50
Ki = 200
Kd = 0.5

# noise
noise_std = 1e-6

print(spot_size_with_defocus(z_focus, z_focus, lam, D))
# =============================================================
# DISTURBANCE
# =============================================================

def disturbance(t):
    return z_ccd_nominal if t < t_disturb else z_ccd_disturbed


# =============================================================
# SIMULATION
# =============================================================

def simulate_with_gains(Kp, Ki, Kd, seed=42):
    np.random.seed(seed)

    z_lens = 0.0
    v = 0.0

    I = 0.0
    prev_error = 0.0
    D_filtered = 0.0
    alpha = 0.1

    spot_sizes = []

    for t in time:
        z_ccd = disturbance(t)
        z = z_ccd - z_lens

        # measurement
        spot = spot_size_with_defocus(z, z_focus, lam, D)
        noise = np.random.normal(0, noise_std)
        measured = spot + noise

        # signed error (KEY FIX)
        error = z - z_focus

        # PID
        P = Kp * error

        I += error * dt
        I = np.clip(I, -0.02, 0.02)

        d_raw = (error - prev_error) / dt
        D_filtered = alpha * d_raw + (1 - alpha) * D_filtered
        D_term = Kd * D_filtered

        control = P + Ki * I + D_term
        prev_error = error

        # actuator dynamics
        v += (control - v) / tau * dt
        z_lens += v * dt
        z_lens = np.clip(z_lens, -0.1, z_ccd)

        spot_sizes.append(measured)

    return np.array(spot_sizes)


# =============================================================
# RQ1 — FIRST & SECOND MOMENTS
# =============================================================

def rq1_analyze_coefficient(mode, values, n_repeats=5):
    means = []
    stds = []

    for val in values:
        Kp = val if mode == 'P' else 0
        Ki = val if mode == 'I' else 0
        Kd = val if mode == 'D' else 0

        all_data = []

        for rep in range(n_repeats):
            spot = simulate_with_gains(Kp, Ki, Kd, seed=rep)

            # only steady-state (last 20%)
            start = int(0.8 * len(spot))
            all_data.append(spot[start:])

        all_data = np.concatenate(all_data)

        means.append(np.mean(all_data))
        stds.append(np.std(all_data))

    return values, np.array(means), np.array(stds)


def plot_rq1():
    # tuned ranges for stable system
    Kp_values = np.linspace(0, 150, 60)
    Ki_values = np.linspace(0, 400, 60)
    Kd_values = np.linspace(0, 2.0, 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, (mode, values, label) in enumerate([
        ('P', Kp_values, 'Proportional (Kp)'),
        ('I', Ki_values, 'Integral (Ki)'),
        ('D', Kd_values, 'Derivative (Kd)')
    ]):
        print(f"Analyzing {mode}...")

        coeffs, means, stds = rq1_analyze_coefficient(mode, values)

        # First moment
        axes[0, idx].plot(coeffs, means * 1e6, 'b-', linewidth=2)
        axes[0, idx].set_title(f'First Moment ({mode})')
        axes[0, idx].set_xlabel(label)
        axes[0, idx].set_ylabel('Mean Spot (µm)')
        axes[0, idx].grid(True, alpha=0.3)

        # Second moment
        axes[1, idx].plot(coeffs, stds * 1e6, 'g-', linewidth=2)
        axes[1, idx].set_title(f'Second Moment ({mode})')
        axes[1, idx].set_xlabel(label)
        axes[1, idx].set_ylabel('Std Dev (µm)')
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('RQ1: PID Coefficient Effects (Stable Autofocus Model)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rq1_moments.pdf')
    plt.show()


# =============================================================
# RQ2 — TIME DOMAIN RESPONSES
# =============================================================

def classify_response(signal, tolerance=0.05):

    disturb_idx = int(t_disturb / dt)
    settle_delay = 0.15  # seconds (tunable)
    check_start = int((t_disturb + settle_delay) / dt)
    target = np.mean(signal[-100:])

    post_settle = signal[check_start:]

    if np.mean(post_settle) > 3 * target:
        return "unstable"

    window_end = int((t_disturb + 0.3) / dt)
    segment = signal[disturb_idx:window_end]

    within = np.abs(segment - target) / target < tolerance

    if np.all(within):
        return "stable"

    diffs = np.diff(segment)
    zero_cross = np.sum(np.diff(np.sign(diffs)) != 0)

    if zero_cross > 10:
        return "oscillatory"

    return "underdamped"

def plot_rq2():
    combinations = [
        # -----------------------------
        # P-ONLY
        # -----------------------------
        (20, 0, 0, 'P low (stable, slow)'),
        (100, 0, 0, 'P high (damped oscillatory)'),
        (200, 0, 0, 'P very high (damped oscillatory)'),
        (300, 0, 0, 'P extreme (damped oscillatory)'),

        # -----------------------------
        # PI CONTROL
        # -----------------------------
        (40, 100, 0, 'PI moderate (stable)'),
        (80, 250, 0, 'PI aggressive (underdamped)'),
        (120, 400, 0, 'PI very aggressive (damped oscillatory)'),
        (120, 800, 0, 'PI extreme (damped oscillatory)'),

        # -----------------------------
        # PD CONTROL
        # -----------------------------
        (40, 0, 0.5, 'PD moderate (stable)'),
        (80, 0, 1.5, 'PD aggressive (stable)'),
        (120, 0, 0.1, 'PD weak damping (damped oscillatory)'),
        (60, 0, 5.0, 'PD extreme D (stable)'),

        # -----------------------------
        # FULL PID
        # -----------------------------
        (50, 150, 0.5, 'PID balanced (stable)'),
        (120, 300, 1.5, 'PID aggressive (stable)'),
        (150, 400, 0.2, 'PID low D (damped oscillatory)'),
        (200, 600, 0.5, 'PID extreme (damped oscillatory)'),

        # -----------------------------
        # EDGE CASES
        # -----------------------------
        (0, 300, 0, 'I-only (unstable)'),
        (0, 0, 2.0, 'D-only (unstable)'),
    ]

    n_plots = len(combinations)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    colors = {
        'stable': 'green',
        'underdamped': 'orange',
        'oscillatory': 'purple',
        'unstable': 'red'
    }

    for idx, (Kp, Ki, Kd, label) in enumerate(combinations):
        spot = simulate_with_gains(Kp, Ki, Kd)
        classification = classify_response(spot)

        ax = axes[idx]
        ax.plot(time, spot * 1e6, color=colors.get(classification, 'black'),
                label="beam diameter")

        ax.axvline(t_disturb, color='red', linestyle='--', label='disturbance')

        ax.set_title(f'{label}\nKp={Kp}, Ki={Ki}, Kd={Kd}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spot (µm)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(len(combinations), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('RQ2: Time-Domain Responses (Stable Autofocus)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rq2_responses.pdf')
    plt.show()


# =============================================================
# EXECUTE
# =============================================================

print("\n=== RQ1 ===")
plot_rq1()

print("\n=== RQ2 ===")
plot_rq2()

