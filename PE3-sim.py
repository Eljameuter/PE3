"""
Simplified Autofocus PID Simulation
Physics: Gaussian beam propagation through lens to CCD
Disturbance: CCD position changes at t=0.5s
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================
# PHYSICS MODELS
# =============================================================
def min_beam_radius(f,wav,aperature):
    """Airy disk radius for a given CCD distance from the lens."""
    return 1.029 * wav * f / aperature  # [m]

def beam_radius_at_distance(x, f_lens, wav, aperature):
    """Gaussian beam radius at distance x from lens"""
    W0 = wav*f_lens/(np.pi*1e-3/2) #min_beam_radius(f_lens, wav, aperature)
    Z_R = np.pi * (W0 ** 2)*1.5168 / wav
    z = x - f_lens
    return W0 * np.sqrt(1.0 + (z / Z_R) ** 2)


# =============================================================
# SYSTEM PARAMETERS
# =============================================================

wav = 520e-9  # wavelength [m]
f_lens = 300e-3  # focal length [m]
d_lens = 80e-3  # lens aperture [m]
d_laser = 1e-3  # input beam diameter [m]

# positions
x_initial = 0
x_ccd_nominal = 0.30  # [m]
x_ccd_disturbed = x_ccd_nominal+0.5  # [m]
t_disturb = 0.5  # disturbance time [s]

# Simulation parameters
dt = 0.001  # time step [s]
record_time = 1.0  # total simulation time [s]
time = np.arange(0, record_time, dt)

# PID and system parameters
d_set = 2*beam_radius_at_distance(f_lens, f_lens, wav,d_lens)

tau = 0.005  # lens response time constant
readout_noise = 0.01*d_set  # noise standard deviation


# =============================================================
# SIMULATION ENGINE
# =============================================================

def disturbance(t):
    """CCD position over time"""
    return x_ccd_nominal if t < t_disturb else x_ccd_disturbed


def simulate_run(Kp, Ki, Kd, seed=42):
    """Run PID autofocus simulation"""
    np.random.seed(seed)

    x_lens = x_initial
    errors = []
    beam_diameters = []

    for i, t in enumerate(time):
        # Current CCD position
        x_ccd = disturbance(t)

        # Distance from lens to CCD
        dist = x_ccd - x_lens

        # Measure beam diameter with noise
        noise = np.random.normal(0, readout_noise)
        measured_d = 2 * beam_radius_at_distance(dist, f_lens, wav, d_lens) + noise
        beam_diameters.append(measured_d)

        # PID control
        error = d_set - measured_d
        errors.append(error)

        P = Kp * error

        if i > 0:
            I = Ki * np.trapezoid(errors, dx=dt)
            D = Kd * (error - errors[-2]) / dt if i > 1 else 0
        else:
            I, D = 0, 0

        control = P + I + D

        # Update lens position
        x_lens += (control) * dt / tau
        x_lens = np.clip(x_lens, -1, x_ccd)

    return np.array(beam_diameters)


# =============================================================
# RESEARCH QUESTION 1: First and Second Moments
# =============================================================

def rq1_analyze_coefficient(mode, values, n_repeats=5):
    """
    Analyze effect of varying one coefficient while others are zero.
    Returns: (coefficient_values, means, stds)
    """
    means = []
    stds = []

    for val in values:
        # Set coefficients based on mode
        Kp = val if mode == 'P' else 0
        Ki = val if mode == 'I' else 0
        Kd = val if mode == 'D' else 0

        # Run multiple repeats
        all_diameters = []
        for rep in range(n_repeats):
            diameters = simulate_run(Kp, Ki, Kd, seed=rep)
            all_diameters.append(diameters)

        # Calculate first and second moments (averaged over time and repeats)
        all_diameters = np.array(all_diameters)  # shape: (repeats, time_points)

        # First moment: mean diameter over time and repeats
        first_moment = np.mean(all_diameters)

        # Second moment: standard deviation of diameter over time and repeats
        second_moment = np.std(all_diameters)

        means.append(first_moment)
        stds.append(second_moment)

    return values, np.array(means), np.array(stds)


def plot_rq1():
    """Plot first and second moments for P, I, D sweeps"""
    # Coefficient ranges
    Kp_values = np.linspace(0, 1e4, 100)
    Ki_values = np.linspace(0, 1e4, 100)
    Kd_values = np.linspace(0, 1e4, 100)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Analyze and plot for each mode
    for idx, (mode, values, label, unit) in enumerate([
        ('P', Kp_values, 'Proportional (Kp)', ''),
        ('I', Ki_values, 'Integral (Ki)', ''),
        ('D', Kd_values, 'Derivative (Kd)', '')
    ]):
        print(f"Analyzing {mode} coefficient...")
        coeffs, means, stds = rq1_analyze_coefficient(mode, values)
        # First moment plot
        axes[0, idx].plot(coeffs, means, 'b-', linewidth=2)
        axes[0, idx].axhline(d_set, color='r', linestyle='--', label=f'd_set = {d_set * 1e6:.1f} µm')
        axes[0, idx].set_xlabel(f'{label}')
        axes[0, idx].set_ylabel('Mean Diameter (m)')
        axes[0, idx].set_title(f'First Moment: Effect of {mode}')
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].legend()

        # Second moment plot
        axes[1, idx].plot(coeffs, stds, 'g-', linewidth=2)
        axes[1, idx].set_xlabel(f'{label}')
        axes[1, idx].set_ylabel('Std Deviation (m)')
        axes[1, idx].set_title(f'Second Moment: Effect of {mode}')
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('Research Question 1: PID Coefficient Effects on Beam Diameter', fontsize=14)
    plt.tight_layout()
    plt.savefig('rq1_moments.pdf')
    plt.show()


# =============================================================
# RESEARCH QUESTION 2: Time Domain Responses
# =============================================================

def classify_response(diameters, settling_time=0.3, tolerance=0.05):
    """
    Classify system response based on beam diameter behavior
    """
    # Time indices after disturbance
    t_start_idx = int(t_disturb / dt)
    t_end_idx = int((t_disturb + settling_time) / dt)

    if t_end_idx >= len(diameters):
        t_end_idx = len(diameters) - 1

    post_disturbance = diameters[t_start_idx:t_end_idx]

    # Check if settled within tolerance
    within_tolerance = np.abs(post_disturbance - d_set) / d_set < tolerance

    if np.all(within_tolerance):
        return "overdamped" if len(post_disturbance) > 100 else "stable"
    elif np.any(np.abs(diameters[t_start_idx:]) > 3 * d_set):
        return "unstable"
    else:
        # Check for oscillations
        diffs = np.diff(post_disturbance)
        zero_crossings = np.sum(np.diff(np.sign(diffs)) != 0)
        if zero_crossings > 10:
            return "oscillatory"
        else:
            return "underdamped"


def plot_rq2_combinations(combinations):
    """
    Plot time-domain responses for different PID combinations
    """
    n_plots = len(combinations)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color map for different response types
    colors = {
        'stable': 'green',
        'overdamped': 'blue',
        'underdamped': 'orange',
        'oscillatory': 'purple',
        'unstable': 'red'
    }

    for idx, (Kp, Ki, Kd, label) in enumerate(combinations):
        diameters = simulate_run(Kp, Ki, Kd, seed=42)
        classification = classify_response(diameters)

        ax = axes[idx]
        ax.plot(time, diameters * 1e6, color=colors.get(classification, 'black'),
                linewidth=1.5, label=f'Response: {classification}')
        ax.axhline(d_set * 1e6, color='red', linestyle='--', linewidth=1,
                   label=f'Setpoint: {d_set * 1e6:.1f} µm')
        ax.axvline(t_disturb, color='orange', linestyle=':', linewidth=1.5,
                   label='Disturbance')

        # Add 5% tolerance band
        ax.fill_between(time,
                        (d_set * 0.95) * 1e6,
                        (d_set * 1.05) * 1e6,
                        color='gray', alpha=0.2, label='±5% band')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Beam Diameter (µm)')
        ax.set_title(f'{label}: Kp={Kp}, Ki={Ki}, Kd={Kd}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Set reasonable y-limits
        y_max = min(1.5 * d_set * 1e6, np.percentile(diameters * 1e6, 95))
        #ax.set_ylim(0, max(y_max, d_set * 1.5e6))

    # Hide unused subplots
    for idx in range(len(combinations), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Research Question 2: Time-Domain Responses for Different PID Combinations',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('rq2_responses.pdf', dpi=150)
    plt.show()


def run_rq2():
    """Run RQ2 analysis with selected PID combinations"""

    # Selected PID combinations to test
    combinations = [
        # (Kp, Ki, Kd, label)
        (0.5, 0, 0, 'P-only (low)'),
        (5, 0, 0, 'P-only (high)'),
        (1, 1, 0, 'PI (low)'),
        (5, 5, 0, 'PI (high)'),
        (1, 0, 0.01, 'PD (low)'),
        (5, 0, 0.05, 'PD (high)'),
        (1, 1, 0.01, 'PID (balanced)'),
        (10, 10, 0.05, 'PID (aggressive)'),
    ]

    print("Running RQ2 simulations...")
    plot_rq2_combinations(combinations)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RQ2 Summary Statistics")
    print("=" * 60)
    for Kp, Ki, Kd, label in combinations:
        diameters = simulate_run(Kp, Ki, Kd, seed=42)
        classification = classify_response(diameters)

        # Calculate steady-state error (last 20% of simulation)
        steady_state = diameters[int(0.8 * len(diameters)):]
        mean_ss = np.mean(steady_state)
        std_ss = np.std(steady_state)

        print(f"\n{label}:")
        print(f"  Classification: {classification}")
        print(f"  Steady-state mean: {mean_ss * 1e6:.2f} µm (target: {d_set * 1e6:.2f} µm)")
        print(f"  Steady-state std: {std_ss * 1e6:.2f} µm")
        print(f"  Error: {(mean_ss - d_set) * 1e6:.2f} µm")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("Autofocus PID Simulation")
    print(f"Setpoint diameter: {d_set * 1e6:.2f} µm")
    print(f"Simulation time: {record_time} s, dt={dt} s")
    print("-" * 50)

    # Run RQ1
    print("\n=== RESEARCH QUESTION 1 ===")
    plot_rq1()

    # Run RQ2
    print("\n=== RESEARCH QUESTION 2 ===")
    run_rq2()