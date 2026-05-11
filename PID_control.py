"""
Created on Thu May  7 12:51:11 2026

@author: Elja en Luna

"""
from matplotlib import pyplot as plt
import os
import threading
from PIL import Image, ImageDraw
from pypylon import pylon
import pytrinamic
from pytrinamic.connections import ConnectionManager
from pytrinamic.modules import TMCM6110
from scipy.ndimage import convolve
import pandas as pd
from datetime import datetime
import numpy as np
import time

"""
Move a TMCM6110-controlled linear stage through its full travel in 1 mm steps
and capture one image at each step using a Basler camera (pypylon).

EDIT THESE VALUES BEFORE RUNNING:
- COM_PORT
- AXIS_INDEX
- FULL_RANGE_MM
- STEPS_PER_MM   (depends on your stage mechanics)
- SAVE_FOLDER

Requirements:
pip install pypylon pytrinamic

"""
# ================================================================================================
# Determine set point from sweep 

# Apply disturbance --> bring lens completely to the right for example 

# Choose values for K_p, K_i and K_d

# Iteration:
# 1. capture an image of the interference pattern and determine its diameter (brightest pixel + signal)
# 2. determine the error e(t) = set point - signal and the control signal u(t) 
# unit control signal: steps per second 
# 3. send the control signal to the motor; we measure the time; the motor moves with a speed u(t)
# during a time interval dt; we capture an image and measure the time every x steps and determine 
# the corresponding signal; we save the diameters and times in arrays
# 4. after the time interval dt, determine the error again; we measure until we have reached t_0 
# ================================================================================================

# ==========================================================
# PID SETTINGS
# ==========================================================
set_point = 550000 # signal
T = 120 # seconds
dt = 0.5 # seconds



# ==========================================================
# USER SETTINGS
# ==========================================================
COM_PORT = "COM6"
AXIS_INDEX = 0              # motor index used in your example
FULL_RANGE_MM = 10          # total travel range of stage in mm
STEP_MM = 0.5               # move in 1 mm increments
STEPS_PER_MM = int(1e-3/(0.5e-9*8))

SETTLE_TIME = 0.5           # seconds after move before image capture
GRAB_TIMEOUT = 3000         # ms
SAVE_FOLDER = "scan_images"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# ==========================================================
# HELPERS
# ==========================================================
def save_image(camera, filename):
    result = camera.GrabOne(3000)

    if not result.GrabSucceeded():
        raise RuntimeError("Grab failed")

    img = pylon.PylonImage()
    img.AttachGrabResultBuffer(result)
    img.Save(pylon.ImageFileFormat_Png, filename)
    img.Release()

def sum_around_brightest(path, radius=40):
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    #dark = np.array(Image.open("Dark/dark_1.jpeg").convert("L"), dtype=np.float32)
    #img = img - dark
    # Sum of 5 consecutive pixels in each row → find the peak window center
    kernel = np.ones((5, 5))
    smoothed = convolve(img, kernel, mode='reflect')
    row, col = np.unravel_index(np.argmax(smoothed), img.shape)

    r0, r1 = max(row - radius, 0), min(row + radius + 1, img.shape[0])
    c0, c1 = max(col - radius, 0), min(col + radius + 1, img.shape[1])

    return img[r0:r1, c0:c1].sum(), (row, col)

def return_home_and_close(motor, cam):
    """Safe shutdown: return to zero, stop camera."""
    print("Returning to zero...")
    motor.move_to(0)
    while not motor.get_position_reached():
        time.sleep(0.05)
    cam.StopGrabbing()
    cam.Close()
    print("Shutdown complete.")

panic = threading.Event()

def listen_for_panic():
    """Runs in background thread — sets panic flag when 'c' is pressed."""
    while not panic.is_set():
        key = input()
        if key.strip().lower() == "c":
            print("\n⚠️  PANIC: returning to zero and shutting down...")
            panic.set()
            break
# ==========================================================
# MAIN
# ==========================================================
def pid(kp,kd,ki):
    K_P = kp
    K_D = kd
    K_I = ki
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    pytrinamic.show_info()

    # ---------------- Camera Setup ----------------
    tlf = pylon.TlFactory.GetInstance()
    cam = pylon.InstantCamera(tlf.CreateFirstDevice())

    cam.Open()
    # ---------------- Start panic listener ----------------
    listener = threading.Thread(target=listen_for_panic, daemon=True)
    listener.start()
    print("Running — press 'c' + Enter at any time to panic-stop.\n")

    # ---------------- Motor Setup -----------------
    connection_manager = ConnectionManager(
        f"--interface usb_tmcl --port {COM_PORT}"
    )

    with connection_manager.connect() as interface:
        module = TMCM6110(interface)
        motor = module.motors[AXIS_INDEX]

        print("Configuring motor...")

        motor.drive_settings.max_current = 200
        motor.drive_settings.standby_current = 0
        motor.drive_settings.boost_current = 0
        motor.drive_settings.microstep_resolution = (
            motor.ENUM.microstep_resolution_256_microsteps
        )

        motor.max_acceleration = 1000
        motor.max_velocity = 1000

        # Zero current position
        motor.actual_position = 0

        # --------------------------------------------------
        # PID control algorithm
        # --------------------------------------------------
        times = []
        signals = []
        errors = []
        positions = []
        start_time = datetime.now()

        for i in range(int(T/dt)): 
            # capture and save image
            filename = "PID_storage/photo.png"
            print(f"Capturing {filename}")
            save_image(cam, filename)
            if panic.is_set():
                break
            # save corresponding measurement time 
            now = datetime.now()
            elapsed_time = now - start_time
            elapsed_seconds = elapsed_time.total_seconds()
            times.append(elapsed_seconds)
            print(f" elapsed time: {elapsed_seconds}")
            # determine signal!
            signal, pos = sum_around_brightest(filename)
            signals.append(signal)
            print(f" signal: {signal}")
            # determine control function
            error = set_point - signal 
            errors.append(error)
            
            P = K_P * error
            I = K_I * np.trapezoid(errors, dx=dt) if i > 1 else 0
            D = K_D * (errors[i] - errors[i - 1]) / dt if i > 0 else 0
            control = P + I + D
            control = np.clip(control, -STEPS_PER_MM, STEPS_PER_MM)
            print(f" control: {control}")
            new_pos = int(float(motor.actual_position)+control)
            new_pos = np.clip(new_pos, 0,10*STEPS_PER_MM)
            positions.append(new_pos)
            print(f" new position: {new_pos}")
            # apply control function 
            print("Move to new position: " + str(new_pos))
            motor.move_to(int(new_pos))
            time.sleep(dt)

            print("Stopping")
            motor.stop()

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        ax1.plot(times, signals, color="#2C7BB6", linewidth=1.5)
        ax1.axhline(set_point, color="#D7191C", linewidth=1, linestyle="--", label="Set point")
        ax1.set_ylabel("Signal (a.u.)", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
        ax1.set_axisbelow(True)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.plot(times, errors, color="#F07A13", linewidth=1.5)
        ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Error (a.u.)", fontsize=11)
        ax2.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
        ax2.set_axisbelow(True)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax3.plot(times, positions, color="#1A9641", linewidth=1.5)
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Position (steps)", fontsize=11)
        ax3.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
        ax3.set_axisbelow(True)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        fig.suptitle("PID Response", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.savefig(f"PID_storage/pid_response_{timestamp}.png", dpi=150, bbox_inches="tight")
        # --------------------------------------------------
        # Return home
        # --------------------------------------------------
        return_home_and_close(motor, cam)

    cam.StopGrabbing()
    cam.Close()

    print("Done.")
    return times, signals, positions, errors

coeff_values = [0.1,0.5] #np.arange(0.1, 0.3, 0.1)

COMB_DIR = "combined_data_rq1"


def rq1_collect_statistics(n_repeats=3):

    os.makedirs(COMB_DIR, exist_ok=True)

    stats = {
        m: {
            "coeff": [],
            "mean": [],
            "stdev": [],
            "mean_err": [],
            "stdev_err": []
        }
        for m in ["P", "I", "D"]
    }

    all_traces = {
        m: {}
        for m in ["P", "I", "D"]
    }

    for mode in ["P", "I", "D"]:

        for value in coeff_values:

            signal_runs = []
            time_axis = None

            for repeat in range(n_repeats):

                kp, ki, kd = 0, 0, 0

                if mode == "P":
                    kp = value
                elif mode == "I":
                    ki = value
                elif mode == "D":
                    kd = value

                print(f"Running {mode}={value:.2f}, repeat {repeat}")

                times, signals, positions, errors = pid(kp, kd, ki)

                signal_runs.append(signals)

                if time_axis is None:
                    time_axis = np.array(times)

            signal_runs = np.array(signal_runs)

            mean_signal = np.mean(signal_runs, axis=0)
            std_signal = np.std(signal_runs, axis=0, ddof=1)

            N = len(mean_signal)

            mean_val = np.mean(mean_signal)
            sigma_val = np.std(mean_signal, ddof=1)

            sigma_mean = (1 / N) * np.sqrt(np.sum(std_signal ** 2))
            sigma_stdev = sigma_val / np.sqrt(2 * (N - 1))

            stats[mode]["coeff"].append(value)
            stats[mode]["mean"].append(mean_val)
            stats[mode]["stdev"].append(sigma_val)
            stats[mode]["mean_err"].append(sigma_mean)
            stats[mode]["stdev_err"].append(sigma_stdev)

            # Store traces for plotting
            all_traces[mode][value] = {
                "time": time_axis,
                "mean_signal": mean_signal,
                "std_signal": std_signal
            }
            # Save final averaged result
            pd.DataFrame({
                "time": time_axis,
                "signal": mean_signal,
                "stdev": std_signal
            }).to_csv(
                f"{COMB_DIR}/{timestamp}_{mode}_{value:.2f}.csv",
                index=False
            )

    return stats, all_traces

def rq1_plot_signals(all_traces):

    for mode, label in zip(["P", "I", "D"], ["Kp", "Ki", "Kd"]):
        plt.figure(figsize=(8, 5))

        for value, trace in all_traces[mode].items():

            plt.plot(
                trace["time"],
                trace["mean_signal"],
                label=f"{label}={value:.2f}"
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title(f"Signal vs Time ({label} sweep)")
        plt.grid()
        plt.legend()

        plt.tight_layout()

        plt.savefig(f"rq1_signal_vs_time_{mode}_{timestamp}.pdf")

        plt.show()

def rq1_plot_statistics(stats):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for mode, label in zip(["P", "I", "D"], ["Kp", "Ki", "Kd"]):
        axes[0].axhline(set_point, color="black", linewidth=1, linestyle="--", label="Set point")
        axes[0].errorbar(
            stats[mode]["coeff"],
            stats[mode]["mean"],
            yerr=stats[mode]["mean_err"],
            label=label,
            capsize=3
        )

        axes[1].errorbar(
            stats[mode]["coeff"],
            np.array(stats[mode]["stdev"])/np.array(stats[mode]["mean"]),
            yerr=stats[mode]["stdev_err"],
            label=label,
            capsize=3
        )

    axes[0].set_title("Mean Signal vs PID Coefficient")
    axes[1].set_title("Relative Signal Std Dev vs PID Coefficient")

    axes[0].set_ylabel("Mean signal")
    axes[1].set_ylabel("Relative signal standard deviation")

    for ax in axes:
        ax.set_xlabel("Coefficient value")
        ax.grid()
        ax.legend()

    plt.tight_layout()

    plt.savefig(f"rq1_statistics_{timestamp}.pdf")

    plt.show()

def run_rq1():

    stats, all_traces = rq1_collect_statistics(n_repeats=2)

    rq1_plot_statistics(stats)

    rq1_plot_signals(all_traces)

run_rq1()