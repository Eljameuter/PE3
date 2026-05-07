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
set_point = 619013.0 # signal
K_P = 0.2
K_I = 0 
K_D = 0 
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
def main():
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
            print(elapsed_seconds)
            # determine signal!
            signal, pos = sum_around_brightest(filename)
            signals.append(signal)
            print(signal)
            # determine control function
            error = set_point - signal 
            errors.append(error)
            
            P = K_P * error
            I = K_I * np.trapezoid(errors, dx=dt) if i > 1 else 0
            D = K_D * (errors[i] - errors[i - 1]) / dt if i > 0 else 0
            control = P + I + D
            control = np.clip(control, -STEPS_PER_MM, STEPS_PER_MM)
            print(control)
            new_pos = int(float(motor.actual_position)+control)
            new_pos = np.clip(new_pos, 0,10*STEPS_PER_MM)
            positions.append(new_pos)
            print(new_pos)
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(f"PID_storage/pid_response_{timestamp}.png", dpi=150, bbox_inches="tight")
        plt.show()
        # --------------------------------------------------
        # Return home
        # --------------------------------------------------
        return_home_and_close(motor, cam)

    cam.StopGrabbing()
    cam.Close()

    print("Done.")

if __name__ == "__main__":
    main()
