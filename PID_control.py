"""
Created on Thu May  7 12:51:11 2026

@author: Elja en Luna

"""

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
# 1. capture an image of the interference pattern and determine its diameter (brightest pixel + FWHM)
# 2. determine the error e(t) = set point - FWHM and the control signal u(t) 
# unit control signal: steps per second 
# 3. send the control signal to the motor; we measure the time; the motor moves with a speed u(t)
# during a time interval dt; we capture an image and measure the time every x steps and determine 
# the corresponding FWHM; we save the diameters and times in arrays
# 4. after the time interval dt, determine the error again; we measure until we have reached t_0 
# ================================================================================================

# ==========================================================
# PID SETTINGS
# ==========================================================
set_point = 30 # pixels
K_P = 0 
K_I = 0 
K_D = 0 
T = 10 # seconds
dt = 0.5 # seconds

import os
import time
import platform
from pypylon import pylon
import pytrinamic
from pytrinamic.connections import ConnectionManager
from pytrinamic.modules import TMCM6110

from datetime import datetime
import numpy as np
import time

# ==========================================================
# USER SETTINGS
# ==========================================================
COM_PORT = "COM3"
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
        FWHMs = []
        errors = []        
        start_time = datetime.now()

        for i in range(int(T/dt)): 
            # capture and save image
            filename = os.path.join(
                SAVE_FOLDER,
                f"img_{i:04d}_{position_mm:.1f}mm.png"
            )
            print(f"Capturing {filename}")
            save_image(cam, filename)
            
            # save corresponding measurement time 
            now = datetime.now()
            elapsed_time = now - start_time
            elapsed_seconds = elapsed_time.total_seconds()
            times.append(elapsed_seconds)

            # determine FWHM!
            FWHMs.append(FWHM)
            
            # determine control function
            error = set_point - FWHM 
            errors.append(error)
            
            P = K_P * error
            I = K_I * np.trapezoid(errors, dx=dt) if i > 1 else 0
            D = K_D * (errors[i] - errors[i - 1]) / dt if i > 0 else 0
            control = P + I + D
            
            # apply control function 
            print("Rotating")
            motor_0.rotate(1500) # how do we apply the control function...
            time.sleep(dt)

            print("Stopping")
            motor_0.stop()
            time.sleep() # do we need this...

        # --------------------------------------------------
        # Return home
        # --------------------------------------------------
        print("Returning to zero...")
        motor.move_to(0)
        wait_until_position_reached(motor)

    cam.StopGrabbing()
    cam.Close()

    print("Done.")

if __name__ == "__main__":
    main()