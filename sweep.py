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

import os
import time
import platform
from pypylon import pylon
import pytrinamic
from pytrinamic.connections import ConnectionManager
from pytrinamic.modules import TMCM6110

# ==========================================================
# USER SETTINGS
# ==========================================================
COM_PORT = "COM6"
AXIS_INDEX = 1                 # motor index used in your example
FULL_RANGE_MM = 20           # total travel range of stage in mm
STEP_MM = 1                  # move in 1 mm increments
STEPS_PER_MM = 1/5e-7

SETTLE_TIME = 0.05             # seconds after move before image capture
GRAB_TIMEOUT = 3000           # ms
SAVE_FOLDER = "scan_images"
# ==========================================================
# HELPERS
# ==========================================================
def mm_to_steps(mm):
    return int(mm * STEPS_PER_MM)


def wait_until_position_reached(motor):
    while not motor.get_position_reached():
        time.sleep(0.05)



def save_image(camera, filename):
    # Apply best-found settings (first tested settings)
    camera.ExposureTime.SetValue(100)   # microseconds
    camera.Gain.SetValue(0)            # dB

    with camera.RetrieveResult(GRAB_TIMEOUT) as result:
        if result.GrabSucceeded():
            img = pylon.PylonImage()
            img.AttachGrabResultBuffer(result)

            ipo = pylon.ImagePersistenceOptions()
            ipo.SetQuality(90)

            img.Save(pylon.ImageFileFormat_Png, filename)

            img.Release()
        else:
            raise RuntimeError("Image grab failed")



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
    cam.StartGrabbing()

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
        # Scan through full range
        # --------------------------------------------------
        total_steps = int(FULL_RANGE_MM / STEP_MM) + 1

        print(f"Starting scan: {FULL_RANGE_MM} mm range")
        print(f"{total_steps} images will be captured.")

        for i in range(total_steps):
            position_mm = i * STEP_MM
            target_steps = mm_to_steps(position_mm)

            print(f"Moving to {position_mm:.1f} mm")
            motor.move_to(target_steps)

            wait_until_position_reached(motor)
            time.sleep(SETTLE_TIME)

            filename = os.path.join(
                SAVE_FOLDER,
                f"img_{i:04d}_{position_mm:.1f}mm.png"
            )

            print(f"Capturing {filename}")
            save_image(cam, filename)

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