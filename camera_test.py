"""
Sweep motor using ROTATE only (no move_to), stop every 1 mm, capture image.

Because move_to is unreliable on your setup, this version uses:
- rotate(speed)
- monitor actual_position
- stop()

It rotates forward through the full travel, stopping every 1 mm.

EDIT THESE SETTINGS:
- COM_PORT
- AXIS_INDEX
- FULL_RANGE_MM
- STEPS_PER_MM   <-- critical calibration value
"""

import os
import time
import platform
from pypylon import pylon
import pytrinamic
from pytrinamic.connections import ConnectionManager
from pytrinamic.modules import TMCM6110

# ==================================================
# USER SETTINGS
# ==================================================
COM_PORT = "COM6"
AXIS_INDEX = 1

FULL_RANGE_MM = 100
STEP_MM = 1

STEPS_PER_MM = 2560      # must match your mechanics
ROTATE_SPEED = 1500      # tune as needed

SAVE_FOLDER = "scan_images"
SETTLE_TIME = 0.4
GRAB_TIMEOUT = 3000

# ==================================================
# HELPERS
# ==================================================
def mm_to_steps(mm):
    return int(mm * STEPS_PER_MM)


def save_image(camera, filename):
    with camera.RetrieveResult(GRAB_TIMEOUT) as result:
        if not result.GrabSucceeded():
            raise RuntimeError("Camera grab failed")

        img = pylon.PylonImage()
        img.AttachGrabResultBuffer(result)
        img.Save(pylon.ImageFileFormat_Png, filename)
        img.Release()


def rotate_until_position(motor, target_steps, speed):
    """
    Rotate motor until encoder/actual_position reaches target_steps.
    Uses only rotate() + stop()
    """
    current = motor.actual_position

    if target_steps > current:
        motor.rotate(abs(speed))
        while motor.actual_position < target_steps:
            time.sleep(0.01)

    else:
        motor.rotate(-abs(speed))
        while motor.actual_position > target_steps:
            time.sleep(0.01)

    motor.stop()


# ==================================================
# MAIN
# ==================================================
def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # ---------------- Camera ----------------
    tlf = pylon.TlFactory.GetInstance()
    cam = pylon.InstantCamera(tlf.CreateFirstDevice())
    cam.Open()
    cam.StartGrabbing()

    # ---------------- Motor -----------------
    pytrinamic.show_info()

    connection_manager = ConnectionManager(
        f"--interface usb_tmcl --port {COM_PORT}"
    )

    with connection_manager.connect() as interface:
        module = TMCM6110(interface)
        motor = module.motors[AXIS_INDEX]

        # Drive settings
        motor.drive_settings.max_current = 200
        motor.drive_settings.standby_current = 0
        motor.drive_settings.boost_current = 0
        motor.drive_settings.microstep_resolution = (
            motor.ENUM.microstep_resolution_256_microsteps
        )

        motor.max_acceleration = 1000
        motor.max_velocity = 1000

        # Zero current location
        motor.actual_position = 0

        total_images = int(FULL_RANGE_MM / STEP_MM) + 1

        print("Starting forward sweep...")

        for i in range(total_images):
            pos_mm = i * STEP_MM
            target_steps = mm_to_steps(pos_mm)

            print(f"Moving to {pos_mm:.1f} mm")
            rotate_until_position(motor, target_steps, ROTATE_SPEED)

            time.sleep(SETTLE_TIME)

            filename = os.path.join(
                SAVE_FOLDER,
                f"img_{i:04d}_{pos_mm:.1f}mm.png"
            )

            print("Capturing", filename)
            save_image(cam, filename)

        print("Sweep complete.")

        # Optional return home
        print("Returning to zero...")
        rotate_until_position(motor, 0, ROTATE_SPEED)

    cam.StopGrabbing()
    cam.Close()

    print("Done.")


if __name__ == "__main__":
    main()