import pytrinamic
from pytrinamic.connections import ConnectionManager
from pytrinamic.modules import TMCM6110
import time
import numpy as np
pytrinamic.show_info()
connection_manager = ConnectionManager("--interface usb_tmcl --port COM6")  # using USB
# -------------------------
# System & Experiment Parameters
# -------------------------
dt = 1 / 100
record_time = 0.5
time = np.arange(0, record_time, dt)


LED1_voltage = 0             # initial lens position
tau = 0.005                    # LED response time constant

# Disturbance LED: step from 0.4 V → 0.6 V at t = 0.05 s
def disturbance(t):
    return 0.4 if t < 0.05 else 0.6


def measure_run(Kp, Ki, Kd):
    """
    """
    lens = 0
    errors = []
    voltages = []

    for i, t in enumerate(time):
        dist_CCD = disturbance(t)

        # Write both LEDs simultaneously
        daq.write(
            [AO_PID, AO_DIST],
            [[led_voltage], [dist_voltage]],
            sample_rate=1 / dt,
            samples_per_channel=1
        )

        # Read photodiode
        _, data = daq.read(
            sample_rate=1 / dt,
            samples_per_channel=1
        )
        pd_voltage = data[0][0]
        voltages.append(pd_voltage)

        # PID calculation
        error = V_set - pd_voltage
        errors.append(error)

        P = Kp * error
        I = Ki * np.trapezoid(errors, dx=dt) if i > 1 else 0
        D = Kd * (errors[i] - errors[i - 1]) / dt if i > 0 else 0

        control = P + I + D

        # Update LED with system dynamics
        led_voltage += (control - led_voltage) * dt / tau
        led_voltage = np.clip(led_voltage, 0, 4)

    return np.array(voltages)


with connection_manager.connect() as my_interface:
    module = TMCM6110(my_interface)
    lens_motor = module.motors[1]
    CCD_motor = module.motors[2]
    # If you use a different motor_1 be sure you have the right configuration setup otherwise the script may not work.

    print("Preparing parameters")
    # preparing drive settings
    lens_motor.drive_settings.max_current = 200
    lens_motor.drive_settings.standby_current = 0
    lens_motor.drive_settings.boost_current = 0
    lens_motor.drive_settings.microstep_resolution = lens_motor.ENUM.microstep_resolution_256_microsteps
    print(lens_motor.drive_settings)
    lens_motor.drive_settings.max_current = 200
    lens_motor.drive_settings.standby_current = 0
    lens_motor.drive_settings.boost_current = 0
    lens_motor.drive_settings.microstep_resolution = lens_motor.ENUM.microstep_resolution_256_microsteps
    print(lens_motor.drive_settings)

    # preparing linear ramp settings
    lens_motor.max_acceleration = 1000
    lens_motor.max_velocity = 1000
    lens_motor.max_acceleration = 1000
    lens_motor.max_velocity = 1000

    # reset actual position
    lens_motor.actual_position = 0
    lens_motor.actual_position = 0

    # read actual position
    print("ActualPostion = {}".format(lens_motor.actual_position))
    print("ActualPostion = {}".format(lens_motor.actual_position))
    time.sleep(2)

    # doubling moved distance
    print("Doubling moved distance")
    lens_motor.move_by(lens_motor.actual_position)
    lens_motor.move_by(lens_motor.actual_position)


print("\nReady.")
