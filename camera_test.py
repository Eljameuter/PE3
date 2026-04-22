"""This sample shows how grabbed images can be saved using pypylon only (no
need to use openCV).

Available image formats are     (depending on platform):
 - pylon.ImageFileFormat_Bmp    (Windows)
 - pylon.ImageFileFormat_Tiff   (Linux, Windows)
 - pylon.ImageFileFormat_Jpeg   (Windows)
 - pylon.ImageFileFormat_Png    (Linux, Windows)
 - pylon.ImageFileFormat_Raw    (Windows)
"""
"""
Basler / pypylon Airy Disk Search Script
---------------------------------------
This script sweeps through several camera settings that strongly affect
airy disk visibility:

- Exposure time
- Gain
- Pixel format (if supported)
- Binning (if supported)

Instead of saving images, it displays them in a matplotlib grid so you can
visually compare settings and identify the sharpest Airy disk pattern.

Requirements:
pip install pypylon matplotlib numpy
"""

from pypylon import pylon
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------

# Try combinations of these settings
EXPOSURES_US = [100, 300, 1000, 3000, 10000]
GAINS_DB = [0, 6, 12]

# Optional if camera supports it
BINNINGS = [1, 2]

# Number of images to display
MAX_PLOTS = 12

# -------------------------------------------------
# CAMERA SETUP
# -------------------------------------------------

tlf = pylon.TlFactory.GetInstance()
cam = pylon.InstantCamera(tlf.CreateFirstDevice())

cam.Open()

# Continuous acquisition mode
cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def try_set(node, value):
    try:
        node.SetValue(value)
        return True
    except Exception:
        return False

def get_image():
    result = cam.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)

    if result.GrabSucceeded():
        img = converter.Convert(result)
        arr = img.GetArray()
        result.Release()
        return arr
    else:
        result.Release()
        return None

# -------------------------------------------------
# BUILD TEST LIST
# -------------------------------------------------

tests = list(itertools.product(EXPOSURES_US, GAINS_DB, BINNINGS))
tests = tests[:MAX_PLOTS]

n = len(tests)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
axes = np.array(axes).reshape(-1)

# -------------------------------------------------
# RUN TESTS
# -------------------------------------------------

for idx, (exp, gain, binning) in enumerate(tests):

    # Exposure
    try_set(cam.ExposureTime, exp)

    # Gain
    if hasattr(cam, "Gain"):
        try_set(cam.Gain, gain)

    # Binning if available
    if hasattr(cam, "BinningHorizontal"):
        try_set(cam.BinningHorizontal, binning)

    if hasattr(cam, "BinningVertical"):
        try_set(cam.BinningVertical, binning)

    img = get_image()

    ax = axes[idx]

    if img is not None:
        ax.imshow(img, cmap="gray")
    else:
        ax.text(0.5, 0.5, "Grab Failed", ha="center", va="center")

    ax.set_title(
        f"Exp={exp} us\nGain={gain} dB\nBin={binning}"
    )
    ax.axis("off")

# Hide unused axes
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------
# CLEANUP
# -------------------------------------------------

cam.StopGrabbing()
cam.Close()