# INPUTS
# FOLDER
FOLDER = "/Users/raaromero/p301/output/"
CENTER_SIZE = 200
# %%
import tifffile as tiff
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exp_func(x, A, Fc, Ao):
    return A * (1 - np.exp(-x / Fc)) + Ao


# Load the images
im_names = [f for f in os.listdir(FOLDER) if f.endswith(".tiff")]
# Sort imnames by filename
im_names.sort()
num_images = len(im_names)
channel_names = ["R", "G", "B"]
channel_colors = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}

average_values = {channel: [] for channel in channel_names}

# Process each image
for i in range(num_images):
    file_path = os.path.join(FOLDER, im_names[i])
    image = (
        tiff.imread(file_path).astype(np.float32) / 2**6
    )  # Divide by 2**6 to account for 10-bit encoding

    # Get the center 200x200 pixels
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    center_region = image[
        center_y - CENTER_SIZE // 2 : center_y + CENTER_SIZE // 2,
        center_x - CENTER_SIZE // 2 : center_x + CENTER_SIZE // 2,
    ]

    # Compute the average pixel value for each channel
    average_values["R"].append(np.mean(center_region[:, :, 0]))
    average_values["G"].append(np.mean(center_region[:, :, 1]))
    average_values["B"].append(np.mean(center_region[:, :, 2]))

# Compute the mean of the last 100 images for normalization
normalization_means = {
    channel: np.mean(average_values[channel][-100:])
    for channel in channel_names
}
# Normalize the average values
normalized_values = {
    channel: np.array(average_values[channel]) / normalization_means[channel]
    for channel in channel_names
}

frames = np.arange(num_images)
for channel in channel_names:
    plt.figure(figsize=(14, 8))
    plt.plot(
        frames,
        normalized_values[channel],
        "o",
        label=f"{channel} data",
        color=channel_colors[channel],
    )
    popt, _ = curve_fit(
        exp_func, frames, normalized_values[channel], p0=[1, 100, 0]
    )
    plt.plot(
        frames,
        exp_func(frames, *popt),
        "-",
        label=f"{channel} fit: A={popt[0]:.2f}, Fc={popt[1]:.2f}, Ao={popt[2]:.2f}",
        color=channel_colors[channel],
    )
    plt.xlabel("Frame")
    plt.ylabel("Normalized Average Pixel Value")
    plt.legend()
    plt.title(f"Camera Bright Frame Exposure Stability - {channel} Channel")
    plt.grid(True)
    plt.show()

# Combined plot for all channels
plt.figure(figsize=(14, 8))
for channel in channel_names:
    plt.plot(
        frames,
        normalized_values[channel],
        "o",
        label=f"{channel} data",
        color=channel_colors[channel],
    )
    popt, _ = curve_fit(
        exp_func, frames, normalized_values[channel], p0=[1, 100, 0]
    )
    plt.plot(
        frames,
        exp_func(frames, *popt),
        "-",
        label=f"{channel} fit: A={popt[0]:.2f}, Fc={popt[1]:.2f}, Ao={popt[2]:.2f}",
        color=channel_colors[channel],
    )

plt.xlabel("Frame")
plt.ylabel("Normalized Average Pixel Value")
plt.legend()
plt.title("Camera Bright Frame Exposure Stability - All Channels")
plt.grid(True)
plt.show()

# %%
