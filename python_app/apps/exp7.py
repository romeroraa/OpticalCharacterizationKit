from flask import Blueprint, render_template, request
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

exp7_blueprint = Blueprint(
    "exp7", __name__, template_folder="templates", static_folder="static"
)

plt.switch_backend("Agg")


def exp_func(x, A, Fc, Ao):
    return A * (1 - np.exp(-x / Fc)) + Ao


def process_images(folder_path, center_size=200):
    im_names = [f for f in os.listdir(folder_path) if f.endswith(".tiff")]
    im_names.sort()
    num_images = len(im_names)
    channel_names = ["R", "G", "B"]
    channel_colors = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}

    average_values = {channel: [] for channel in channel_names}

    for I in range(num_images):
        file_path = os.path.join(folder_path, im_names[I])
        image = (
            tiff.imread(file_path).astype(np.float32) / 2**6
        )  # Divide by 2**6 to account for 10-bit encoding

        # Get the center 200x200 pixels
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        center_region = image[
            center_y - center_size // 2 : center_y + center_size // 2,
            center_x - center_size // 2 : center_x + center_size // 2,
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
        channel: np.array(average_values[channel])
        / normalization_means[channel]
        for channel in channel_names
    }

    return normalized_values, channel_names, channel_colors, num_images


@exp7_blueprint.route("/", methods=["GET", "POST"])
def exp7():
    if request.method == "POST":
        folder_path = request.form.get("folder_path")
        if not folder_path:
            return "No folder path provided"
        if not os.path.exists(folder_path):
            return "The provided folder path does not exist"

        (
            normalized_values,
            channel_names,
            channel_colors,
            num_images,
        ) = process_images(folder_path)

        frames = np.arange(num_images)
        plot_paths = []

        # Combined plot for all channels
        plt.figure(figsize=(18, 10))
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
        plt.title("Bright Image Warm Up - All Channels")
        plt.grid(True)
        combined_plot_path = os.path.join(
            exp7_blueprint.static_folder, "combined_channel_plot.png"
        )
        plt.savefig(combined_plot_path)
        plot_paths.append("combined_channel_plot.png")
        plt.close()

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
            plt.title(f"Bright Image Warm Up - {channel} Channel")
            plt.grid(True)
            plot_path = os.path.join(
                exp7_blueprint.static_folder, f"{channel}_channel_plot.png"
            )
            plt.savefig(plot_path)
            plot_paths.append(f"{channel}_channel_plot.png")
            plt.close()

        return render_template("exp7/exp7.html", plots=plot_paths)
    return render_template("exp7/exp7.html", plots=None)
