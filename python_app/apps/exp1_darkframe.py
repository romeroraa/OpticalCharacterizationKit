from flask import Blueprint, request, render_template, current_app, send_file
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

sns.set_theme(style="whitegrid", font_scale=1.5)

exp1_darkframe_blueprint = Blueprint(
    "exp1_darkframe", __name__, template_folder="templates"
)


def load_tiff_images_and_average(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if (
            filename.endswith((".tiff", ".tif"))
            and filename != "average_image.tiff"
        ):
            file_path = os.path.join(folder_path, filename)
            img = tifffile.imread(file_path)
            images.append(img)

    if not images:
        return None, None

    average_image = np.mean(images, axis=0).astype(images[0].dtype)
    return images, average_image


def compute_statistics(image_channel_data):
    min_val = np.min(image_channel_data)
    median_val = np.median(image_channel_data)
    max_val = np.max(image_channel_data)
    mean_val = np.mean(image_channel_data)
    std_val = np.std(image_channel_data)
    return min_val, median_val, max_val, mean_val, std_val


def display_statistics_table(statistics):
    data = {
        "Channel": ["Min DN", "Median DN", "Max DN", "Mean DN", "SD DN"],
        "R": [
            f"{statistics['R']['min']:.2f}",
            f"{statistics['R']['median']:.2f}",
            f"{statistics['R']['max']:.2f}",
            f"{statistics['R']['mean']:.2f}",
            f"{statistics['R']['std']:.2f}",
        ],
        "G": [
            f"{statistics['G']['min']:.2f}",
            f"{statistics['G']['median']:.2f}",
            f"{statistics['G']['max']:.2f}",
            f"{statistics['G']['mean']:.2f}",
            f"{statistics['G']['std']:.2f}",
        ],
        "B": [
            f"{statistics['B']['min']:.2f}",
            f"{statistics['B']['median']:.2f}",
            f"{statistics['B']['max']:.2f}",
            f"{statistics['B']['mean']:.2f}",
            f"{statistics['B']['std']:.2f}",
        ],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["gray", "tab:red", "tab:green", "tab:blue"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_histogram(images):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["Red", "Green", "Blue"]
    subset_size = 100000
    bins = np.arange(0, 15, 1)

    for i, color in enumerate(colors):
        data = np.array([x[:, :, i] for x in images]).flatten() / 64
        if len(data) > subset_size:
            data = np.random.choice(data, subset_size, replace=False)
        sns.histplot(
            data,
            bins=bins,
            color=f"tab:{color.lower()}",
            ax=ax,
            label=color,
            alpha=0.5,
        )

    plt.xlabel("Digital Number")
    plt.ylabel("Count")
    plt.xlim(0, 14)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_fixed_pattern_noise(average_image):
    channels = ["Red", "Green", "Blue"]
    global_min = np.min(average_image)
    global_max = np.percentile(average_image, 99)
    images_base64 = []

    for i in range(3):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.imshow(
            average_image[:, :, i],
            cmap="gray",
            vmin=global_min,
            vmax=global_max,
        )
        plt.colorbar()
        plt.title(f"{channels[i]} Channel")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        images_base64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return images_base64


@exp1_darkframe_blueprint.route("/", methods=["GET", "POST"])
def experiment1():
    if request.method == "POST":
        folder_path = request.form["folder_path"]
        if not os.path.exists(folder_path):
            return render_template(
                "exp1_darkframe.html",
                error="Folder does not exist.",
                folder_path=None,
            )

        images, average_image = load_tiff_images_and_average(folder_path)
        if images is None:
            return render_template(
                "exp1_darkframe.html",
                error="No .tiff images found in the folder.",
                folder_path=None,
            )

        average_image = average_image / 64
        statistics = {
            "R": {},
            "G": {},
            "B": {},
        }

        for i, color in enumerate(["R", "G", "B"]):
            channel_data = average_image[:, :, i]
            (
                statistics[color]["min"],
                statistics[color]["median"],
                statistics[color]["max"],
                statistics[color]["mean"],
                statistics[color]["std"],
            ) = compute_statistics(channel_data)

        statistics_table = display_statistics_table(statistics)
        histogram_image = plot_histogram(images)
        fpn_images = plot_fixed_pattern_noise(average_image)

        return render_template(
            "exp1_darkframe.html",
            folder_path=folder_path,
            statistics_table=statistics_table,
            histogram_image=histogram_image,
            fpn_images=fpn_images,
        )

    return render_template("exp1_darkframe.html", folder_path=None)
