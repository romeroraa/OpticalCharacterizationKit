# %%
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.5)


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
        print("No .tiff images found in the folder:", folder_path)
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
    plt.show()


def plot_histogram(images):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["Red", "Green", "Blue"]
    subset_size = 100000  # Use a subset of the data to speed up plotting
    bins = np.arange(0, 15, 1)  # Set bin edges from 0 to 14

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
    plt.xlim(0, 14)  # Set x-axis limit to 14
    plt.legend()
    plt.show()


def plot_fixed_pattern_noise(average_image):
    channels = ["Red", "Green", "Blue"]
    global_min = np.min(average_image)
    global_max = np.percentile(average_image, 99)

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
        plt.show()


# Main function to load images, compute the average image, display results and plot fixed pattern noise
def process_folder(folder_path):
    images, average_image = load_tiff_images_and_average(folder_path)
    if images is None:
        return

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

    display_statistics_table(statistics)
    plot_histogram(images)
    plot_fixed_pattern_noise(average_image)


# Example usage
folder_path = "/Users/raaromero/p301/output/df2/50ms"
process_folder(folder_path)

# %%
