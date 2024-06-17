# %%
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".tiff", ".tif")):
            file_path = os.path.join(folder_path, filename)
            img = tifffile.imread(file_path)
            images.append(img)
    return images

def compute_average_image(images):
    return np.mean(images, axis=0).astype(images[0].dtype)

def plot_2d_image(data, channel_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='gray')
    plt.title(f'{channel_name} Channel Intensity')
    plt.colorbar()
    plt.show()

def plot_3d_channel(data, channel_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    # Apply Gaussian smoothing
    data_smoothed = gaussian_filter(data, sigma=1)
    ax.plot_surface(x, y, data_smoothed, cmap='viridis', edgecolor='none')
    ax.set_title(f'{channel_name} Channel', pad=20)
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Intensity', labelpad=10)
    ax.set_zlim(np.min(data_smoothed), np.max(data_smoothed))  # Set z-axis limits
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

def process_folder(folder_path):
    images = load_images_from_folder(folder_path)
    if not images:
        print(f"No .tiff images found in the folder: {folder_path}")
        return

    average_image = compute_average_image(images)
    average_image = average_image / 64  # Convert to 10-bit

    channels = ['Red', 'Green', 'Blue']
    for i, channel in enumerate(channels):
        plot_2d_image(average_image[:, :, i], channel)
        plot_3d_channel(average_image[:, :, i], channel)

# Example usage
folder_path = "/Users/raaromero/p301/output/flat_field_1"
process_folder(folder_path)

# %%
