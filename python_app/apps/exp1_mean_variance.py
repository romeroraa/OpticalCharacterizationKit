from flask import Blueprint, request, render_template, current_app
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

sns.set_theme(style="whitegrid", font_scale=1.5)

exp1_mean_variance_blueprint = Blueprint('exp1_mean_variance', __name__, template_folder='templates')

def load_tiff_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".tiff", ".tif")) and filename != "average_image.tiff":
            file_path = os.path.join(folder_path, filename)
            img = tifffile.imread(file_path)
            images.append(img)

    if not images:
        print("No .tiff images found in the folder:", folder_path)
        return None

    return images

def sample_points_and_calculate_stats(images, num_points=1000):
    sampled_means = {"R": [], "G": [], "B": [], "Gray": []}
    sampled_variances = {"R": [], "G": [], "B": [], "Gray": []}

    for img in images:
        height, width, _ = img.shape
        points = np.random.choice(height * width, num_points, replace=False)
        sampled_pixels = img.reshape(-1, 3)[points]

        for channel, color in zip(["R", "G", "B"], range(3)):
            sampled_means[channel].append(np.mean(sampled_pixels[:, color]))
            sampled_variances[channel].append(np.var(sampled_pixels[:, color]))

        gray_pixels = np.mean(sampled_pixels, axis=1)
        sampled_means["Gray"].append(np.mean(gray_pixels))
        sampled_variances["Gray"].append(np.var(gray_pixels))

    return sampled_means, sampled_variances

def plot_mean_vs_variance(means, variances, channel, color, exposure_times):
    plt.figure(figsize=(10, 6))
    plt.scatter(means[channel], variances[channel], label=f"{channel} Channel", color=color)

    fit = np.polyfit(means[channel], variances[channel], 1)
    pred = np.poly1d(fit)
    plt.plot(means[channel], pred(means[channel]), "--", color="orange", label=f"y = {fit[0]:.2f}x + {fit[1]:.2f}")

    plt.xlabel("Mean Digital Number (DN)")
    plt.ylabel("Variance")
    plt.title(f"Mean vs Variance for {channel} Channel")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8'), fit

def process_linearity_folder_with_sampling(base_folder_path, num_points=1000):
    means = {"R": [], "G": [], "B": [], "Gray": []}
    variances = {"R": [], "G": [], "B": [], "Gray": []}
    exposure_times = []

    for folder_name in os.listdir(base_folder_path):
        if folder_name != ".DS_Store":
            path = os.path.join(base_folder_path, folder_name)
            images = load_tiff_images(path)
            if images is not None:
                exposure_time = float(folder_name)
                exposure_times.append(exposure_time)

                sampled_means, sampled_variances = sample_points_and_calculate_stats(images, num_points)

                for channel in means.keys():
                    means[channel].extend(sampled_means[channel])
                    variances[channel].extend(sampled_variances[channel])

    plots = {}
    fits = {}
    for channel, color in zip(means.keys(), ['tab:red', 'tab:green', 'tab:blue', 'gray']):
        plot, fit = plot_mean_vs_variance(means, variances, channel, color, exposure_times)
        plots[channel] = plot
        fits[channel] = fit

    return plots, fits

@exp1_mean_variance_blueprint.route('/', methods=['GET', 'POST'])
def experiment1_mean_variance():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        if not os.path.exists(folder_path):
            return render_template('exp1_mean_variance.html', error="Folder does not exist.", folder_path=None)
        
        plots, fits = process_linearity_folder_with_sampling(folder_path)
        
        return render_template('exp1_mean_variance.html', folder_path=folder_path, plots=plots, fits=fits)
    
    return render_template('exp1_mean_variance.html', folder_path=None)