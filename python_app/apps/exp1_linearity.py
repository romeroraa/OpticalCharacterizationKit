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

exp1_linearity_blueprint = Blueprint('exp1_linearity', __name__, template_folder='templates')

def load_tiff_images_and_average(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".tiff", ".tif")) and filename != "average_image.tiff":
            file_path = os.path.join(folder_path, filename)
            img = tifffile.imread(file_path)
            images.append(img)

    if not images:
        return None

    average_image = np.mean(images, axis=0).astype(images[0].dtype)
    return images, average_image

def calculate_statistics(channel_data, exposure_times):
    fit, cov = np.polyfit(exposure_times, channel_data, 1, cov=True)
    slope, intercept = fit
    residuals = channel_data - np.poly1d(fit)(exposure_times)
    rmse = np.sqrt(np.mean(residuals**2))
    slope_error = np.sqrt(cov[0, 0])
    intercept_error = np.sqrt(cov[1, 1])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((channel_data - np.mean(channel_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return slope, slope_error, intercept, intercept_error, rmse, r_squared

def plot_and_analyze(channel_data, channel_name, exposure_times):
    fit = np.polyfit(exposure_times, channel_data, 1)
    pred = np.poly1d(fit)
    predicted = pred(np.linspace(min(exposure_times), max(exposure_times), 100))
    residuals = channel_data - pred(exposure_times)
    r_squared = 1 - np.sum(residuals**2) / np.sum((channel_data - np.mean(channel_data)) ** 2)
    plt.plot(np.linspace(min(exposure_times), max(exposure_times), 100), predicted, "--", color=f"tab:{channel_name.lower()}")
    plt.scatter(exposure_times, channel_data, label=f"$R^2={r_squared:.3f}$", color=f"tab:{channel_name.lower()}")
    plt.xlim(-0.01, 1.01)
    return residuals

def display_linearity_table(statistics):
    data = {
        "Channel": ["Slope (m)", "Uncertainty in Slope (%)", "Y-intercept (b)", "Uncertainty in Y-intercept", "RMSE", "R^2"],
        "R": [f"{statistics['R']['slope']:.2f}", f"{statistics['R']['slope_error']:.2f}%", f"{statistics['R']['intercept']:.2f}", f"{statistics['R']['intercept_error']:.2f}", f"{statistics['R']['rmse']:.2f}", f"{statistics['R']['r_squared']:.4f}"],
        "G": [f"{statistics['G']['slope']:.2f}", f"{statistics['G']['slope_error']:.2f}%", f"{statistics['G']['intercept']:.2f}", f"{statistics['G']['intercept_error']:.2f}", f"{statistics['G']['rmse']:.2f}", f"{statistics['G']['r_squared']:.4f}"],
        "B": [f"{statistics['B']['slope']:.2f}", f"{statistics['B']['slope_error']:.2f}%", f"{statistics['B']['intercept']:.2f}", f"{statistics['B']['intercept_error']:.2f}", f"{statistics['B']['rmse']:.2f}", f"{statistics['B']['r_squared']:.4f}"],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center", colColours=["gray", "tab:red", "tab:green", "tab:blue"])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def process_linearity_folder(base_folder_path):
    r_average_image, g_average_image, b_average_image = {}, {}, {}

    for folder_name in os.listdir(base_folder_path):
        if folder_name != ".DS_Store":
            path = os.path.join(base_folder_path, folder_name)
            _, average_image = load_tiff_images_and_average(path)
            if average_image is not None:
                exposure_time = int(folder_name) / 1000000
                r_average_image[exposure_time] = np.mean(average_image[:, :, 0] / 64)
                g_average_image[exposure_time] = np.mean(average_image[:, :, 1] / 64)
                b_average_image[exposure_time] = np.mean(average_image[:, :, 2] / 64)

    r = list(r_average_image.values())
    g = list(g_average_image.values())
    b = list(b_average_image.values())
    et = list(r_average_image.keys())

    statistics = {"R": {}, "G": {}, "B": {}}
    for color, channel in zip(["R", "G", "B"], [r, g, b]):
        slope, slope_error, intercept, intercept_error, rmse, r_squared = calculate_statistics(channel, et)
        statistics[color] = {
            "slope": slope,
            "slope_error": (slope_error / slope) * 100,
            "intercept": intercept,
            "intercept_error": intercept_error,
            "rmse": rmse,
            "r_squared": r_squared,
        }

    statistics_table = display_linearity_table(statistics)

    plt.figure(figsize=(10, 6))
    r_residuals = plot_and_analyze(r, "Red", et)
    g_residuals = plot_and_analyze(g, "Green", et)
    b_residuals = plot_and_analyze(b, "Blue", et)
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Average Pixel Value")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    linearity_plot = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.figure(figsize=(10, 6))
    plt.scatter(et, r_residuals, color="tab:red", label="R residuals")
    plt.scatter(et, g_residuals, color="tab:green", label="G residuals")
    plt.scatter(et, b_residuals, color="tab:blue", label="B residuals")
    plt.axhline(y=0, color="gray", linestyle="--")
    plt.title("Residuals of RGB Channels vs. Exposure Time")
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Residuals")
    plt.legend(loc="lower left")
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    residuals_plot = base64.b64encode(buf.getvalue()).decode("utf-8")

    return statistics_table, linearity_plot, residuals_plot

@exp1_linearity_blueprint.route('/', methods=['GET', 'POST'])
def experiment1_linearity():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        if not os.path.exists(folder_path):
            return render_template('exp1_linearity.html', error="Folder does not exist.", folder_path=None)
        
        statistics_table, linearity_plot, residuals_plot = process_linearity_folder(folder_path)
        
        return render_template('exp1_linearity.html', folder_path=folder_path, statistics_table=statistics_table, linearity_plot=linearity_plot, residuals_plot=residuals_plot)
    
    return render_template('exp1_linearity.html', folder_path=None)
