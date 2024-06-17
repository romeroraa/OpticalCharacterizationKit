from flask import Blueprint, request, render_template, current_app
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import io
import base64

exp1_flat_field_blueprint = Blueprint('exp1_flat_field', __name__, template_folder='templates')

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
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_3d_channel(data, channel_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    data_smoothed = gaussian_filter(data, sigma=1)
    ax.plot_surface(x, y, data_smoothed, cmap='viridis', edgecolor='none')
    ax.set_title(f'{channel_name} Channel', pad=20)
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Intensity', labelpad=10)
    ax.set_zlim(np.min(data_smoothed), np.max(data_smoothed))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def process_folder(folder_path):
    images = load_images_from_folder(folder_path)
    if not images:
        return None, None

    average_image = compute_average_image(images)
    average_image = average_image / 64

    channels = ['Red', 'Green', 'Blue']
    plots_2d = []
    plots_3d = []

    for i, channel in enumerate(channels):
        plots_2d.append(plot_2d_image(average_image[:, :, i], channel))
        plots_3d.append(plot_3d_channel(average_image[:, :, i], channel))

    return plots_2d, plots_3d

@exp1_flat_field_blueprint.route('/', methods=['GET', 'POST'])
def experiment1_flat_field():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        if not os.path.exists(folder_path):
            return render_template('exp1_flat_field.html', error="Folder does not exist.", folder_path=None)
        
        plots_2d, plots_3d = process_folder(folder_path)
        
        if plots_2d is None or plots_3d is None:
            return render_template('exp1_flat_field.html', error="No .tiff images found in the folder.", folder_path=None)

        return render_template('exp1_flat_field.html', folder_path=folder_path, plots_2d=plots_2d, plots_3d=plots_3d)
    
    return render_template('exp1_flat_field.html', folder_path=None)
