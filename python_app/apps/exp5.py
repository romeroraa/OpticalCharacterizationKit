from flask import Blueprint, request, render_template, current_app
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff
from scipy.ndimage import laplace
import io
import base64

sns.set_theme(style="white", font_scale=1.5)

exp5_blueprint = Blueprint('exp5', __name__, template_folder='templates')

def variance_of_laplacian(image):
    return np.var(laplace(image))

def get_square(im, size=7, return_max_pixel=False, max_pixel=None):
    if max_pixel is None:
        max_pixel = np.unravel_index(np.argmax(im), im.shape)
        x, y, _ = max_pixel
    else:
        x, y = max_pixel
        x = x + size // 2
        y = y + size // 2
    x1 = x - size
    x2 = x + size
    y1 = y - size
    y2 = y + size
    if return_max_pixel:
        return im[x1:x2, y1:y2], max_pixel
    else:
        return im[x1:x2, y1:y2]

def display_rgb_channels(im, size=4, normalize=True):
    square = get_square(im, size)
    square_R = square.copy()
    square_R[:, :, 1] = 0
    square_R[:, :, 2] = 0

    square_G = square.copy()
    square_G[:, :, 0] = 0
    square_G[:, :, 2] = 0

    square_B = square.copy()
    square_B[:, :, 0] = 0
    square_B[:, :, 1] = 0

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    if normalize:
        ax[0].imshow(square_R / np.max(square_R))
        ax[1].imshow(square_G / np.max(square_G))
        ax[2].imshow(square_B / np.max(square_B))
    else:
        ax[0].imshow(square_R)
        ax[1].imshow(square_G)
        ax[2].imshow(square_B)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_images(folder_path):
    IMAGES = os.listdir(folder_path)
    IMAGES = [i for i in IMAGES if i.endswith((".tiff", ".tif"))]
    IMAGES.sort()

    sharpness_values = {channel: [] for channel in ['Red', 'Green', 'Blue']}

    for i in IMAGES:
        im = tiff.imread(os.path.join(folder_path, i))
        im = im / 2**6
        for channel, index in zip(['Red', 'Green', 'Blue'], range(3)):
            channel_image = im[:, :, index]
            sharpness = variance_of_laplacian(channel_image)
            sharpness_values[channel].append(sharpness)

    return IMAGES, sharpness_values

def plot_sharpness(sharpness_values, IMAGES):
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        sns.lineplot(x=range(len(sharpness_values[channel])), y=sharpness_values[channel], marker="o", ax=ax[i], color=f'tab:{channel.lower()}')
        ax[i].set_xlabel("Image Index")
        ax[i].set_ylabel(f"Sharpness (Variance of Laplacian) - {channel}")
        ax[i].set_title(f"Sharpness of {channel} Channel")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def find_sharpest_images(folder_path, IMAGES, sharpness_values):
    sharpest_images = {}
    for channel in ['Red', 'Green', 'Blue']:
        sharpest_index = np.argmax(sharpness_values[channel])
        sharpest_images[channel] = {
            'index': sharpest_index,
            'filename': IMAGES[sharpest_index],
            'path': os.path.join(folder_path, IMAGES[sharpest_index])
        }
    return sharpest_images

@exp5_blueprint.route('/', methods=['GET', 'POST'])
def experiment5():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        if not os.path.exists(folder_path):
            return render_template('exp5.html', error="Folder does not exist.", folder_path=None)
        
        IMAGES, sharpness_values = analyze_images(folder_path)
        sharpness_plot = plot_sharpness(sharpness_values, IMAGES)
        sharpest_images = find_sharpest_images(folder_path, IMAGES, sharpness_values)

        rgb_channels_plots = {}
        for channel, info in sharpest_images.items():
            sharpest_image = tiff.imread(info['path'])
            sharpest_image = sharpest_image / 2**6
            rgb_channels_plots[channel] = {
                'plot': display_rgb_channels(sharpest_image),
                'filename': info['filename'],
                'index': info['index']
            }

        return render_template('exp5.html', folder_path=folder_path, sharpness_plot=sharpness_plot, rgb_channels_plots=rgb_channels_plots)
    
    return render_template('exp5.html', folder_path=None)
