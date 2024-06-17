from flask import Blueprint, render_template, request, current_app
import os
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns

exp3_blueprint = Blueprint(
    "exp3", __name__, template_folder="templates", static_folder="static"
)

plt.switch_backend("Agg")


def process_experiment_3(
    upload_folder,
    green_spectrograph,
    red_spectrograph,
    blue_spectrograph,
    isphere_spectrograph,
    green_image,
    red_image,
    blue_image,
    isphere_image,
):
    sns.set_theme(style="whitegrid", font_scale=1.5)

    # Load spectrographs
    df_green = pd.read_csv(
        green_spectrograph, skiprows=52, skipfooter=1, engine="python"
    ).reset_index()
    df_green.columns = ["wavelength", "intensity"]
    df_red = pd.read_csv(
        red_spectrograph, skiprows=52, skipfooter=1, engine="python"
    ).reset_index()
    df_red.columns = ["wavelength", "intensity"]
    df_blue = pd.read_csv(
        blue_spectrograph, skiprows=52, skipfooter=1, engine="python"
    ).reset_index()
    df_blue.columns = ["wavelength", "intensity"]
    df_isphere = pd.read_csv(
        isphere_spectrograph, skiprows=52, skipfooter=1, engine="python"
    ).reset_index()
    df_isphere.columns = ["wavelength", "intensity"]

    # Normalize spectrographs
    green_max_intensity = df_green[
        df_green.intensity == df_green.intensity.max()
    ]
    red_max_intensity = df_red[df_red.intensity == df_red.intensity.max()]
    blue_max_intensity = df_blue[df_blue.intensity == df_blue.intensity.max()]

    # Plot normalized LED spectrographs
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Green", "Red", "Blue"]
    for df in [df_green, df_red, df_blue]:
        label = labels.pop(0)
        plot_color = f"tab:{label.lower()}"
        plt.plot(
            df.wavelength,
            df.intensity / df.intensity.max(),
            color=plot_color,
            label=label,
        )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.title("LED Spectrographs")
    plt.legend()
    spectrograph_plot_path = os.path.join(
        upload_folder, "spectrograph_plot.png"
    )
    plt.savefig(spectrograph_plot_path)
    plt.close()

    # Load camera images of the LEDs
    green_led = tiff.imread(green_image) / 2**6
    green_led = green_led[green_led.shape[0] // 2, :, 1]
    green_led = green_led - np.min(green_led)
    green_led_max_intensity = np.argmax(green_led)

    red_led = tiff.imread(red_image) / 2**6
    red_led = red_led[red_led.shape[0] // 2, :, 0]
    red_led_max_intensity = np.argmax(red_led)

    blue_led = tiff.imread(blue_image) / 2**6
    blue_led = blue_led[blue_led.shape[0] // 2, :, 2]
    blue_led_max_intensity = np.argmax(blue_led)

    # Linear regression to match the values from the tiff file to the spectrograph
    y = [
        red_max_intensity.wavelength.values[0],
        green_max_intensity.wavelength.values[0],
        blue_max_intensity.wavelength.values[0],
    ]
    x = [
        red_led_max_intensity,
        green_led_max_intensity,
        blue_led_max_intensity,
    ]

    f = np.poly1d(np.polyfit(x, y, 1))

    # Interpolate the values
    xnew = np.arange(0, 1280, 1)
    ynew = f(xnew)

    # Combine data
    df_combined = pd.DataFrame(
        {
            "wavelength": ynew,
            "green_intensity_camera": green_led,
            "red_intensity_camera": red_led,
            "blue_intensity_camera": blue_led,
        }
    )
    df_combined = pd.merge_asof(
        left=df_combined,
        right=df_red.rename(columns={"intensity": "red_intensity_spectra"}),
        on="wavelength",
        direction="nearest",
        tolerance=0.1,
    )
    df_combined = pd.merge_asof(
        left=df_combined,
        right=df_green.rename(
            columns={"intensity": "green_intensity_spectra"}
        ),
        on="wavelength",
        direction="nearest",
        tolerance=0.1,
    )
    df_combined = pd.merge_asof(
        left=df_combined,
        right=df_blue.rename(columns={"intensity": "blue_intensity_spectra"}),
        on="wavelength",
        direction="nearest",
        tolerance=0.1,
    )
    df_combined = df_combined.dropna()

    for col in df_combined.columns:
        if col != "wavelength":
            df_combined[f"{col}_normalized"] = (
                df_combined[col] / df_combined[col].max()
            )

    # Plot combined data
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in df_combined.columns:
        if "normalized" in col:
            if "spectra" in col:
                linestyle = "--"
                label = f"{col.split('_')[0].capitalize()} LED Spectrum"
                linewidth = 2
            else:
                linestyle = "-"
                label = f"{col.split('_')[0].capitalize()} LED Camera Image"
                linewidth = 3
            plt.plot(
                df_combined.wavelength,
                df_combined[col],
                color=f"tab:{col.split('_')[0]}",
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            plt.legend(loc="lower left")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    combined_plot_path = os.path.join(upload_folder, "combined_plot.png")
    plt.savefig(combined_plot_path)
    plt.close()

    # Spectral sensitivity
    df_combined["red_sensitivity"] = np.round(
        df_combined["red_intensity_camera_normalized"], 4
    ) / (1 / np.round(df_combined["red_intensity_spectra_normalized"], 4))
    df_combined["green_sensitivity"] = np.round(
        df_combined["green_intensity_camera_normalized"], 4
    ) / (1 / np.round(df_combined["green_intensity_spectra_normalized"], 4))
    df_combined["blue_sensitivity"] = np.round(
        df_combined["blue_intensity_camera_normalized"], 4
    ) / (1 / np.round(df_combined["blue_intensity_spectra_normalized"], 4))

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_combined.columns:
        if "sensitivity" in col:
            plt.plot(
                df_combined.wavelength,
                df_combined[col],
                color=f"tab:{col.split('_')[0]}",
                label=f"{col.split('_')[0].capitalize()}",
            )
            plt.ylabel("Spectral Sensitivity")
            plt.title("Spectral Sensitivity for LED Calibration")
            plt.legend()
            plt.xlabel("Wavelength (nm)")
    sensitivity_plot_path = os.path.join(upload_folder, "sensitivity_plot.png")
    plt.savefig(sensitivity_plot_path)
    plt.close()

    # Spectral Images of LED calibration
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(ynew, green_led / np.max(green_led), color="tab:green")
    plt.plot(ynew, red_led / np.max(red_led), color="tab:red")
    plt.plot(ynew, blue_led / np.max(blue_led), color="tab:blue")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.title("Spectral Images of LED Calibration")
    calibration_plot_path = os.path.join(upload_folder, "calibration_plot.png")
    plt.savefig(calibration_plot_path)
    plt.close()

    # Load integrating sphere spectra and plot alongside spectrum for integrating sphere
    df_isphere = pd.read_csv(
        isphere_spectrograph, skiprows=52, skipfooter=1, engine="python"
    ).reset_index()
    df_isphere.columns = ["wavelength", "intensity"]

    isphere = tiff.imread(isphere_image) / 2**6
    isphere = isphere[isphere.shape[0] // 2, :, :]
    isphere = isphere - np.min(isphere, axis=0)
    isphere = isphere / np.max(isphere, axis=0)

    df_combined = pd.DataFrame(
        {
            "wavelength": ynew,
            "green_intensity_camera": isphere[:, 1],
            "red_intensity_camera": isphere[:, 0],
            "blue_intensity_camera": isphere[:, 2],
        }
    )
    df_combined = pd.merge_asof(
        left=df_combined,
        right=df_isphere.rename(columns={"intensity": "intensity_spectra"}),
        on="wavelength",
        direction="nearest",
        tolerance=0.1,
    )
    df_combined = df_combined.dropna()

    for col in df_combined.columns:
        if col != "wavelength":
            df_combined[f"{col}_normalized"] = (
                df_combined[col] / df_combined[col].max()
            )

    fig, ax = plt.subplots(figsize=(12, 8))
    for col in df_combined.columns:
        if "normalized" in col:
            if "spectra" in col:
                linestyle = "--"
                label = "Integrating Sphere Spectrum"
                linewidth = 3
                color = "black"
            else:
                linestyle = "-"
                label = f"{col.split('_')[0].capitalize()} LED Camera Image"
                linewidth = 2
                color = f"tab:{col.split('_')[0]}"
            plt.plot(
                df_combined.wavelength,
                df_combined[col],
                color=color,
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            plt.legend(loc="lower left")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    integrating_sphere_plot_path = os.path.join(
        upload_folder, "integrating_sphere_plot.png"
    )
    plt.savefig(integrating_sphere_plot_path)
    plt.close()

    # Spectral sensitivity for integrating sphere
    df_combined["red_sensitivity"] = np.round(
        df_combined["red_intensity_camera_normalized"], 4
    ) / (1 / np.round(df_combined["intensity_spectra_normalized"], 4))
    df_combined["green_sensitivity"] = np.round(
        df_combined["green_intensity_camera_normalized"], 4
    ) / (1 / np.round(df_combined["intensity_spectra_normalized"], 4))
    df_combined["blue_sensitivity"] = np.round(
        df_combined["blue_intensity_camera_normalized"], 4
    ) / (1 / np.round(df_combined["intensity_spectra_normalized"], 4))

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_combined.columns:
        if "sensitivity" in col:
            plt.plot(
                df_combined.wavelength,
                df_combined[col],
                color=f"tab:{col.split('_')[0]}",
                label=f"{col.split('_')[0].capitalize()}",
            )
            plt.ylabel("Spectral Sensitivity")
            plt.legend()
            plt.xlabel("Wavelength (nm)")
    integrating_sphere_sensitivity_plot_path = os.path.join(
        upload_folder, "integrating_sphere_sensitivity_plot.png"
    )
    plt.savefig(integrating_sphere_sensitivity_plot_path)
    plt.close()

    return [
        spectrograph_plot_path,
        combined_plot_path,
        sensitivity_plot_path,
        calibration_plot_path,
        integrating_sphere_plot_path,
        integrating_sphere_sensitivity_plot_path,
    ]


@exp3_blueprint.route("/", methods=["GET", "POST"])
def exp3():
    if request.method == "POST":
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        green_spectrograph = request.files["green_spectrograph"]
        red_spectrograph = request.files["red_spectrograph"]
        blue_spectrograph = request.files["blue_spectrograph"]
        isphere_spectrograph = request.files["isphere_spectrograph"]
        green_image = request.files["green_image"]
        red_image = request.files["red_image"]
        blue_image = request.files["blue_image"]
        isphere_image = request.files["isphere_image"]

        green_spectrograph_path = os.path.join(
            upload_folder, green_spectrograph.filename
        )
        red_spectrograph_path = os.path.join(
            upload_folder, red_spectrograph.filename
        )
        blue_spectrograph_path = os.path.join(
            upload_folder, blue_spectrograph.filename
        )
        isphere_spectrograph_path = os.path.join(
            upload_folder, isphere_spectrograph.filename
        )
        green_image_path = os.path.join(upload_folder, green_image.filename)
        red_image_path = os.path.join(upload_folder, red_image.filename)
        blue_image_path = os.path.join(upload_folder, blue_image.filename)
        isphere_image_path = os.path.join(
            upload_folder, isphere_image.filename
        )

        green_spectrograph.save(green_spectrograph_path)
        red_spectrograph.save(red_spectrograph_path)
        blue_spectrograph.save(blue_spectrograph_path)
        isphere_spectrograph.save(isphere_spectrograph_path)
        green_image.save(green_image_path)
        red_image.save(red_image_path)
        blue_image.save(blue_image_path)
        isphere_image.save(isphere_image_path)

        plots = process_experiment_3(
            upload_folder,
            green_spectrograph_path,
            red_spectrograph_path,
            blue_spectrograph_path,
            isphere_spectrograph_path,
            green_image_path,
            red_image_path,
            blue_image_path,
            isphere_image_path,
        )

        return render_template("exp3/exp3.html", plots=plots)
    return render_template("exp3/exp3.html", plots=None)
