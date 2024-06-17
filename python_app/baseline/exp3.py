# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff

sns.set_theme(style="whitegrid", font_scale=1.5)
# Green Spectrograph to be uploaded
PATH_TO_SPECTROGRAPHS = "/Users/raaromero/p301/e3/e3 spectrographs"
green_spectrograph = "p305_a3_green_LED.csv"

# Red Spectrograph to be uploaded
red_spectrograph = "p305_a3_red_LED.csv"

# Blue Spectrograph to be uploaded
blue_spectrograph = "p305_a3_white_LED.csv"

df_green = pd.read_csv(
    f"{PATH_TO_SPECTROGRAPHS}/{green_spectrograph}",
    skiprows=52,
    skipfooter=1,
    engine="python",
).reset_index()
df_green.columns = ["wavelength", "intensity"]
df_red = pd.read_csv(
    f"{PATH_TO_SPECTROGRAPHS}/{red_spectrograph}",
    skiprows=52,
    skipfooter=1,
    engine="python",
).reset_index()
df_red.columns = ["wavelength", "intensity"]
df_blue = pd.read_csv(
    f"{PATH_TO_SPECTROGRAPHS}/{blue_spectrograph}",
    skiprows=52,
    skipfooter=1,
    engine="python",
).reset_index()
df_blue.columns = ["wavelength", "intensity"]

# Normalize
green_max_intensity = df_green[df_green.intensity == df_green.intensity.max()]
red_max_intensity = df_red[df_red.intensity == df_red.intensity.max()]
blue_max_intensity = df_blue[df_blue.intensity == df_blue.intensity.max()]

# %%

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
plt.show()


# %%
# Load camera images of the LEDs

green_led_tiffile = "/Users/raaromero/p301/e3/green_led/Basler_acA1300-200uc__23312178__20240308_170125533_0000.tiff"
green_led = tiff.imread(green_led_tiffile) / 2**6
green_led = green_led[green_led.shape[0] // 2, :, 1]
green_led = green_led - np.min(green_led)
green_led_max_intensity = np.argmax(green_led)

red_led_tiffile = "/Users/raaromero/p301/e3/red_led/Basler_acA1300-200uc__23312178__20240308_170249795_0000.tiff"
red_led = tiff.imread(red_led_tiffile) / 2**6
red_led = red_led[red_led.shape[0] // 2, :, 0]
red_led_max_intensity = np.argmax(red_led)

blue_led_tiffile = "/Users/raaromero/p301/e3/white_led/Basler_acA1300-200uc__23312178__20240308_170352752_0000.tiff"
blue_led = tiff.imread(blue_led_tiffile) / 2**6
blue_led = blue_led[blue_led.shape[0] // 2, :, 2]
blue_led_max_intensity = np.argmax(blue_led)
# Figure out how to match the values from the tiff file to the spectrograph by
# linear regression

y = [
    red_max_intensity.wavelength.values[0],
    green_max_intensity.wavelength.values[0],
    blue_max_intensity.wavelength.values[0],
]
x = [red_led_max_intensity, green_led_max_intensity, blue_led_max_intensity]

f = np.poly1d(np.polyfit(x, y, 1))

# From 0 to 1280, interpolate the values
xnew = np.arange(0, 1280, 1)
ynew = f(xnew)

# %%
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
    right=df_green.rename(columns={"intensity": "green_intensity_spectra"}),
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

# %%
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

# %%
# I want to divide each normalized camera image by the normalized spectra to get a spectral sensitivity function
# for each channel
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

# %%
plt.plot(ynew, green_led / np.max(green_led), color="tab:green")
plt.plot(ynew, red_led / np.max(red_led), color="tab:red")
plt.plot(ynew, blue_led / np.max(blue_led), color="tab:blue")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Intensity")
plt.title("Spectral Images of LED calibration")


# %%
# Load integrating sphere spectra and plot alongside spectrum for integrating sphere
# Create a spectral sensitivity function for the camera from the actual spectrum for each channel
# and the image captured by the camera of the integrating sphere

is_spectrograph = "p305_a3_sphere_3A.csv"
# is_spectrograph="p305_a3_broad_lamp.csv"
df_isphere = pd.read_csv(
    f"{PATH_TO_SPECTROGRAPHS}/{is_spectrograph}",
    skiprows=52,
    skipfooter=1,
    engine="python",
).reset_index()
df_isphere.columns = ["wavelength", "intensity"]

# %%

integrating_sphere_tiffile = "/Users/raaromero/p301/e3/is/Basler_acA1300-200uc__23312178__20240308_171614944_0000.tiff"
isphere = tiff.imread(integrating_sphere_tiffile) / 2**6
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
            label = f"Integrating Sphere Spectrum"
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

# %%
df_combined["red_sensitivity"] = np.round(
    df_combined["red_intensity_camera_normalized"], 4
) / (1 / np.round(df_combined["intensity_spectra_normalized"], 4))
df_combined["green_sensitivity"] = np.round(
    df_combined["green_intensity_camera_normalized"], 4
) / (1 / np.round(df_combined["intensity_spectra_normalized"], 4))
df_combined["blue_sensitivity"] = np.round(
    df_combined["blue_intensity_camera_normalized"], 4
) / (1 / np.round(df_combined["intensity_spectra_normalized"], 4))
# %%
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
# %%
