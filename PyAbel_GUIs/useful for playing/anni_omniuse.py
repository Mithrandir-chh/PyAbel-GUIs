import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import abel
from ..called_functions.generate_list import generate_ranges
from ..called_functions.generate_list import generate_rolling_ranges
from ..called_functions.anni import Anora



# Verti -1, Hori 2, circi 0

# Load Block
imagefile = '/Users/haohuiche/Desktop/WUSTL/FL24/PLab/2-20-2023 Diss 20350 REMPI 32452- 2P3-2_ calibration 25000 frames.dat'
IM = np.loadtxt(imagefile)
IM_N = IM / np.max(IM)
IM_Centered = abel.tools.center.center_image (IM, method='convolution')
origin = 'image_center'
recon_, distr = abel.Transform(IM_Centered, direction='inverse', method='rbasex',
                       origin=origin, verbose=True).transform
rbasex_r, rbasex_I, rbasex_beta = recon_.distr.rIbeta()
plt.imshow(recon_, aspect='auto')  # aspect='auto' helps with non-square data
plt.colorbar(label='Intensity')
plt.title('DAT File Visualization')
plt.show()

# Get the image dimensions
height, width = recon_.shape

# Define the center of the image
x0 = width / 2
y0 = height / 2

# Initialize Anora
Anni = Anora(recon_, x0, y0)

# Define multiple radial ranges
ranges = [(155, 170)]
r_range = generate_ranges(ranges, step=1)

# Lists to store results
r_centers = []
beta2_values = []
beta2_errors = []
intensities = []

# Loop over radial ranges and calculate beta2 only for ranges above average intensity
for r_min, r_max in r_range:
    # Calculate average intensity for this range
    range_intensity = Anni.get_average_intensity_for_range(r_min, r_max)

    # Only proceed if the range intensity is above the image average
    if range_intensity > Anni.avg_intensity:
        # Plotting Block:: beta2 vs. radial center (only for ranges that passed the intensity threshold)
        beta2_fit, beta2_err, theta_deg, W_theta, theta_plot_deg, W_fit = Anni.calculate_beta2(r_min, r_max)
        r_center = (r_min + r_max) / 2

        # Store the results
        r_centers.append(r_center)
        beta2_values.append(beta2_fit)
        beta2_errors.append(beta2_err)
        intensities.append(range_intensity)

        # # Plot the angular distribution and fit for each radial range
        # plt.figure(figsize=(6, 4))
        # plt.plot(theta_deg, W_theta, 'o', label='Data')
        # plt.plot(theta_plot_deg, W_fit, '-', label=f'Fit β₂={beta2_fit:.2f}')
        # plt.xlabel('Theta (degrees)')
        # plt.ylabel('Normalized Intensity')
        # plt.title(f'Angular Distribution (r = {r_min}-{r_max})\nIntensity: {range_intensity:.2f}')
        # plt.legend()
        # plt.show()
if r_centers:  # Only create plot if we have data
    print("\nInverse Method: basex")
    print("\nResults for regions above average intensity:")
    print(f"Average image intensity: {Anni.avg_intensity:.4f}")
    print("\nRadial Center\tβ₂\t\tError\t\tIntensity")
    for r_center, beta2, error, intensity in zip(r_centers, beta2_values, beta2_errors, intensities):
        print(f"{r_center:.1f}\t\t{beta2:.4f}\t± {error:.4f}\t{intensity:.4f}")
    plt.figure(figsize=(8, 6))
    plt.errorbar(r_centers, beta2_values, yerr=beta2_errors, fmt='o', capsize=5)
    plt.xlabel('Radial Position (pixels)')
    plt.ylabel('Anisotropy Parameter β₂')
    plt.title('Anisotropy Parameter vs. Radial Position\n(For regions above average intensity)')
    plt.grid(True)
    plt.show()
    print(f"rbasex gives: {rbasex_r, rbasex_I, rbasex_beta}")
else:
    print("No radial ranges had intensity above the average image intensity.")


