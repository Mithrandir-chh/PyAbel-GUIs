import numpy as np
from scipy.special import legendre
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import abel

'''''Verti -1, Hori 2, circi 0'''''

def find_major_peaks(speeds, height_threshold=0.3, distance=5, prominence=0.1):
    # Identify significant peaks in the speed distribution.
    #
    # Parameters:
    # speeds : (radii, intensities)
    #     The speed distribution data from angular_integration_3D
    # height_threshold : float
    #     Minimum peak height relative to maximum intensity (0 to 1)
    # distance : int
    #     Minimum distance between peaks in pixels
    # prominence : float
    #     Minimum prominence of peaks relative to maximum intensity (0 to 1)
    #
    # Returns:
    # peaks_info : dict
    #     Dictionary containing:
    #     - 'peak_positions': radii values where peaks occur
    #     - 'peak_intensities': intensity values at the peaks
    #     - 'peak_properties': additional properties from scipy.signal.find_peaks
    #     - 'peak_indices': indices where peaks occur
    radii, intensities = speeds

    # Calculate absolute threshold from relative value
    abs_height = height_threshold * np.max(intensities)
    abs_prominence = prominence * np.max(intensities)

    # Find peaks
    peak_indices, properties = find_peaks(intensities,
                                          height=abs_height,
                                          distance=distance,
                                          prominence=abs_prominence)

    return {
        'peak_positions': radii[peak_indices],
        'peak_intensities': intensities[peak_indices],
        'peak_properties': properties,
        'peak_indices': peak_indices
    }

import numpy as np


def radial_integrate(image):
    # Get image dimensions and center
    ny, nx = image.shape
    center_y, center_x = ny // 2, nx // 2

    # Create coordinate grid
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)

    # Create output arrays
    r_unique = np.arange(r.max() + 1)
    integrated = np.zeros_like(r_unique, dtype=float)

    # Sum values at each radius
    for i in range(len(r_unique)):
        mask = r == i
        integrated[i] = np.sum(image[mask])

    return r_unique, integrated


# Load and process image
imagefile = ('data/'
             'peaks_test_high_intensity_width=10_anni=2_anni=2_anni=-1@200,260,290_forward_more_noisy.dat')
IM = np.loadtxt(imagefile)
IM_N = IM / np.max(IM)

origin = 'convolution'
fig = plt.figure(figsize=(15, 5))
gs = plt.GridSpec(1, 5)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])


# Plot the raw data
im1 = ax1.imshow(IM, origin='lower')
fig.colorbar(im1, ax=ax1, fraction=0.1, shrink=0.5, pad=0.09)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')

recon = abel.Transform(IM, direction='forward', method='direct',
                       origin=origin, verbose=True).transform
recon_ = abel.Transform(IM, direction='inverse', method='basex',
                       origin=origin, verbose=True).transform

speeds = abel.tools.vmi.angular_integration_3D(recon_)

r, integrated = radial_integrate(recon_)

# error = np.abs(IM - recon_)
# mse = np.mean(error**2)  # Mean Squared Error

# print(f"Mean Squared Error (MSE): {mse:.6f}")

peaks = find_major_peaks(speeds)
print("\nPeak Analysis Results:")
print("-" * 50)
print(f"{'Index':<10}{'Position':<15}{'Intensity':<15}")
print("-" * 50)
for idx, pos, intensity in zip(peaks['peak_indices'],
                             peaks['peak_positions'],
                             peaks['peak_intensities']):
    print(f"{idx:<10}{pos:<15.2f}{intensity:<15.4f}")

im2 = ax2.imshow(recon, origin='lower')
fig.colorbar(im2, ax=ax2, fraction=0.1, shrink=0.5, pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')

ax3.plot(r, integrated, 'b-', label='Speed Distribution')
ax3.set_xlabel('Speed (pixel)')
ax3.set_ylabel('Yield')
ax3.legend()


im4 = ax4.imshow(recon_, origin='lower')
fig.colorbar(im4, ax=ax4, fraction=0.1, shrink=0.5, pad=0.03)
ax4.set_xlabel('x (pixels)')
ax4.set_ylabel('y (pixels)')
# Find and plot peaks
peaks = find_major_peaks(speeds)

ax5.plot(*speeds, 'b-', label='Speed Distribution')
ax5.plot(peaks['peak_positions'], peaks['peak_intensities'], 'ro', label='Major Peaks')
ax5.set_xlabel('Speed (pixel)')
ax5.set_ylabel('Yield')
ax5.legend()


plt.tight_layout()
plt.show()