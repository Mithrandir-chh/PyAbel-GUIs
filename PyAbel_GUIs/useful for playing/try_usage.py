import numpy as np
from scipy.special import legendre
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import abel

# Verti -1, Hori 2, circi 0
folder = ('data')
image = ('(example) Subc-C2H4I2_37000 diss_32105REMPI_2point05 avg and point077 avg_ 3 hours attempt  1_ 7-6-2022.dat')
imagefile = (f"{folder}/{image}")

def find_major_peaks(speeds, height_threshold=0.3, distance=5, prominence=0.1):
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

def save_as_dat(image, filename, directory='data/forward transformed'):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Ensure filename has .dat extension
    if not filename.endswith('.dat'):
        filename += '.dat'

    # Save the array
    filepath = os.path.join(directory, filename)
    np.savetxt(filepath, image)

fwrd = np.loadtxt(imagefile)
# IM_N = IM / np.max(IM)  # normalized dat file

origin = 'convolution'

# fwrd = abel.Transform(IM, direction='forward', method='direct',
#                        origin=origin, verbose=True).transform
# save_as_dat(fwrd, f"{image}_forward_transformed.dat")

invrs = abel.Transform(fwrd, direction='inverse', method='basex',
                       origin=origin, verbose=True).transform
invrs_bsx = abel.Transform(fwrd, direction='inverse', method='basex',
                       origin=origin, verbose=True).transform
invrs_rbsx = abel.Transform(fwrd, direction='inverse', method='rbasex',
                       origin=origin, verbose=True).transform
# invrs_lnbsx = abel.Transform(fwrd, direction='inverse', method='linbasex',
#                        origin=origin, verbose=True).transform
invrs_daun = abel.Transform(fwrd, direction='inverse', method='daun',
                       origin=origin, verbose=True).transform
invrs_direct = abel.Transform(fwrd, direction='inverse', method='direct',
                       origin=origin, verbose=True).transform
invrs_hansenlaw = abel.Transform(fwrd, direction='inverse', method='hansenlaw',
                       origin=origin, verbose=True).transform
invrs_onion_bordas = abel.Transform(fwrd, direction='inverse', method='onion_bordas',
                       origin=origin, verbose=True).transform
invrs_onion_peeling = abel.Transform(fwrd, direction='inverse', method='onion_peeling',
                       origin=origin, verbose=True).transform
invrs_three_point = abel.Transform(fwrd, direction='inverse', method='three_point',
                       origin=origin, verbose=True).transform
invrs_two_point = abel.Transform(fwrd, direction='inverse', method='direct',
                       origin=origin, verbose=True).transform


speeds = abel.tools.vmi.angular_integration_3D(fwrd)

r, integrated = radial_integrate(invrs)

# error = np.abs(fwrd - invrs)
# mse = np.mean(error**2)  # Mean Squared Error
error_bsx_rbsx = np.abs(invrs_bsx - invrs_rbsx)
# error_bsx_lnbsx = np.abs(invrs_bsx - invrs_lnbsx)
error_bsx_drct = np.abs(invrs_bsx - invrs_direct)
error_bsx_hansen = np.abs(invrs_bsx - invrs_hansenlaw)
error_bsx_daun = np.abs(invrs_bsx - invrs_daun)
error_bsx_onionb = np.abs(invrs_bsx - invrs_onion_bordas)
error_bsx_onionp = np.abs(invrs_bsx - invrs_onion_peeling)
error_bsx_trip = np.abs(invrs_bsx - invrs_three_point)
error_bsx_zweip = np.abs(invrs_bsx - invrs_two_point)
# error_rbsx_lnbsx = np.abs(invrs_rbsx - invrs_lnbsx)
error_rbsx_drct = np.abs(invrs_rbsx - invrs_direct)
error_rbsx_hansen = np.abs(invrs_rbsx - invrs_hansenlaw)
error_rbsx_daun = np.abs(invrs_rbsx - invrs_daun)
error_rbsx_onionb = np.abs(invrs_rbsx - invrs_onion_bordas)
error_rbsx_onionp = np.abs(invrs_rbsx - invrs_onion_peeling)
error_rbsx_trip = np.abs(invrs_rbsx - invrs_three_point)
error_rbsx_zweip = np.abs(invrs_rbsx - invrs_two_point)
error_drct_hansen = np.abs(invrs_direct - invrs_hansenlaw)
error_drct_daun = np.abs(invrs_direct - invrs_daun)
error_drct_onionb = np.abs(invrs_direct - invrs_onion_bordas)
error_drct_onionp = np.abs(invrs_direct - invrs_onion_peeling)
error_drct_trip = np.abs(invrs_direct - invrs_three_point)
error_drct_zweip = np.abs(invrs_direct - invrs_two_point)
error_hnsn_daun = np.abs(invrs_hansenlaw - invrs_daun)
error_hnsn_onionb = np.abs(invrs_hansenlaw - invrs_onion_bordas)
error_hnsn_onionp = np.abs(invrs_hansenlaw - invrs_onion_peeling)
error_hnsn_trip = np.abs(invrs_hansenlaw - invrs_three_point)
error_hnsn_zweip = np.abs(invrs_hansenlaw - invrs_two_point)
error_daun_onionb = np.abs(invrs_daun - invrs_onion_bordas)
error_daun_onionp = np.abs(invrs_daun - invrs_onion_peeling)
error_daun_trip = np.abs(invrs_daun - invrs_three_point)
error_daun_zweip = np.abs(invrs_daun - invrs_two_point)
error_ob_onionp = np.abs(invrs_onion_bordas - invrs_onion_peeling)
error_ob_trip = np.abs(invrs_onion_bordas - invrs_three_point)
error_ob_zweip = np.abs(invrs_onion_bordas - invrs_two_point)
error_trip_zweip = np.abs(invrs_three_point - invrs_two_point)

mse_bsx_rbsx = np.mean(error_bsx_rbsx**2)  # Mean Squared Error
# mse_bsx_lnbsx = np.mean(error_bsx_lnbsx**2)  # Mean Squared Error
mse_bsx_drct = np.mean(error_bsx_drct**2)  # Mean Squared Error
mse_bsx_hansen = np.mean(error_bsx_hansen**2)  # Mean Squared Error
mse_bsx_daun = np.mean(error_bsx_daun**2)  # Mean Squared Error
mse_bsx_onionb = np.mean(error_bsx_onionb**2)  # Mean Squared Error
mse_bsx_onionp = np.mean(error_bsx_onionp**2)  # Mean Squared Error
mse_bsx_trip = np.mean(error_bsx_trip**2)  # Mean Squared Error
mse_bsx_zweip = np.mean(error_bsx_zweip**2)  # Mean Squared Error
# mse_rbsx_lnbsx = np.mean(error_rbsx_lnbsx**2)  # Mean Squared Error
mse_rbsx_drct = np.mean(error_rbsx_drct**2)  # Mean Squared Error
mse_rbsx_hansen = np.mean(error_rbsx_hansen**2)  # Mean Squared Error
mse_rbsx_daun = np.mean(error_rbsx_daun**2)  # Mean Squared Error
mse_rbsx_onionb = np.mean(error_rbsx_onionb**2)  # Mean Squared Error
mse_rbsx_onionp = np.mean(error_rbsx_onionp**2)  # Mean Squared Error
mse_rbsx_trip = np.mean(error_rbsx_trip**2)  # Mean Squared Error
mse_rbsx_zweip = np.mean(error_rbsx_zweip**2)  # Mean Squared Error
mse_drct_hansen = np.mean(error_drct_hansen**2)  # Mean Squared Error
mse_drct_daun = np.mean(error_drct_daun**2)  # Mean Squared Error
mse_drct_onionb = np.mean(error_drct_onionb**2)  # Mean Squared Error
mse_drct_onionp = np.mean(error_drct_onionp**2)  # Mean Squared Error
mse_drct_trip = np.mean(error_drct_trip**2)  # Mean Squared Error
mse_drct_zweip = np.mean(error_drct_zweip**2)  # Mean Squared Error
mse_hnsn_daun = np.mean(error_hnsn_daun**2)  # Mean Squared Error
mse_hnsn_onionb = np.mean(error_hnsn_onionb**2)  # Mean Squared Error
mse_hnsn_onionp = np.mean(error_hnsn_onionp**2)  # Mean Squared Error
mse_hnsn_trip = np.mean(error_hnsn_trip**2)  # Mean Squared Error
mse_hnsn_zweip = np.mean(error_hnsn_zweip**2)  # Mean Squared Error
mse_daun_onionb = np.mean(error_daun_onionb**2)  # Mean Squared Error
mse_daun_onionp = np.mean(error_daun_onionp**2)  # Mean Squared Error
mse_daun_trip = np.mean(error_daun_trip**2)  # Mean Squared Error
mse_daun_zweip = np.mean(error_daun_zweip**2)  # Mean Squared Error
mse_ob_onionp = np.mean(error_ob_onionp**2)  # Mean Squared Error
mse_ob_trip = np.mean(error_ob_trip**2)  # Mean Squared Error
mse_ob_zweip = np.mean(error_ob_zweip**2)  # Mean Squared Error
mse_trip_zweip = np.mean(error_trip_zweip**2)  # Mean Squared Error


# print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Squared Error (MSE)bsx-rbsx: {mse_bsx_rbsx:.6f}")
# print(f"Mean Squared Error (MSE)bsx-lnbsx: {mse_bsx_lnbsx:.6f}")
print(f"Mean Squared Error (MSE)bsx-drct: {mse_bsx_drct:.6f}")
print(f"Mean Squared Error (MSE)bsx-hnsn: {mse_bsx_hansen:.6f}")
print(f"Mean Squared Error (MSE)bsx-daun: {mse_bsx_daun:.6f}")
print(f"Mean Squared Error (MSE)bsx-ob: {mse_bsx_onionb:.6f}")
print(f"Mean Squared Error (MSE)bsx-op: {mse_bsx_onionp:.6f}")
print(f"Mean Squared Error (MSE)bsx-tp: {mse_bsx_trip:.6f}")
print(f"Mean Squared Error (MSE)bsx-zp: {mse_bsx_zweip:.6f}")
# print(f"Mean Squared Error (MSE)rbsx-lnbsx: {mse_rbsx_lnbsx:.6f}")
print(f"Mean Squared Error (MSE)rbsx-drct: {mse_rbsx_drct:.6f}")
print(f"Mean Squared Error (MSE)rbsx-hnsn: {mse_rbsx_hansen:.6f}")
print(f"Mean Squared Error (MSE)rbsx-daun: {mse_rbsx_daun:.6f}")
print(f"Mean Squared Error (MSE)rbsx-ob: {mse_rbsx_onionb:.6f}")
print(f"Mean Squared Error (MSE)rbsx-op: {mse_rbsx_onionp:.6f}")
print(f"Mean Squared Error (MSE)rbsx-tp: {mse_rbsx_trip:.6f}")
print(f"Mean Squared Error (MSE)rbsx-zp: {mse_rbsx_zweip:.6f}")
print(f"Mean Squared Error (MSE)drct-hnsn: {mse_drct_hansen:.6f}")
print(f"Mean Squared Error (MSE)drct-daun: {mse_drct_daun:.6f}")
print(f"Mean Squared Error (MSE)drct-ob: {mse_drct_onionb:.6f}")
print(f"Mean Squared Error (MSE)drct-op: {mse_drct_onionp:.6f}")
print(f"Mean Squared Error (MSE)drct-tp: {mse_drct_trip:.6f}")
print(f"Mean Squared Error (MSE)drct-zp: {mse_drct_zweip:.6f}")
print(f"Mean Squared Error (MSE)hnsn-daun: {mse_hnsn_daun:.6f}")
print(f"Mean Squared Error (MSE)hnsn-ob: {mse_hnsn_onionb:.6f}")
print(f"Mean Squared Error (MSE)hnsn-op: {mse_hnsn_onionp:.6f}")
print(f"Mean Squared Error (MSE)hnsn-tp: {mse_hnsn_trip:.6f}")
print(f"Mean Squared Error (MSE)hnsn-zp: {mse_hnsn_zweip:.6f}")
print(f"Mean Squared Error (MSE)daun-ob: {mse_daun_onionb:.6f}")
print(f"Mean Squared Error (MSE)daun-op: {mse_daun_onionp:.6f}")
print(f"Mean Squared Error (MSE)daun-tp: {mse_daun_trip:.6f}")
print(f"Mean Squared Error (MSE)daun-zp: {mse_daun_zweip:.6f}")
print(f"Mean Squared Error (MSE)ob-op: {mse_ob_onionp:.6f}")
print(f"Mean Squared Error (MSE)ob-tp: {mse_ob_trip:.6f}")
print(f"Mean Squared Error (MSE)ob-zp: {mse_ob_zweip:.6f}")
print(f"Mean Squared Error (MSE)tp-zp: {mse_trip_zweip:.6f}")



peaks = find_major_peaks(speeds)
print("\nPeak Analysis Results:")
print("-" * 50)
print(f"{'Index':<10}{'Position':<15}{'Intensity':<15}")
print("-" * 50)
for idx, pos, intensity in zip(peaks['peak_indices'],
                             peaks['peak_positions'],
                             peaks['peak_intensities']):
    print(f"{idx:<10}{pos:<15.2f}{intensity:<15.4f}")

fig = plt.figure(figsize=(15, 5))
gs = plt.GridSpec(3, 5)
# Create the original 5 subplots in the top row
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
ax6 = fig.add_subplot(gs[1, 0])
ax7 = fig.add_subplot(gs[1, 1])
ax8 = fig.add_subplot(gs[1, 2])
ax9 = fig.add_subplot(gs[1, 3])
ax10 = fig.add_subplot(gs[1, 4])
ax11 = fig.add_subplot(gs[2, 0])
ax12 = fig.add_subplot(gs[2, 1])
ax13 = fig.add_subplot(gs[2, 2])
ax14 = fig.add_subplot(gs[2, 3])
ax15 = fig.add_subplot(gs[2, 4])


# Plot the raw data
im1 = ax1.imshow(fwrd, origin='lower')
fig.colorbar(im1, ax=ax1, fraction=0.1, shrink=0.5, pad=0.09)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')

im2 = ax2.imshow(fwrd, origin='lower')
fig.colorbar(im2, ax=ax2, fraction=0.1, shrink=0.5, pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')

ax3.plot(r, integrated, 'b-', label='Speed Distribution')
ax3.set_xlabel('Speed (pixel)')
ax3.set_ylabel('Yield')
ax3.legend()


im4 = ax4.imshow(invrs, origin='lower')
fig.colorbar(im4, ax=ax4, fraction=0.1, shrink=0.5, pad=0.03)
ax4.set_xlabel('x (pixels)')
ax4.set_ylabel('y (pixels)')
# Find and plot peaks
peaks = find_major_peaks(speeds)

# Plot peaks
ax5.plot(*speeds, 'b-', label='Speed Distribution')
ax5.plot(peaks['peak_positions'], peaks['peak_intensities'], 'ro', label='Major Peaks')
ax5.set_xlabel('Speed (pixel)')
ax5.set_ylabel('Yield')
ax5.legend()

im6 = ax6.imshow(invrs_bsx, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im6, ax=ax6, fraction=0.1, shrink=0.5, pad=0.03)
ax6.set_xlabel('x (pixels)')
ax6.set_ylabel('y (pixels)')
ax6.set_title('basex')

im7 = ax7.imshow(invrs_rbsx, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im7, ax=ax7, fraction=0.1, shrink=0.5, pad=0.03)
ax7.set_xlabel('x (pixels)')
ax7.set_ylabel('y (pixels)')
ax7.set_title('rbasex')

im8 = ax8.imshow(invrs_direct, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im8, ax=ax8, fraction=0.1, shrink=0.5, pad=0.03)
ax8.set_xlabel('x (pixels)')
ax8.set_ylabel('y (pixels)')
ax8.set_title('direct')

im9 = ax9.imshow(invrs_hansenlaw, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im9, ax=ax9, fraction=0.1, shrink=0.5, pad=0.03)
ax9.set_xlabel('x (pixels)')
ax9.set_ylabel('y (pixels)')
ax9.set_title('Hansen_Law')

im10 = ax10.imshow(invrs_daun, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im10, ax=ax10, fraction=0.1, shrink=0.5, pad=0.03)
ax10.set_xlabel('x (pixels)')
ax10.set_ylabel('y (pixels)')
ax10.set_title('Daun')

# im11 = ax11.imshow(invrs_lnbsx, origin='lower', vmin=0)
# fig.colorbar(im11, ax=ax11, fraction=0.1, shrink=0.5, pad=0.03)
# ax11.set_xlabel('x (pixels)')
# ax11.set_ylabel('y (pixels)')
# ax11.set_title('Linbasex')

im12 = ax12.imshow(invrs_onion_bordas, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im12, ax=ax12, fraction=0.1, shrink=0.5, pad=0.03)
ax12.set_xlabel('x (pixels)')
ax12.set_ylabel('y (pixels)')
ax12.set_title('Onion_Brodas')

im13 = ax13.imshow(invrs_onion_peeling, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im13, ax=ax13, fraction=0.1, shrink=0.5, pad=0.03)
ax13.set_xlabel('x (pixels)')
ax13.set_ylabel('y (pixels)')
ax13.set_title('Onion_Peeling')

im14 = ax14.imshow(invrs_three_point, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im14, ax=ax14, fraction=0.1, shrink=0.5, pad=0.03)
ax14.set_xlabel('x (pixels)')
ax14.set_ylabel('y (pixels)')
ax14.set_title('Three_Point')

im15 = ax15.imshow(invrs_two_point, origin='lower', vmin=0, vmax=0.1)
fig.colorbar(im15, ax=ax15, fraction=0.1, shrink=0.5, pad=0.03)
ax15.set_xlabel('x (pixels)')
ax15.set_ylabel('y (pixels)')
ax15.set_title('Two_Point')


plt.tight_layout()
plt.show()