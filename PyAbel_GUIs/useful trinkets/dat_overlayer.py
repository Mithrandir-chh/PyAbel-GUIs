import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_dat_file(filename):
    try:
        return np.loadtxt(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def create_overlay_plot(file1, file2, color1='red', color2='blue', alpha=0.5):
    data1 = load_dat_file(file1)
    data2 = load_dat_file(file2)

    if data1 is None or data2 is None:
        return

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(data1, cmap=LinearSegmentedColormap.from_list('custom1', ['white', color1]))
    plt.colorbar(label='Intensity (File 1)')
    plt.title('File 1')

    plt.subplot(132)
    plt.imshow(data2, cmap=LinearSegmentedColormap.from_list('custom2', ['white', color2]))
    plt.colorbar(label='Intensity (File 2)')
    plt.title('File 2')

    plt.subplot(133)

    data1_norm = (data1 - data1.min()) / (data1.max() - data1.min())
    data2_norm = (data2 - data2.min()) / (data2.max() - data2.min())

    plt.imshow(data1_norm, cmap=LinearSegmentedColormap.from_list('custom1', ['white', color1]), alpha=alpha)
    plt.imshow(data2_norm, cmap=LinearSegmentedColormap.from_list('custom2', ['white', color2]), alpha=alpha)
    plt.colorbar(label='Normalized Intensity')
    plt.title('Overlay')

    plt.tight_layout()
    plt.show()


def main():

    file1 = 'data/gerber_peaks_test_eq_high_intensity_width=6_anni=2@50.dat'
    file2 = 'data/gerber_peaks_test_eq_high_intensity_width=6_anni=-1,2@40,50.dat'


    create_overlay_plot(file1, file2)

    # create_overlay_plot(file1, file2, color1='green', color2='purple', alpha=0.7)


if __name__ == "__main__":
    main()