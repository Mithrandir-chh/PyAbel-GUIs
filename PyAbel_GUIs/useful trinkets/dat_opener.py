import numpy as np
import matplotlib.pyplot as plt


def display_dat_image(filename1, filename2):
    image1 = np.loadtxt(filename1)
    image2 = np.loadtxt(filename2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    im1 = ax1.imshow(image1, origin='lower')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')
    ax1.set_title('Image 1')

    im2 = ax2.imshow(image2, origin='lower')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    ax2.set_xlabel('x (pixels)')
    ax2.set_ylabel('y (pixels)')
    ax2.set_title('Image 2')
    plt.tight_layout()

    plt.show()

display_dat_image(
    'your file1 path'
,
    'your file2 path')
