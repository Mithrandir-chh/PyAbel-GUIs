import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
import os
import abel


class GerberLikeImage:
    def __init__(self, n=361):
        self.size = n
        self.center = n // 2

        # Create coordinate grids in pixel space
        y, x = np.ogrid[:n, :n]
        self.r_pixels = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)
        self.theta = np.arctan2(-(x - self.center), y - self.center)

        # Default peak parameters in pixel units
        # [radius_pixels, intensity, beta2, width_pixels]
        self.peak_params = [
            [40, 1.2, -0.4, 2],
            [90, 1.5, 1.0, 2],
            [134, 2.0, 1.0, 2],
            [196, 2.0, 1.0, 2],
        ]

    def clear_peaks(self):
        self.peak_params = []
    def add_peak(self, radius_pixels, intensity, beta2, width_pixels=2):
        self.peak_params.append([radius_pixels, intensity, beta2, width_pixels])

    def generate_peak(self, r0_pixels, width_pixels, intensity, beta2):
        # Calculate radial distribution in pixel space
        radial = np.exp(-(self.r_pixels - r0_pixels) ** 2 / (2 * width_pixels ** 2))

        # Calculate angular distribution in pixel space
        angular = 1 + beta2 * legendre(2)(np.cos(self.theta))

        # Combine radial and angular components
        peak = radial * angular

        # Calculate normalization in pixel space
        # Use the actual area element in polar coordinates: R * dR * dTheta
        # Create pixel-space grid for R and Theta
        y, x = np.indices((self.size, self.size))
        R = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)
        Theta = np.arctan2(y - self.center, x - self.center)

        # Recalculate radial and angular components for normalization
        radial_norm = np.exp(-(R - r0_pixels) ** 2 / (2 * width_pixels ** 2))
        angular_norm = 1 + beta2 * legendre(2)(np.cos(Theta))

        # Compute the area element
        dA = 1

        # Compute the total volume (integrated intensity)
        volume = np.sum(radial_norm * angular_norm * dA)

        # Calculate normalized intensity
        N = intensity / volume

        # Return final peak distribution
        return N * peak

    def generate_image(self):
        """Generate complete test image with all peaks."""
        image = np.zeros((self.size, self.size))

        for r0, intensity, beta2, width in self.peak_params:
            image += self.generate_peak(r0, width, intensity, beta2)

        return image

    def get_theoretical_beta(self, r_pixels):
        """Calculate theoretical beta2 parameter at each radius."""
        beta2 = np.zeros_like(r_pixels, dtype=float)
        intensity = np.zeros_like(r_pixels, dtype=float)

        for r0, amp, b2, width in self.peak_params:
            contrib = amp * np.exp(-(r_pixels - r0) ** 2 / (2 * width ** 2))
            intensity += contrib
            beta2 += b2 * contrib

        mask = intensity > 0
        beta2[mask] /= intensity[mask]

        return beta2

    def add_noise(self, image, noise_level=0.01):
        # Store original range
        orig_min = image.min()
        orig_max = image.max()

        # Shift to positive if needed
        if orig_min < 0:
            image = image - orig_min

        # Scale and ensure no zeros
        scaled = (image / image.max()) / noise_level
        scaled = np.maximum(scaled, 1e-10)

        # Generate noisy image
        noisy = np.random.poisson(scaled)
        noisy = noisy * noise_level

        # Shift back if needed
        if orig_min < 0:
            noisy = noisy + orig_min

        # Rescale to original range
        noisy_min = noisy.min()
        noisy_max = noisy.max()
        noisy_norm = (noisy - noisy_min) / (noisy_max - noisy_min)
        matched = noisy_norm * (orig_max - orig_min) + orig_min

        return matched

def save_as_dat(image, filename, directory='gerber_test_images_tests'):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Ensure filename has .dat extension
    if not filename.endswith('.dat'):
        filename += '.dat'

    # Save the array
    filepath = os.path.join(directory, filename)
    np.savetxt(filepath, image)
def experiment_with_peaks():
    """Test function with pixel-based peak positions."""
    gerber = GerberLikeImage(n=1000)

    # Clear default peaks
    gerber.clear_peaks()

    # Add peaks with pixel-based positions
    gerber.add_peak(radius_pixels=200, intensity=200.0, beta2=2, width_pixels=30)
    gerber.add_peak(radius_pixels=260, intensity=200.0, beta2=2, width_pixels=30)
    gerber.add_peak(radius_pixels=290, intensity=40.0, beta2=-1, width_pixels=6)

    image = gerber.generate_image()
    origin = 'convolution'
    recon = abel.Transform(image, direction='forward', method='direct',
                           origin=origin, verbose=True).transform
    noisy_image = gerber.add_noise(recon, 0.1)

    save_as_dat(image, 'peaks_test_high_intensity_width=10_anni=2_anni=2_anni=-1@200,260,290.dat')
    save_as_dat(recon, 'peaks_test_high_intensity_width=10_anni=2_anni=2_anni=-1@200,260,290_forward.dat')
    save_as_dat(noisy_image, 'peaks_test_high_intensity_width=10_anni=2_anni=2_anni=-1@200,260,290_forward_more_noisy.dat')


    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.colorbar()
    plt.title('Test Peaks (Pixel-Based Positions)')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(recon)
    plt.colorbar()
    plt.title('Test Peaks (Pixel-Based Positions)')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(noisy_image)
    plt.colorbar()
    plt.title('Test Peaks (Pixel-Based Positions)')
    plt.show()


if __name__ == "__main__":
    experiment_with_peaks()
    # transform_method = 'basex'
    # results = generate_and_analyze(transform_method)
    # plt.show()