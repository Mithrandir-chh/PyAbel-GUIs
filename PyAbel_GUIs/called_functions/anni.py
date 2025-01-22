import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import abel


# for node: Verti -1, Hori 2, circi 0
class Anora:
    def __init__(self, image, x0=None, y0=None):
        """
        Initialize
        Parameters:
        - image: 2D NumPy array for image.
        - x0, y0: Coordinates of the image center. If None, defaults to the center of the image.
        """
        self.lower_bound = -1.0  # change beta2 lower bound here
        self.upper_bound = 2.0  # change beta2 upper bound here
        self.image = image
        self.height, self.width = image.shape
        self.x0 = x0 if x0 is not None else self.width / 2
        self.y0 = y0 if y0 is not None else self.height / 2
        # Calculate the average intensity of the entire image
        self.avg_intensity = np.mean(self.image)

    def get_average_intensity_for_range(self, r_min, r_max, num_theta_bins=360):
        """
        Calculate the average intensity for a given radial range.

        Parameters:
        - r_min: Minimum radius (in pixels)
        - r_max: Maximum radius (in pixels)
        - num_theta_bins: Number of angular bins

        Returns:
        - float: Average intensity in the radial range
        """
        # Define the angular bins
        theta = np.linspace(0, 2 * np.pi, num_theta_bins, endpoint=False)

        # Define the radial points within the radial range
        num_r_bins = r_max - r_min + 1
        r = np.linspace(r_min, r_max, num_r_bins)

        # Create the grid of (r, theta) values
        R, Theta = np.meshgrid(r, theta)

        # Convert polar coordinates to Cartesian coordinates
        X = self.x0 + R * np.cos(Theta)
        Y = self.y0 + R * np.sin(Theta)

        # Flatten X and Y arrays
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        # Ensure that X and Y are within the image boundaries
        X_flat = np.clip(X_flat, 0, self.width - 1)
        Y_flat = np.clip(Y_flat, 0, self.height - 1)

        # Create coordinates array for interpolation
        coords = np.vstack((Y_flat, X_flat))

        # Interpolate the intensity values at the (X, Y) coordinates
        intensity_values = ndimage.map_coordinates(self.image, coords, order=1, mode='constant', cval=0.0)

        # Calculate the average intensity
        return np.mean(intensity_values)

    def calculate_beta2(self, r_min, r_max, num_theta_bins=360):
        """
        Calculate the anisotropy parameter beta2 for a given radial range.

        Parameters:
        - r_min: Minimum radius (in pixels).
        - r_max: Maximum radius (in pixels).
        - num_theta_bins: Number of angular bins.

        Returns:
        - beta2_fit: Fitted anisotropy parameter beta2.
        - beta2_err: Fitting error of beta2.
        - theta: Array of theta values (in degrees).
        - W_theta: Normalized angular distribution.
        - W_fit: Fitted angular distribution over theta.
        """
        # Define the angular bins
        theta = np.linspace(0, 2 * np.pi, num_theta_bins, endpoint=False)

        # Define the radial points within the radial range
        num_r_bins = r_max - r_min + 1
        r = np.linspace(r_min, r_max, num_r_bins)

        # Create the grid of (r, theta) values
        R, Theta = np.meshgrid(r, theta)

        # Convert polar coordinates to Cartesian coordinates
        X = self.x0 + R * np.cos(Theta)
        Y = self.y0 + R * np.sin(Theta)

        # Flatten X and Y arrays
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        # Ensure that X and Y are within the image boundaries
        X_flat = np.clip(X_flat, 0, self.width - 1)
        Y_flat = np.clip(Y_flat, 0, self.height - 1)

        # Create coordinates array for interpolation
        coords = np.vstack((Y_flat, X_flat))  # Y coordinates first because image is indexed as image[y, x]

        # Interpolate the intensity values at the (X, Y) coordinates
        intensity_values = ndimage.map_coordinates(self.image, coords, order=1, mode='constant', cval=0.0)

        # Reshape the intensity values back to the grid shape
        intensity_grid = intensity_values.reshape(R.shape)

        # Sum over the radial axis to get the angular distribution
        angular_distribution_values = np.sum(intensity_grid, axis=1)

        # Normalize the angular distribution
        total_intensity = np.sum(angular_distribution_values)
        if total_intensity == 0:
            # Avoid division by zero
            W_theta = angular_distribution_values
        else:
            W_theta = angular_distribution_values / total_intensity

        # Define the theoretical angular distribution function
        def angular_distribution(theta, A, beta2):
            P2 = 0.5 * (3 * np.sin(theta) ** 2 - 1)
            return A * (1 + beta2 * P2)

        # Initial guesses for A and beta2
        initial_guess = [1.0, 0.0]

        # Bounds for the parameters: A > 0, beta2 between -1 and 2
        bounds = ([0.0, self.lower_bound], [np.inf, self.upper_bound])

        # Perform the curve fitting
        try:
            popt, pcov = curve_fit(
                angular_distribution, theta, W_theta, p0=initial_guess, bounds=bounds
            )
            A_fit, beta2_fit = popt
            perr = np.sqrt(np.diag(pcov))
            A_err, beta2_err = perr
        except RuntimeError:
            # If the fit fails, set beta2 and error to NaN
            beta2_fit = np.nan
            beta2_err = np.nan

        # Generate the fitted angular distribution for plotting
        theta_plot = np.linspace(0, 2 * np.pi, 1000)
        W_fit = angular_distribution(theta_plot, A_fit, beta2_fit)

        return beta2_fit, beta2_err, np.degrees(theta), W_theta, np.degrees(theta_plot), W_fit
