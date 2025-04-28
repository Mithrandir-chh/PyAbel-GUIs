import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy import ndimage


class Anora:
    def __init__(self, image, x0=None, y0=None):
        """
               Initialize
               Parameters:
               - image: 2D NumPy array for image.
               - x0, y0: Coordinates of the image center. If None, defaults to the center of the image.
               """
        self.image = np.asarray(image, dtype=float)
        self.height, self.width = self.image.shape
        self.x0 = x0 if x0 is not None else self.width / 2
        self.y0 = y0 if y0 is not None else self.height / 2
        self.lower_bound = -10
        self.upper_bound = 20
        self.tot_average_intensity = np.mean(self.image)

    # Utilities
    def average_intensity(self, r_min: int, r_max: int, *, num_theta_bins: int = 360):
        """
                Calculate average intensity of an r range
                Return Intensity (float) within the annulus r_min–r_max (pixels).
                """
        _, counts, _ = self._extract_counts(r_min, r_max, num_theta_bins, normalize=False)
        return float(counts.mean())

    def _extract_counts(self, r_min: int, r_max: int, num_theta_bins: int,
                        normalize: bool):
        """
                Polarize the coordinates
                Parameters:
                - r_min, r_max: Coordinates of the image center.
                - num_theta_bins: Number of theta bins.
                - normalize: If True, normalize the intensity.

                Return (θ_rad, counts, σ) for annulus; counts may be normalized."""
        if r_min >= r_max:
            raise ValueError("r_min must be < r_max")
        if num_theta_bins < 4:
            raise ValueError("need at least 4 angular bins")

        theta = np.linspace(0.0, 2 * np.pi, num_theta_bins, endpoint=False)
        radii = np.arange(r_min, r_max + 1)
        R, TH = np.meshgrid(radii, theta)
        X = self.x0 + R * np.cos(TH)
        Y = self.y0 + R * np.sin(TH)
        coords = np.vstack((Y.ravel(), X.ravel()))
        values = ndimage.map_coordinates(self.image, coords, order=1, mode="constant", cval=0.0)
        counts = values.reshape(TH.shape).sum(axis=1)
        sigma = self._poisson_sigma(counts)

        if normalize and counts.sum() > 0:
            counts = counts / counts.sum()
            sigma = sigma / counts.sum()  # propagate same scaling
        return theta, counts, sigma

    # Science!
    def calculate_beta2(self, r_min, r_max, *, num_theta_bins=360,
                        method="robust",  # "robust" | "RANSAC"
                        RANSAC_trials=250, RANSAC_min_samples=25,
                        RANSAC_global_thresh=3.5, RANSAC_special_thresh=2, show=True):
        """
        Calculate the anisotropy parameter beta2 for a given radial range.

        Parameters:
        - r_min (Int): Minimum radius (in pixels).
        - r_max (Int): Maximum radius (in pixels). Int
        - num_theta_bins (Int): Number of angular bins.
        - method: Method to use for fitting the anisotropy parameter.
        - RANSAC_trials: Max number of trials for RANSAC. Default: 250.
        - RANSAC_min_samples: Starting number of samples for RANSAC. Default: 250.
        - RANSAC_global_thresh: Threshold factor to determine inliers for RANSAC. Default: 3.5
        - RANSAC_special_thresh: Threshold factor to determine inliers for RANSAC at around 90 and 270 degrees. Deflt: 2
        - show (bool): If True, show the figure.

        Returns:
        - A (tuple): A tuple with the following elements: A_fit, A_err
        - beta2 (tuple): A tuple with the following elements: beta2_fit, beta2_err
        - theta_deg (float): A NumPy array of floats representing the angular bins in degrees
        - W_theta: A NumPy array of floats representing the ///normalized/// angular distribution
        - theta_fit_deg: A NumPy array of floats representing the same angular bins as theta_deg (in degrees)
        - W_fit: A NumPy array of floats representing the fitted angular distribution
        - inliers: A NumPy array of booleans indicating which data points are considered inliers during RANSAC
        """
        theta, counts, sigma = self._extract_counts(r_min, r_max, num_theta_bins, normalize=False)

        if method == "robust":
            params, cov, inliers = self._fit_robust(theta, counts, sigma)
        elif method == "RANSAC":
            params, cov, inliers = self._fit_RANSAC(theta, counts, sigma,
                                                    trials=RANSAC_trials,
                                                    min_samples=RANSAC_min_samples,
                                                    mad_scale=RANSAC_global_thresh,
                                                    mad_special_scale=RANSAC_special_thresh)
        else:
            raise ValueError("method must be 'robust' or 'RANSAC'")

        A_fit, beta2_fit = params
        A_err, beta2_err = (np.sqrt(np.diag(cov)) if cov is not None else (np.nan, np.nan))

        # Normalize intensity with probability density
        bin_width = 2 * np.pi / num_theta_bins
        data_prob = counts / counts.sum()  # probability per bin
        # fit evaluated at the same bin centres → normalize by discrete sum
        fit_bins = self._angular_distribution(theta, A_fit, beta2_fit)
        fit_bins_norm = fit_bins / fit_bins.sum()

        # also make a smooth curve for eye‑guidance (same scaling!)
        theta_plot = np.linspace(0.0, 2 * np.pi, 720)
        fit_pdf = self._angular_distribution(theta_plot, A_fit, beta2_fit)
        fit_pdf /= np.trapz(fit_pdf, theta_plot)  # pdf – area = 1
        fit_plot = fit_pdf * bin_width  # probability per *bin*

        if show:
            plt.figure(figsize=(9, 6))
            plt.plot(np.degrees(theta), data_prob, "o", ms=5, alpha=0.6, label="data (norm.)")
            plt.plot(np.degrees(theta_plot), fit_plot, "-", lw=2, label="fit (norm.)")
            plt.scatter(np.degrees(theta[inliers]), data_prob[inliers],
                        s=70, facecolors="none", edgecolors="lime", linewidths=1.4, label="inliers")
            plt.xlabel("θ (deg)")
            plt.ylabel("probability per bin")
            plt.title(rf"r = {(r_min + r_max) / 2}  |  β₂ = {beta2_fit:.3f} ± {beta2_err:.3f}  |  A = {A_fit:.3g}")
            plt.grid(alpha=0.25)
            plt.legend(framealpha=0.9)
            plt.tight_layout()
            plt.show()

        return {
            "A": (A_fit, A_err),
            "beta2": (beta2_fit, beta2_err),
            "theta_deg": np.degrees(theta),
            "W_theta": data_prob,
            "theta_fit_deg": np.degrees(theta),  # fit on same discrete bins
            "W_fit": fit_bins_norm,
            "inliers": inliers,
        }

    # Model & statistics helpers
    @staticmethod
    def _angular_distribution(theta: np.ndarray, A: float, beta2: float) -> np.ndarray:
        P2 = 0.5 * (3 * np.sin(theta) ** 2 - 1)
        return A * (1 + beta2 * P2)

    @staticmethod
    def _poisson_sigma(counts: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(counts, 1.0, None))

    # Robust-least‑squares and RANSAC fitters
    def _fit_robust(self, theta, counts, sigma):
        """
            Fit the annisotropy with RANSAC algorithm.
            Parameters:
                theta (array): an array of all angular bins
                counts (array): an array of sum of intensities at each angular bin (at a specified r range)
                sigma (array): an array of standard deviations (sqrt(N)) of each angular bin (at a specified r range)

            Returns:
                res.x (tuple): a tuple with fitted parameters A and beta2
                cov (matrix): a 2x2 covariance matrix of fitted parameters
                np.ones_like (array): a boolean array of shape same of theta with True all over
                """
        def resid(p):  # residual
            return (self._angular_distribution(theta, *p) - counts) / sigma

        bounds = ([0.0, self.lower_bound], [np.inf, self.upper_bound])
        init = [counts.mean(), 0.0]
        res = least_squares(resid, init, bounds=bounds, loss="soft_l1", f_scale=2.0)
        if not res.success:
            raise RuntimeError("Robust fit did not converge")
        # rough covariance from JᵀJ
        J = res.jac
        rss = 2.0 * res.cost  # because cost = ½‖r‖²
        dof = max(1, len(theta) - J.shape[1])
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        JTJ_inv = VT.T @ np.diag(1.0 / (s ** 2 + 1e-12)) @ VT
        cov = JTJ_inv * rss / dof
        return res.x, cov, np.ones_like(theta, dtype=bool)

    def _fit_RANSAC(self, theta, counts, sigma, *, trials: int, min_samples: int, mad_scale: float, mad_special_scale):
        """
            Fit the anisotropy with RANSAC algorithm.
            Parameters:
                theta (array): an array of all angular bins
                counts (array): an array of sum of counts at each angular bin
                sigma (array): an array of standard deviations sqrt(N) of each ring
                trials (int): max number of RANSAC trials
                min_samples (int): min number of inliers of a RANSAC trial
                mad_scale (float): threshold factor for determining in/outliers for angular bins out of 90/270 degrees
                mad_special_scale (float): threshold factor for angular bins around 90/270 degrees

            Returns:
                p_fin (tuple): a tuple with fitted parameters A and beta2
                cov (matrix): a 2x2 covariance matrix of fitted parameters
                best_inliers (array): a boolean array of whether the corresponding theta is an inlier (0-360)
        """
        rng = np.random.default_rng()
        best_score, best_par, best_inliers = -np.inf, None, None
        bounds = ([0.0, self.lower_bound], [np.inf, self.upper_bound])
        init = [counts.mean(), 0.0]
        theta_deg = np.degrees(theta) % 360

        for _ in range(trials):
            subset = rng.choice(theta.size, size=min_samples, replace=False)
            try:
                p_sub, _ = curve_fit(self._angular_distribution, theta[subset], counts[subset],
                                     p0=init, bounds=bounds, sigma=sigma[subset], absolute_sigma=True)
            except RuntimeError:
                continue
            res_abs = np.abs(self._angular_distribution(theta, *p_sub) - counts) / sigma
            mad = np.median(res_abs)

            thresh = np.full_like(theta, mad_scale * mad + 1e-12)

            # Special threshold for 90 (±5) and 270 (±5) degrees regions
            is_special_angle = ((theta_deg >= 80) & (theta_deg <= 100)) | ((theta_deg >= 260) & (theta_deg <= 280))

            # Apply a different threshold for these special angles (e.g., 1.5x the normal threshold)
            special_angle_factor = 0.9  # Adjust this factor as needed
            special_angle_scaling = 2.5
            thresh[is_special_angle] = mad_special_scale * mad + 1e-12
            inl = res_abs < thresh
            if inl.sum() < min_samples:
                continue
            score = inl.sum() - 5.0 * res_abs[inl].mean()
            if score > best_score:
                best_score, best_par, best_inliers = score, p_sub, inl
            if best_score >= 0.95 * theta.size:
                break
        if best_par is None:
            return self._fit_robust(theta, counts, sigma)
        # consensus refit
        try:
            p_fin, cov = curve_fit(self._angular_distribution, theta[best_inliers], counts[best_inliers],
                                   p0=best_par, bounds=bounds, method='trf', sigma=sigma[best_inliers], absolute_sigma=False)
        except RuntimeError:
            return self._fit_robust(theta, counts, sigma)
        return p_fin, cov, best_inliers
