import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import abel
from ..called_functions.anni import Anora

# for node: Verti -1, Hori 2, circi 0

# Load Block
folder = ('data')
image = ('peaks_test_high_intensity_width=10_anni=2_anni=2_anni=-1@200,260,290_more_noisy.dat')
imagefile = ('examples/'f"/{folder}/{image}")
r_range = [
(190, 195), (191, 196), (192, 197), (193, 198), (194, 199),
(195, 200), (196, 201), (197, 202), (198, 203), (199, 204),
(200, 205), (201, 206), (202, 207), (203, 208), (204, 209),
(205, 210), (206, 211), (207, 212), (208, 213), (209, 214),
(210, 215), (211, 216), (212, 217), (213, 218), (214, 219),
(215, 220), (216, 221), (217, 222), (218, 223), (219, 224),
(220, 225), (221, 226), (222, 227), (223, 228), (224, 229),
(225, 230), (226, 231), (227, 232), (228, 233), (229, 234),
(230, 235), (231, 236), (232, 237), (233, 238), (234, 239),
(235, 240), (236, 241), (237, 242), (238, 243), (239, 244),
(240, 245), (241, 246), (242, 247), (243, 248), (244, 249),
(245, 250), (246, 251), (247, 252), (248, 253), (249, 254),
(250, 255), (251, 256), (252, 257), (253, 258), (254, 259),
(255, 260), (256, 261), (257, 262), (258, 263), (259, 264),
(260, 265), (261, 266), (262, 267), (263, 268), (264, 269),
(265, 270), (266, 271), (267, 272), (268, 273), (269, 274),
(270, 275), (271, 276), (272, 277), (273, 278), (274, 279),
(275, 280), (276, 281), (277, 282), (278, 283), (279, 284),
(280, 285), (281, 286), (282, 287), (283, 288), (284, 289),
(285, 290), (286, 291), (287, 292), (288, 293), (289, 294),
(290, 295), (291, 296), (292, 297), (293, 298), (294, 299),
(295, 300), (296, 301), (297, 302), (298, 303), (299, 304),
(300, 305), (301, 306), (302, 307), (303, 308), (304, 309),
(305, 310), (306, 311), (307, 312), (308, 313), (309, 314),
(310, 315), (311, 316), (312, 317), (313, 318), (314, 319),
(315, 320), (316, 321), (317, 322)]
fwrd = np.loadtxt(imagefile)
# IM_N = IM / np.max(IM)
origin = 'convolution'
# fwrd = abel.Transform(IM, direction='forward', method='direct',
#                        origin=origin, verbose=True).transform
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


def analyze_inverse_method(invrs_data, method_name, x0, y0, r_range):
    Anni = Anora(invrs_data, x0, y0)

    # Lists to store results
    r_centers = []
    beta2_values = []
    beta2_errors = []
    intensities = []

    print(f"\nAnalyzing inverse method: {method_name}")
    print(f"Average image intensity: {Anni.avg_intensity:.4f}")

    # Loop over radial ranges
    for r_min, r_max in r_range:

        range_intensity = Anni.get_average_intensity_for_range(r_min, r_max)

        if range_intensity > 0.1*Anni.avg_intensity:
            beta2_fit, beta2_err, theta_deg, W_theta, theta_plot_deg, W_fit = Anni.calculate_beta2(r_min, r_max)
            r_center = (r_min + r_max) / 2

            # Store results
            r_centers.append(r_center)
            beta2_values.append(beta2_fit)
            beta2_errors.append(beta2_err)
            intensities.append(range_intensity)


    if r_centers:
        print("\nResults for regions above average intensity:")
        print("Radial Center\tβ₂\t\tError\t\tIntensity")
        for r_center, beta2, error, intensity in zip(r_centers, beta2_values, beta2_errors, intensities):
            print(f"{r_center:.1f}\t\t{beta2:.4f}\t± {error:.4f}\t{intensity:.4f}")

        # Plot beta2 vs radial position
        plt.figure(figsize=(8, 6))
        plt.errorbar(r_centers, beta2_values, yerr=beta2_errors, fmt='o', capsize=5)
        plt.xlabel('Radial Position (pixels)')
        plt.ylabel('Anisotropy Parameter β₂')
        plt.title(f'{method_name}: Anisotropy Parameter vs. Radial Position\n(For regions above average intensity)')
        plt.grid(True)
        plt.show()
    else:
        print("No radial ranges had intensity above the average image intensity.")

    return r_centers, beta2_values, beta2_errors, intensities


# Create dictionary of inverse transforms and their names
inverse_methods = {
    'BASEX': invrs_bsx,
    'rBASEX': invrs_rbsx,
    # 'rBASEX': invrs_rbsx,
    'DAUN': invrs_daun,
    'DIRECT': invrs_direct,
    'HANSENLAW': invrs_hansenlaw,
    'ONION_BORDAS': invrs_onion_bordas,
    'ONION_PEELING': invrs_onion_peeling,
    'THREE_POINT': invrs_three_point,
    'TWO_POINT': invrs_two_point
}


height, width = invrs.shape

x0 = width / 2
y0 = height / 2


# Dictionary to store results for all methods
all_results = {}

# Analyze each inverse method
for method_name, invrs_data in inverse_methods.items():
    results = analyze_inverse_method(invrs_data, method_name, x0, y0, r_range)
    all_results[method_name] = results


plt.figure(figsize=(12, 8))
for method_name, (r_centers, beta2_values, beta2_errors, _) in all_results.items():
    if r_centers:
        plt.errorbar(r_centers, beta2_values, yerr=beta2_errors,
                     fmt='-o', capsize=5, label=method_name)

plt.xlabel('Radial Position (pixels)')
plt.ylabel('Anisotropy Parameter β₂')
plt.title('Comparison of Anisotropy Parameters Across All Methods')
plt.legend()
plt.grid(True)
plt.show()
