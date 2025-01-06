import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import abel
from ..called_functions.anni import Anora

# Verti -1, Hori 2, circi 0

# Load Block
folder = ('data')
image = ('peaks_test_high_intensity_width=10_anni=2_anni=-1@200,210_more_noisy.dat')
imagefile = (f"{folder}/{image}")
r_range = [(100, 100), (101, 101), (102, 102), (103, 103), (104, 104),
(105, 105), (106, 106), (107, 107), (108, 108), (109, 109),
(110, 110), (111, 111), (112, 112), (113, 113), (114, 114),
(115, 115), (116, 116), (117, 117), (118, 118), (119, 119),
(120, 120), (121, 121), (122, 122), (123, 123), (124, 124),
(125, 125), (126, 126), (127, 127), (128, 128), (129, 129),
(130, 130), (131, 131), (132, 132), (133, 133), (134, 134),
(135, 135), (136, 136), (137, 137), (138, 138), (139, 139),
(140, 140), (141, 141), (142, 142), (143, 143), (144, 144),
(145, 145), (146, 146), (147, 147), (148, 148), (149, 149),
(150, 150), (151, 151), (152, 152), (153, 153), (154, 154),
(155, 155), (156, 156), (157, 157), (158, 158), (159, 159),
(160, 160), (161, 161), (162, 162), (163, 163), (164, 164),
(165, 165), (166, 166), (167, 167), (168, 168), (169, 169),
(170, 170), (171, 171), (172, 172), (173, 173), (174, 174),
(175, 175), (176, 176), (177, 177), (178, 178), (179, 179),
(180, 180), (181, 181), (182, 182), (183, 183), (184, 184),
(185, 185), (186, 186), (187, 187), (188, 188), (189, 189),
(190, 190), (191, 191), (192, 192), (193, 193), (194, 194),
(195, 195), (196, 196), (197, 197), (198, 198), (199, 199),
(200, 200), (201, 201), (202, 202), (203, 203), (204, 204),
(205, 205), (206, 206), (207, 207), (208, 208), (209, 209),
(210, 210), (211, 211), (212, 212), (213, 213), (214, 214),
(215, 215), (216, 216), (217, 217), (218, 218), (219, 219),
(220, 220), (221, 221), (222, 222), (223, 223), (224, 224),
(225, 225), (226, 226), (227, 227), (228, 228), (229, 229),
(230, 230), (231, 231), (232, 232), (233, 233), (234, 234),
(235, 235), (236, 236), (237, 237), (238, 238), (239, 239),
(240, 240), (241, 241), (242, 242), (243, 243), (244, 244),
(245, 245), (246, 246), (247, 247), (248, 248), (249, 249),
(250, 250), (251, 251), (252, 252), (253, 253), (254, 254),
(255, 255), (256, 256), (257, 257), (258, 258), (259, 259),
(260, 260), (261, 261), (262, 262), (263, 263), (264, 264),
(265, 265), (266, 266), (267, 267), (268, 268), (269, 269),
(270, 270), (271, 271), (272, 272), (273, 273), (274, 274),
(275, 275), (276, 276), (277, 277), (278, 278), (279, 279),
(280, 280), (281, 281), (282, 282), (283, 283), (284, 284),
(285, 285), (286, 286), (287, 287), (288, 288), (289, 289),
(290, 290), (291, 291), (292, 292), (293, 293), (294, 294),
(295, 295), (296, 296), (297, 297), (298, 298), (299, 299),
(300, 300)]
IM = np.loadtxt(imagefile)
IM_N = IM / np.max(IM)
origin = 'convolution'
fwrd = abel.Transform(IM, direction='forward', method='direct',
                       origin=origin, verbose=True).transform
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
    """
    Analyze a single inverse transform method.

    Parameters:
    -----------
    invrs_data : numpy.ndarray
        The inverse transformed image data
    method_name : str
        Name of the inverse method used
    x0, y0 : float
        Center coordinates of the image
    r_range : list of tuples
        List of (r_min, r_max) ranges to analyze
    """

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
    'rBASEX': invrs_rbsx,
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
