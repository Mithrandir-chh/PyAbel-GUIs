import numpy as np
import abel
from scipy.ndimage import shift
from scipy.signal import find_peaks, peak_prominences
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from six.moves import tkinter as tk
from six.moves import tkinter_ttk as ttk
from six.moves import tkinter_tkfiledialog as filedialog
from called_functions.anni import Anora

Abel_methods = ['basex', 'direct', 'hansenlaw', 'linbasex', 'onion_bordas',
                'onion_peeling', 'rbasex', 'three_point', 'two_point']

center_methods = ['com', 'convolution', 'gaussian', 'slice']

root = tk.Tk()
root.wm_title("Simple GUI PyAbel")

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)


IM = None
AIM = None
raw_IM = None
centered_IM = None
speed_distribution = None
radial_coords = None


def update_plot(image=None):
    global IM, canvas, vmin_slider, vmax_slider, vmin_entry, vmax_entry
    if image is None:
        image = IM
    if image is not None:
        f.clf()
        a = f.add_subplot(111)
        vmin = float(vmin_entry.get())
        vmax = float(vmax_entry.get())
        im = a.imshow(image, vmin=vmin, vmax=vmax)
        f.colorbar(im)
        canvas.draw()


# def update_slider_range(data):
#     global last_vmin, last_vmax
#     # Only update if we don't have saved values
#     if last_vmin is None or last_vmax is None:
#         vmin = np.min(data)
#         vmax = np.max(data)
#         range_size = vmax - vmin
#
#         if range_size <= 2:
#             resolution = 0.05
#         else:
#             resolution = (vmax - vmin) / 100
#
#         vmin_slider.config(from_=vmin, to=vmax, resolution=resolution)
#         vmax_slider.config(from_=vmin, to=vmax, resolution=resolution)
#         vmin_slider.set(vmin)
#         vmax_slider.set(vmax)
#         vmin_entry.delete(0, tk.END)
#         vmin_entry.insert(0, f"{vmin:.2f}")
#         vmax_entry.delete(0, tk.END)
#         vmax_entry.insert(0, f"{vmax:.2f}")
def update_slider_range(data):
    vmin = np.min(data)
    vmax = np.max(data)
    range_size = vmax - vmin

    if range_size <= 2:
        resolution = 0.05
    else:
        resolution = (vmax - vmin) / 100

    vmin_slider.config(from_=vmin, to=vmax, resolution=resolution)
    vmax_slider.config(from_=vmin, to=vmax, resolution=resolution)
    vmin_slider.set(vmin)
    vmax_slider.set(vmax)
    vmin_entry.delete(0, tk.END)
    vmin_entry.insert(0, f"{vmin:.2f}")
    vmax_entry.delete(0, tk.END)
    vmax_entry.insert(0, f"{vmax:.2f}")

def slider_changed(value):
    global last_vmin, last_vmax
    vmin = vmin_slider.get()
    vmax = vmax_slider.get()
    last_vmin = vmin
    last_vmax = vmax
    vmin_entry.delete(0, tk.END)
    vmin_entry.insert(0, f"{vmin:.2f}")
    vmax_entry.delete(0, tk.END)
    vmax_entry.insert(0, f"{vmax:.2f}")
    update_plot()


def entry_changed(*args):
    global last_vmin, last_vmax
    try:
        vmin = float(vmin_entry.get())
        vmax = float(vmax_entry.get())
        last_vmin = vmin
        last_vmax = vmax
        vmin_slider.set(vmin)
        vmax_slider.set(vmax)
        update_plot()
    except ValueError:
        pass

def _display():
    global IM, raw_IM, canvas, text
    text.insert(tk.END, "raw image\n")
    IM = raw_IM.copy()
    update_slider_range(IM)
    update_plot()

def _getfilename():
    global IM, raw_IM, centered_IM, text, last_vmin, last_vmax
    last_vmin = None  # Reset saved values
    last_vmax = None
    fn = filedialog.askopenfilename()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"reading image file {fn}\n")
    canvas.draw()

    if ".txt" in fn or ".dat" in fn:
        IM = np.loadtxt(fn)
    else:
        IM = imread(fn)

    if IM.shape[0] % 2 == 0:
        text.insert(tk.END, "make image odd size")
        IM = shift(IM, (-0.5, -0.5))[:-1, :-1]
    if len(IM.shape) == 3:
        IM = np.mean(IM, axis=2)

    raw_IM = IM.copy()
    centered_IM = None
    update_slider_range(IM)
    _display()

def _center():
    global cent, IM, raw_IM, centered_IM, text
    method = cent.get()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"centering image using abel.tools.center.center_image(method={method})\n")
    canvas.draw()
    centered_IM = abel.tools.center.center_image(raw_IM, method=method, odd_size=True)
    IM = centered_IM.copy()
    update_slider_range(IM)
    update_plot()

def _display_centered():
    global IM, raw_IM, centered_IM, canvas, text
    if centered_IM is not None:
        text.insert(tk.END, "centered raw image\n")
        IM = centered_IM.copy()
        update_slider_range(IM)
        update_plot()
    else:
        text.insert(tk.END, "Please center the image first\n")

def P2(x):
    return (3*x*x-1)/2

def PAD(theta, beta, amp):
    return amp*(1 + beta*P2(np.cos(theta)))

def _transform():
    global IM, AIM, raw_IM, centered_IM, canvas, transform, text
    method = transform.get()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"inverse Abel transform: {method}\n")
    if "basex" in method:
        text.insert(tk.END, "  first time calculation of the basis functions may take a while ...\n")
    if "direct" in method:
        text.insert(tk.END, "   calculation is slowed if Cython unavailable ...\n")
    canvas.draw()

    image_to_transform = centered_IM if centered_IM is not None else raw_IM

    AIM = abel.Transform(image_to_transform, method=method, direction="inverse", symmetry_axis=None)
    # IM = AIM.transform
    # update_slider_range(IM)
    # update_plot()

def _display_transformed():
    global IM, raw_IM, AIM, canvas, text
    if AIM is not None:
        IM = AIM.transform
        update_slider_range(IM)
        update_plot()
    else:
        text.insert(tk.END, "Please transform the image first\n")

def find_peak_ranges(x, y, threshold=10, rel_height=0.8, min_width=50):
    # Find peaks
    peaks, _ = find_peaks(y, height=threshold)

    # Calculate prominences
    prominences, _, _ = peak_prominences(y, peaks)

    peak_ranges = []
    for peak, prominence in zip(peaks, prominences):
        # Define a threshold based on the peak height and prominence
        height_threshold = y[peak] - prominence * rel_height

        # Find left boundary
        left = peak
        while left > 0 and y[left] > height_threshold:
            left -= 1

        # Find right boundary
        right = peak
        while right < len(y) - 1 and y[right] > height_threshold:
            right += 1

        # Ensure minimum width
        if right - left < min_width:
            padding = (min_width - (right - left)) // 2
            left = max(0, left - padding)
            right = min(len(y) - 1, right + padding)

        peak_ranges.append((x[left], x[right]))

    return peak_ranges

def _speed():
    global IM, AIM, canvas, transform, text, speed_distribution, radial_coords
    _transform()
    text.insert(tk.END, "speed distribution\n")
    canvas.draw()

    if transform.get() == 'linbasex':
        radial, speed = AIM.radial, AIM.Beta[0]
    elif transform.get() == 'rbasex':
        radial, speed, _ = AIM.distr.rIbeta()
    else:
        radial, speed = abel.tools.vmi.angular_integration_3D(AIM.transform)

    speed_distribution = speed / speed[50:].max()
    radial_coords = radial

    f.clf()
    a = f.add_subplot(111)
    a.plot(radial, speed_distribution)
    a.set_xlabel("Radial coordinate (pixels)")
    a.set_ylabel("Speed distribution (arb. units)")
    a.axis(xmax=500, ymin=-0.05)

    peak_ranges = find_peak_ranges(radial, speed_distribution, threshold=0.1, rel_height=0.5, min_width=5)

    for i, (left, right) in enumerate(peak_ranges):
        a.axvspan(left, right, alpha=0.2, color=f'C{i}')
        a.annotate(f"Peak {i + 1}", ((left + right) / 2, a.get_ylim()[1]), ha='center')

    canvas.draw()

    text.insert(tk.END, f"Prominent peaks found at ranges: {peak_ranges}\n")

def generate_ranges(ranges, step=3):
    result = []

    if step == 0:
        # Generate point ranges where start = end
        for start_range, end_range in ranges:
            current = start_range
            while current <= end_range:
                result.append((current, current))
                current += 1
    else:
        # Original behavior for step > 0
        for start_range, end_range in ranges:
            current = start_range
            while current < end_range:
                result.append((current, current + step))
                current += step

    return result

def generate_rolling_ranges(ranges, step=3):
    result = []
    for start_range, end_range in ranges:
        current = start_range
        while current < end_range:
            result.append((current, current + step))
            current += 1
    return result


def create_save_popup():
    popup = tk.Toplevel()
    popup.title("Save Data")
    popup.geometry("300x150")

    label = tk.Label(popup, text="Would you like to save the plot data?")
    label.pack(pady=20)

    def save_data(data_dict):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                # Write header
                f.write("# Radial_Center\tBeta2\tBeta2_Error\tIntensity\n")
                # Write data
                for i in range(len(data_dict['r_centers'])):
                    f.write(
                        f"{data_dict['r_centers'][i]:.1f}\t{data_dict['beta2_values'][i]:.4f}\t{data_dict['beta2_errors'][i]:.4f}\t{data_dict['intensities'][i]:.4f}\n")
            popup.destroy()

    def no_save():
        popup.destroy()

    # Create frame for buttons
    button_frame = tk.Frame(popup)
    button_frame.pack(pady=20)

    return popup, save_data, no_save, button_frame


def _anisotropy():
    global IM, AIM, canvas, rmin, rmax, transform, text, step_entry, rolling_var, intensity_threshold

    _transform()

    # Get the image dimensions
    height, width = AIM.transform.shape

    # Define the center of the image
    x0 = width / 2
    y0 = height / 2

    # Initialize Anora
    Anni = Anora(AIM.transform, x0, y0)

    # Get parameters from GUI
    r_min = int(rmin.get())
    r_max = int(rmax.get())
    step = int(step_entry.get())
    is_rolling = rolling_var.get()
    intensity_threshold = float(intensity_threshold_multiplier.get())

    # Generate range list based on toggle state
    ranges = [(r_min, r_max)]
    if is_rolling:
        r_range = generate_rolling_ranges(ranges, step=step)
    else:
        r_range = generate_ranges(ranges, step=step)

    # Lists to store results
    r_centers = []
    beta2_values = []
    beta2_errors = []
    intensities = []

    # Process each range
    for min_r, max_r in r_range:
        range_intensity = Anni.get_average_intensity_for_range(min_r, max_r)

        if range_intensity > intensity_threshold*Anni.avg_intensity:
            beta2_fit, beta2_err, theta_deg, W_theta, theta_plot_deg, W_fit = Anni.calculate_beta2(min_r, max_r)
            r_center = (min_r + max_r) / 2

            r_centers.append(r_center)
            beta2_values.append(beta2_fit)
            beta2_errors.append(beta2_err)
            intensities.append(range_intensity)

    # Create plots
    if r_centers:
        # Store data for saving
        data_dict = {
            'r_centers': r_centers,
            'beta2_values': beta2_values,
            'beta2_errors': beta2_errors,
            'intensities': intensities
        }

        # Clear previous plot
        f.clf()
        a = f.add_subplot(111)

        # Plot beta2 vs radius
        a.errorbar(r_centers, beta2_values, yerr=beta2_errors, fmt='o-', capsize=5)
        a.set_xlabel('Radial Position (pixels)')
        a.set_ylabel('Anisotropy Parameter β₂')
        a.set_title(
            f'Anisotropy Parameter vs. Radial Position\nStep size: {step}, {"Rolling" if is_rolling else "Non-rolling"}')
        a.grid(True)

        canvas.draw()

        # Display results in text box
        text.delete(1.0, tk.END)
        text.insert(tk.END, f"Average image intensity: {Anni.avg_intensity:.4f}\n\n")
        text.insert(tk.END, "Results for regions above average intensity:\n")
        text.insert(tk.END, "Radial Center\tβ₂\t\tError\t\tIntensity\n")
        for r_center, beta2, error, intensity in zip(r_centers, beta2_values, beta2_errors, intensities):
            text.insert(tk.END, f"{r_center:.1f}\t\t{beta2:.4f}\t± {error:.4f}\t{intensity:.4f}\n")

        # Create save popup
        popup, save_data, no_save, button_frame = create_save_popup()

        # Add buttons to the popup
        tk.Button(button_frame, text="Yes", command=lambda: save_data(data_dict)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="No", command=no_save).pack(side=tk.LEFT, padx=10)

    else:
        text.delete(1.0, tk.END)
        text.insert(tk.END, "No radial ranges had intensity above the average image intensity.\n")

# def _auto_anisotropy():
#     global IM, AIM, transform, text, speed_distribution, radial_coords
#
#     if speed_distribution is None or radial_coords is None:
#         text.insert(tk.END, "Please run speed distribution first.\n")
#         return
#
#     _transform()
#
#     peak_ranges = find_peak_ranges(radial_coords, speed_distribution)
#
#     n_peaks = len(peak_ranges)
#
#     # Create a new figure in a separate window
#     fig, axes = plt.subplots(n_peaks, 1, figsize=(8, 4 * n_peaks), sharex=True)
#     if n_peaks == 1:
#         axes = [axes]
#
#     for i, (left, right) in enumerate(peak_ranges):
#         rmx = (int(left), int(right))
#         beta, amp, rad, intensity, theta = abel.tools.vmi.radial_integration(AIM.transform, radial_ranges=[rmx, ])
#         beta = beta[0]
#         amp = amp[0]
#
#         axes[i].plot(theta, intensity[0], label="Data")
#         axes[i].plot(theta, PAD(theta, beta[0], amp[0]), '--', label="Fit")
#
#         axes[i].set_title(f"Peak {i + 1}: Range {rmx[0]}-{rmx[1]}")
#         axes[i].set_ylabel("Intensity")
#         axes[i].legend(loc="best")
#
#         text.insert(tk.END, f"Beta for range {rmx[0]}-{rmx[1]}: {beta[0]:.3g}±{beta[1]:.3g}\n")
#
#     axes[-1].set_xlabel("Theta (radians)")
#     fig.suptitle("Automatic Anisotropy Analysis for Detected Peaks")
#     fig.tight_layout()
#     plt.show()

def _quit():
    root.quit()
    root.destroy()

# Tk Block
tk.Button(master=root, text='Load image file', command=_getfilename).pack(anchor=tk.W)
#############################################################################################
frame_for_center1 = tk.Frame(root)
frame_for_center1.pack(anchor=tk.N, expand=True, padx=(230,0) )
tk.Button(master=frame_for_center1, text='center image', command=_center).pack(side=tk.LEFT)

cent = ttk.Combobox(master=frame_for_center1, values=center_methods, state="readonly",
                    width=11, height=len(center_methods))
cent.current(3)
cent.pack(side=tk.LEFT, padx=(110,0))
#############################################################################################
tk.Button(master=root, text='raw image', command=_display).pack(anchor=tk.W)

frame_for_center2 = tk.Frame(root)
frame_for_center2.pack(anchor=tk.N, expand=True, padx=(220,0) )
tk.Button(master=frame_for_center2, text='inverse Abel transform', command=lambda: [_transform(), _display_transformed()]).pack(side=tk.LEFT)

transform = ttk.Combobox(master=frame_for_center2, values=Abel_methods, state="readonly",
                         width=13, height=len(Abel_methods))
transform.current(2)
transform.pack(side=tk.LEFT, padx=(75,0))
#############################################################################################
tk.Button(master=root, text='centered raw image', command=_display_centered).pack(anchor=tk.W)
#############################################################################################
tk.Button(master=root, text='speed distribution', command=_speed).pack(anchor=tk.N)
#############################################################################################
frame = tk.Frame(root)
frame.pack(expand=True, padx=(260,0), pady=10)

# Manual anisotropy button
tk.Button(frame, text='manual anisotropy', command=_anisotropy).grid(row=0, column=0, columnspan=2, sticky=tk.W)
tk.Label(frame, text="", width=8).grid(row=0, column=2)
# rmin & rmax entries
rmin = tk.Entry(frame, width=5)
rmin.grid(row=0, column=4)
rmin.insert(0, 368)

tk.Label(frame, text="to").grid(row=0, column=5, padx=5)

rmax = tk.Entry(frame, width=5)
rmax.grid(row=0, column=6)
rmax.insert(0, 389)

# Rolling toggle
rolling_var = tk.BooleanVar()
rolling_check = tk.Checkbutton(frame, text="Rolling", variable=rolling_var)
rolling_check.grid(row=1, column=4, sticky=tk.W)
# Step size entry
tk.Label(frame, text="Step:").grid(row=1, column=5, padx=(10, 5))
step_entry = tk.Entry(frame, width=5)
step_entry.grid(row=1, column=6)
step_entry.insert(0, "5")

# Intensity threshold multiplier entry
tk.Label(frame, text="Intensity threshold multiplier:").grid(row=2, column=2, columnspan=4, sticky=tk.E)
intensity_threshold_multiplier = tk.Entry(frame, width=5)
intensity_threshold_multiplier.grid(row=2, column=6)
intensity_threshold_multiplier.insert(0, "1")

# Configure grid to maintain spacing
frame.grid_columnconfigure(1, weight=1)  # Add some space after button
frame.grid_columnconfigure(7, weight=1)  # Add some space at the end


tk.Button(master=root, text='Quit', command=_quit).pack(anchor=tk.SW)

vmin_frame = tk.Frame(root)
vmin_frame.pack(fill=tk.X)
tk.Label(vmin_frame, text="vmin:").pack(side=tk.LEFT)
vmin_entry = tk.Entry(vmin_frame, width=10)
vmin_entry.pack(side=tk.LEFT)
vmin_slider = tk.Scale(vmin_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=slider_changed)
vmin_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

vmax_frame = tk.Frame(root)
vmax_frame.pack(fill=tk.X)
tk.Label(vmax_frame, text="vmax:").pack(side=tk.LEFT)
vmax_entry = tk.Entry(vmax_frame, width=10)
vmax_entry.pack(side=tk.LEFT)
vmax_slider = tk.Scale(vmax_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=slider_changed)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

vmin_entry.bind('<Return>', entry_changed)
vmax_entry.bind('<Return>', entry_changed)


canvas = FigureCanvasTkAgg(f, master=root)
a.annotate("load image file", (0.5, 0.6), horizontalalignment="center")
a.annotate("e.g. data/O2-ANU1024.txt.bz2", (0.5, 0.5), horizontalalignment="center")
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

text = tk.Text(master=root, height=4, fg="blue")
text.pack(fill=tk.X)
text.insert(tk.END, "To start load an image data file using the `Load image file' button\n")

tk.mainloop()