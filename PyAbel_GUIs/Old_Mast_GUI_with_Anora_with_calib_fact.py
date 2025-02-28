import numpy as np
import abel
from scipy.ndimage import shift
from scipy.signal import find_peaks
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.pyplot import imread
from six.moves import tkinter as tk
from six.moves import tkinter_ttk as ttk
from six.moves import tkinter_tkfiledialog as filedialog
from called_functions.anni import Anora

Abel_methods = ['basex', 'direct', 'hansenlaw', 'linbasex', 'onion_bordas',
                'onion_peeling', 'rbasex', 'three_point', 'two_point']

center_methods = ['com', 'convolution', 'gaussian', 'slice']

root = tk.Tk()
root.wm_title("Master GUI PyAbel")

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)

IM = None
AIM = None
raw_IM = None
centered_IM = None
inverse_method = None
speed_distribution = None
radial_coords = None
anni_method = None
anni_stepsize = None
display_in_energy = None
ke_min = None
ke_max = None


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
    return (3 * x * x - 1) / 2


def PAD(theta, beta, amp):
    return amp * (1 + beta * P2(np.cos(theta)))


def _transform():
    global IM, AIM, raw_IM, centered_IM, canvas, transform, text, inverse_method
    inverse_method = transform.get()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"inverse Abel transform: {inverse_method}\n")
    if "basex" in inverse_method:
        text.insert(tk.END, "  first time calculation of the basis functions may take a while ...\n")
    if "direct" in inverse_method:
        text.insert(tk.END, "   calculation is slowed if Cython unavailable ...\n")
    canvas.draw()

    image_to_transform = centered_IM if centered_IM is not None else raw_IM

    AIM = abel.Transform(image_to_transform, method=inverse_method, direction="inverse", symmetry_axis=None)
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


def find_peak_ranges(x, y, threshold=10, rel_height=0.8, min_width=50, prominence_min=None):
    # Find peaks
    peaks, peak_properties = find_peaks(y, height=threshold, prominence=prominence_min)

    # Calculate prominences
    prominences = peak_properties['prominences']

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


def calibration_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Energy Conversion Settings")
    settings_window.geometry("400x200")

    # Center the window
    settings_window.transient(root)
    settings_window.grab_set()

    # Add labels and entry fields
    conversion_frame = tk.LabelFrame(settings_window, text="Energy Conversion Settings", padx=10, pady=10)
    conversion_frame.pack(padx=10, pady=10, fill="x")

    # Parse current conversion factor into base and exponent
    try:
        current_value = float(cm_per_pixel_entry.get())
        base, exponent = f"{current_value:e}".split('e')
        base = float(base)
        exponent = int(exponent)
    except ValueError:
        base = 1
        exponent = 1

    # Conversion factor input with scientific notation
    tk.Label(conversion_frame, text="1 px =").grid(row=0, column=0, padx=5, pady=5)

    # Frame for scientific notation entry
    sci_frame = tk.Frame(conversion_frame)
    sci_frame.grid(row=0, column=1, padx=5, pady=5)

    base_entry = tk.Entry(sci_frame, width=8)
    base_entry.pack(side=tk.LEFT)
    base_entry.insert(0, f"{base}")

    tk.Label(sci_frame, text="×10^").pack(side=tk.LEFT)

    exp_entry = tk.Entry(sci_frame, width=4)
    exp_entry.pack(side=tk.LEFT)
    exp_entry.insert(0, str(exponent))

    tk.Label(conversion_frame, text="kg*m/s").grid(row=0, column=2, padx=5, pady=5)

    # Mass input
    tk.Label(conversion_frame, text="Ion Mass:").grid(row=1, column=0, padx=5, pady=5)
    mass_entry = tk.Entry(conversion_frame, width=10)
    mass_entry.grid(row=1, column=1, padx=5, pady=5)
    mass_entry.insert(0, "126.90447")  # Default mass for I⁻
    tk.Label(conversion_frame, text="amu").grid(row=1, column=2, padx=5, pady=5)

    def apply_settings():
        try:
            base_val = float(base_entry.get())
            exp_val = int(exp_entry.get())
            conversion_factor = base_val * (10 ** exp_val)

            cm_per_pixel_entry.delete(0, tk.END)
            cm_per_pixel_entry.insert(0, f"{conversion_factor}")

            ion_mass_entry.delete(0, tk.END)
            ion_mass_entry.insert(0, mass_entry.get())

            update_ke_from_r()
            settings_window.destroy()
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter valid numbers")

    tk.Button(settings_window, text="Apply", command=apply_settings).pack(pady=10)


def pixel_to_energy(pixel, momentum_per_pixel_squared, ion_mass):
    """Convert pixel value to energy in cm-1"""
    return float(((momentum_per_pixel_squared * pixel) ** 2) * (5.03411 * 10 ** 22) / (2 * (ion_mass * 1.6605402 * 10 ** -27)))


def energy_to_pixel(energy, momentum_per_pixel_squared, ion_mass):
    """Convert energy in cm-1 to pixel value"""
    return float((np.sqrt(energy * 2 * (ion_mass * 1.6605402 * 10 ** -27) / (5.03411 * 10 ** 22)))/momentum_per_pixel_squared)


def update_ke_from_r():
    """Update KE range boxes based on r range values"""
    try:
        r_min_val = float(rmin.get())
        r_max_val = float(rmax.get())
        cm_per_pixel = float(cm_per_pixel_entry.get())
        mass = float(ion_mass_entry.get())

        ke_min_val = pixel_to_energy(r_min_val, cm_per_pixel, mass)
        ke_max_val = pixel_to_energy(r_max_val, cm_per_pixel, mass)

        ke_min.delete(0, tk.END)
        ke_max.delete(0, tk.END)
        ke_min.insert(0, f"{ke_min_val:.2f}")
        ke_max.insert(0, f"{ke_max_val:.2f}")
    except ValueError:
        pass


def update_r_from_ke():
    """Update r range boxes based on KE range values"""
    try:
        ke_min_val = float(ke_min.get())
        ke_max_val = float(ke_max.get())
        cm_per_pixel = float(cm_per_pixel_entry.get())
        mass = float(ion_mass_entry.get())

        r_min_val = energy_to_pixel(ke_min_val, cm_per_pixel, mass)
        r_max_val = energy_to_pixel(ke_max_val, cm_per_pixel, mass)

        rmin.delete(0, tk.END)
        rmax.delete(0, tk.END)
        rmin.insert(0, f"{r_min_val:.0f}")
        rmax.insert(0, f"{r_max_val:.0f}")
    except ValueError:
        pass


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

    speed_distribution = speed / speed.max()
    radial_coords = radial

    f.clf()
    a = f.add_subplot(111)

    peak_ranges = find_peak_ranges(radial, speed_distribution, threshold=0.5, rel_height=0.5, min_width=5,
                                   prominence_min=0.1)

    # Convert x-axis to energy if toggle is on
    if display_in_energy.get():
        x_coords = [pixel_to_energy(r, float(cm_per_pixel_entry.get()), float(ion_mass_entry.get())) for r in radial]
        xlabel = "Energy (cm^-1)"
    else:
        x_coords = radial
        xlabel = "Radial coordinate (pixels)"

    a.plot(x_coords, speed_distribution)
    a.set_xlabel(xlabel)
    a.set_ylabel("Speed distribution (arb. units)")

    if display_in_energy.get():
        a.axis(xmax=pixel_to_energy(500, float(cm_per_pixel_entry.get()), float(ion_mass_entry.get())), ymin=-0.05)
    else:
        a.axis(xmax=500, ymin=-0.05)

    for i, (left, right) in enumerate(peak_ranges):
        a.axvspan(left, right, alpha=0.2, color=f'C{i}')
        if display_in_energy.get():
            center_pixel = (left + right) / 2
            center = pixel_to_energy(center_pixel, float(cm_per_pixel_entry.get()), float(ion_mass_entry.get()))
        else:
            center = (left + right) / 2
        a.annotate(f"Peak {i + 1}", (center, a.get_ylim()[1]), ha='center')
        print(f"Peak {i + 1}", center)
        print(f"Peak {i + 1}", left, right)

    canvas.draw()

    text.insert(tk.END, f"Prominent peaks found at ranges: {peak_ranges}\n")


def generate_ranges(ranges, step=3):
    global anni_method, anni_stepsize
    anni_method = 'Non-Rolling'
    anni_stepsize = step
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
    global anni_method, anni_stepsize
    result = []
    anni_method = 'Rolling'
    anni_stepsize = step
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
                # Write header with appropriate inverse abel method, anisotropy method, and units
                f.write(
                    f"Inverse Abel by {inverse_method}, {anni_method} Anisotropy with step size of {anni_stepsize} pixels\n")
                f.write("# Pixel_Center\t# Energy(cm-1)\tBeta2\tBeta2_Error\tIntensity\n")
                for i in range(len(data_dict['r_centers'])):
                    f.write(
                        f"{data_dict['r_centers'][i]:.1f}\t{data_dict['energies'][i]:.1f}\t{data_dict['beta2_values'][i]:.4f}\t"
                        f"{data_dict['beta2_errors'][i]:.4f}\t{data_dict['intensities'][i]:.4f}\n")
            popup.destroy()

    def no_save():
        popup.destroy()

    button_frame = tk.Frame(popup)
    button_frame.pack(pady=20)

    return popup, save_data, no_save, button_frame


def _anisotropy():
    global IM, AIM, canvas, rmin, rmax, transform, text, step_entry, rolling_var, intensity_threshold, method

    _transform()

    # Get the image dimensions
    height, width = AIM.transform.shape

    # Define the center of the image
    x0 = width / 2
    y0 = height / 2

    # Initialize Anora
    Anni = Anora(AIM.transform, x0, y0)

    # Get parameters
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
    energies = []

    # Process each range
    for min_r, max_r in r_range:
        range_intensity = Anni.get_average_intensity_for_range(min_r, max_r)

        if range_intensity > intensity_threshold * Anni.avg_intensity:
            beta2_fit, beta2_err, theta_deg, W_theta, theta_plot_deg, W_fit = Anni.calculate_beta2(min_r, max_r)
            r_center = (min_r + max_r) / 2

            r_centers.append(r_center)
            beta2_values.append(beta2_fit)
            beta2_errors.append(beta2_err)
            intensities.append(range_intensity)
            energy = pixel_to_energy(r_center, float(cm_per_pixel_entry.get()), float(ion_mass_entry.get()))
            energies.append(energy)

    # Create plots
    if r_centers:
        # Store data for saving
        data_dict = {
            'r_centers': r_centers,
            'beta2_values': beta2_values,
            'beta2_errors': beta2_errors,
            'intensities': intensities,
            'energies': energies
        }

        # Clear previous plot
        f.clf()
        a = f.add_subplot(111)

        if display_in_energy.get():
            x_coords = [pixel_to_energy(r, float(cm_per_pixel_entry.get()), float(ion_mass_entry.get())) for r in
                        r_centers]
            xlabel = "Energy (cm^-1)"
        else:
            x_coords = r_centers
            xlabel = "Radial Position (pixels)"

            # Plot beta2 vs radius/energy
        a.errorbar(x_coords, beta2_values, yerr=beta2_errors, fmt='o-', capsize=5)
        a.set_xlabel(xlabel)
        a.set_ylabel('Anisotropy Parameter β₂')

        title_suffix = "Rolling" if is_rolling else "Non-rolling"
        if display_in_energy.get():
            a.set_title(f'Anisotropy Parameter vs. Energy\nStep size: {step}, {title_suffix}')
        else:
            a.set_title(f'Anisotropy Parameter vs. Radial Position\nStep size: {step}, {title_suffix}')

        a.grid(True)

        canvas.draw()

        # Display results in text box
        text.delete(1.0, tk.END)
        text.insert(tk.END, f"Average image intensity: {Anni.avg_intensity:.4f}\n\n")
        text.insert(tk.END, "Results for regions above average intensity:\n")

        if display_in_energy.get():
            text.insert(tk.END, "Energy (cm^-1)\tβ₂\t\tError\t\tIntensity\n")
            for r_center, beta2, error, intensity in zip(r_centers, beta2_values, beta2_errors, intensities):
                energy = pixel_to_energy(r_center, float(cm_per_pixel_entry.get()), float(ion_mass_entry.get()))
                text.insert(tk.END, f"{energy:.1f}\t\t{beta2:.4f}\t± {error:.4f}\t{intensity:.4f}\n")
        else:
            text.insert(tk.END, "Radial Center\tβ₂\t\tError\t\tIntensity\n")
            for r_center, beta2, error, intensity in zip(r_centers, beta2_values, beta2_errors, intensities):
                text.insert(tk.END, f"{r_center:.1f}\t\t{beta2:.4f}\t± {error:.4f}\t{intensity:.4f}\n")

        # Create save popup with appropriate data format
        popup, save_data, no_save, button_frame = create_save_popup()

        # Add buttons to the popup
        tk.Button(button_frame, text="Yes", command=lambda: save_data(data_dict)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="No", command=no_save).pack(side=tk.LEFT, padx=10)

    else:
        text.delete(1.0, tk.END)
        text.insert(tk.END, "No radial ranges had intensity above the average image intensity.\n")


def _quit():
    root.quit()
    root.destroy()


# Tk Block
tk.Button(master=root, text='Load image file', command=_getfilename).pack(anchor=tk.W)
#############################################################################################
energy_frame = tk.Frame(root)
energy_frame.pack(anchor=tk.N, expand=True, padx=(230, 0))

# Hidden entry for storing the conversion factor
cm_per_pixel_entry = tk.Entry(root)
default_base = 6.808455
default_exp = -24
cm_per_pixel_entry.insert(0, f"{default_base * (10 ** default_exp)}")
cm_per_pixel_entry.pack_forget()

ion_mass_entry = tk.Entry(root)
ion_mass_entry.insert(0, "126.90447")  # Default value
ion_mass_entry.pack_forget()

# Settings button
settings_button = tk.Button(energy_frame, text="Energy Settings", command=calibration_settings)
settings_button.pack(side=tk.LEFT)

display_in_energy = tk.BooleanVar()
energy_toggle = tk.Checkbutton(energy_frame, text="Display in Energy", variable=display_in_energy)
energy_toggle.pack(side=tk.LEFT, padx=(80, 0))
#############################################################################################
frame_for_center1 = tk.Frame(root)
frame_for_center1.pack(anchor=tk.N, expand=True, padx=(240, 0))
tk.Button(master=frame_for_center1, text='center image', command=_center).pack(side=tk.LEFT)

cent = ttk.Combobox(master=frame_for_center1, values=center_methods, state="readonly",
                    width=11, height=len(center_methods))
cent.current(3)
cent.pack(side=tk.LEFT, padx=(110, 0))
#############################################################################################
tk.Button(master=root, text='raw image', command=_display).pack(anchor=tk.W)

frame_for_center2 = tk.Frame(root)
frame_for_center2.pack(anchor=tk.N, expand=True, padx=(225, 0))
tk.Button(master=frame_for_center2, text='inverse Abel transform',
          command=lambda: [_transform(), _display_transformed()]).pack(side=tk.LEFT)

transform = ttk.Combobox(master=frame_for_center2, values=Abel_methods, state="readonly",
                         width=13, height=len(Abel_methods))
transform.current(2)
transform.pack(side=tk.LEFT, padx=(75, 0))
#############################################################################################
tk.Button(master=root, text='centered raw image', command=_display_centered).pack(anchor=tk.W)
#############################################################################################
tk.Button(master=root, text='speed distribution', command=_speed).pack(anchor=tk.N)
#############################################################################################
frame = tk.Frame(root)
frame.pack(expand=True, padx=(360, 0), pady=10)

tk.Button(frame, text='manual anisotropy', command=_anisotropy).grid(row=0, column=0, columnspan=2, sticky=tk.W)
tk.Label(frame, text="", width=5).grid(row=0, column=2)
# rmin & rmax entries
tk.Label(frame, text="Px Range:").grid(row=0, column=3, padx=5)
rmin = tk.Entry(frame, width=5)
rmin.grid(row=0, column=4)
rmin.insert(0, 368)

tk.Label(frame, text="to").grid(row=0, column=5, padx=5)

rmax = tk.Entry(frame, width=5)
rmax.grid(row=0, column=6)
rmax.insert(0, 389)
#kemin & ke max entries
tk.Label(frame, text="KE Range:").grid(row=1, column=3, padx=5)
ke_min = tk.Entry(frame, width=8)
ke_min.grid(row=1, column=4)
tk.Label(frame, text="to").grid(row=1, column=5, padx=5)
ke_max = tk.Entry(frame, width=8)
ke_max.grid(row=1, column=6)

# Rolling toggle
rolling_var = tk.BooleanVar()
rolling_check = tk.Checkbutton(frame, text="Rolling", variable=rolling_var)
rolling_check.grid(row=2, column=4, sticky=tk.W)
# Step size entry
tk.Label(frame, text="Step:").grid(row=2, column=5, padx=(10, 5))
step_entry = tk.Entry(frame, width=5)
step_entry.grid(row=2, column=6)
step_entry.insert(0, "5")

# Intensity threshold multiplier entry
tk.Label(frame, text="Intensity threshold multiplier:").grid(row=3, column=2, columnspan=4, sticky=tk.E)
intensity_threshold_multiplier = tk.Entry(frame, width=5)
intensity_threshold_multiplier.grid(row=3, column=6)
intensity_threshold_multiplier.insert(0, "1")

# r-ke binding commands
rmin.bind('<KeyRelease>', lambda e: update_ke_from_r())
rmax.bind('<KeyRelease>', lambda e: update_ke_from_r())
ke_min.bind('<KeyRelease>', lambda e: update_r_from_ke())
ke_max.bind('<KeyRelease>', lambda e: update_r_from_ke())

# Configure grid to maintain spacing
frame.grid_columnconfigure(1, weight=1)  # Add some space after button
frame.grid_columnconfigure(7, weight=1)  # Add some space at the end
#############################################################################################

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
a.annotate("e.g. data/2_20_23 Diss 20350 REMPI 32452- 2P3-2_ calib. 25k frames.dat", (0.5, 0.5),
           horizontalalignment="center")
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

text = tk.Text(master=root, height=4, fg="blue")
text.pack(fill=tk.X)
text.insert(tk.END, "To start load an image data file using the `Load image file' button\n")

tk.mainloop()
