import numpy as np
import abel
from scipy.ndimage import shift
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from called_functions.anni import Anora

"""
███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗   
████╗  ██║██╔═████╗██╔══██╗██╔════╝██╔════╝██╗
██╔██╗ ██║██║██╔██║██║  ██║█████╗  ███████╗╚═╝
██║╚██╗██║████╔╝██║██║  ██║██╔══╝  ╚════██║██╗
██║ ╚████║╚██████╔╝██████╔╝███████╗███████║╚═╝
╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝   

██╗   ██╗███████╗██████╗ ████████╗██╗           ██╗
██║   ██║██╔════╝██╔══██╗╚══██╔══╝██║          ███║
██║   ██║█████╗  ██████╔╝   ██║   ██║    █████╗╚██║
╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║   ██║    ╚════╝ ██║
 ╚████╔╝ ███████╗██║  ██║   ██║   ██║           ██║
  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝           ╚═╝

██╗  ██╗ ██████╗ ██████╗ ██╗            ██████╗    
██║  ██║██╔═══██╗██╔══██╗██║            ╚════██╗   
███████║██║   ██║██████╔╝██║             █████╔╝   
██╔══██║██║   ██║██╔══██╗██║            ██╔═══╝    
██║  ██║╚██████╔╝██║  ██║██║            ███████╗   
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝            ╚══════╝   

 ██████╗██╗██████╗  ██████╗██╗         ██████╗     
██╔════╝██║██╔══██╗██╔════╝██║        ██╔═████╗    
██║     ██║██████╔╝██║     ██║        ██║██╔██║    
██║     ██║██╔══██╗██║     ██║        ████╔╝██║    
╚██████╗██║██║  ██║╚██████╗██║        ╚██████╔╝    
 ╚═════╝╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝         ╚═════╝                                                
"""


class MainGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Master GUI PyAbel")
        self.geometry("1200x700")

        # Initialize function parameters
        self.raw_IM = None
        self.centered_IM = None
        self.IM = None
        self.fix_IM = None
        self.AIM = None
        self.inverse_method = None
        self.speed_distribution = None
        self.radial_coords = None

        # Default calibration parameters
        self.cm_per_pixel = 6.808455e-24
        self.ion_mass = 126.90447

        # For color scale memory
        self.last_vmin = None
        self.last_vmax = None

        # Initialize GUI parameters

        # GUI main frame in three rows
        # row 0: top frame (load file, center)
        self.top_frame = tk.Frame(self)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # row 1: middle frame (plot_frame, control_frame)
        self.middle_frame = tk.Frame(self)
        self.middle_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # row 2: bottom frame (text output)
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Configure row/column weights to resize
        self.grid_rowconfigure(1, weight=1)  # vertically
        self.grid_columnconfigure(1, weight=1)  # horizontally

        # Build top frame
        self._build_top_frame()

        # Build middle frame
        # Create and build Pleft plot sub-frame
        self.plot_frame = tk.Frame(self.middle_frame)
        self.plot_frame.pack(side=tk.LEFT, fill="both", expand=True)
        self._build_figure_canvas()
        # Create and build Right controls sub-frame
        self.control_frame = tk.Frame(self.middle_frame)
        self.control_frame.pack(side=tk.RIGHT, fill="y")
        self._build_control_panel()

        # Build bottom frame
        self._build_bottom_frame()

    # Top Frame Builders
    def _build_top_frame(self):
        # Row 0: "Load image" button
        load_button = tk.Button(self.top_frame, text="Load image file",
                                command=self._getfilename)
        load_button.grid(row=0, column=0, padx=5, pady=5)

        # Row 0: "Center image" button
        self.top_frame.grid_columnconfigure(0, minsize=30)
        center_button = tk.Button(self.top_frame, text="Center Image",
                                  command=self._center)
        center_button.grid(row=0, column=1, padx=5, pady=5)

        # Combobox for centering methods
        self.center_methods = ['com', 'convolution', 'gaussian', 'slice']
        self.cent_method = ttk.Combobox(self.top_frame, values=self.center_methods, state="readonly", width=11)
        self.cent_method.current(1)  # default = 'convolution'
        self.cent_method.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Row 0: "Raw Image" button
        self.top_frame.grid_columnconfigure(2, minsize=20)
        display_raw_btn = tk.Button(self.top_frame, text="Raw Image",
                                    command=self._display_raw)
        display_raw_btn.grid(row=0, column=3, padx=5, pady=5)

        # Row 0: "Centered Raw Image" button
        display_centered_btn = tk.Button(self.top_frame, text="Centered Raw Image",
                                         command=self._display_centered)
        display_centered_btn.grid(row=0, column=4, padx=5, pady=5)

        # Row 1: Image fixer toggle
        self.fix_center_artifact = tk.BooleanVar()
        fixer_toggle = tk.Checkbutton(self.top_frame, text="Fix Center Artifact",
                                       variable=self.fix_center_artifact)
        fixer_toggle.grid(row=1, column=1, padx=5, pady=5)

        # Row 1: Abel transform combo + button
        abel_btn = tk.Button(self.top_frame, text="Inverse Abel Transform",
                             command=lambda: [self._transform(), self._image_fixer(),
                                              self._display_transformed()])
        abel_btn.grid(row=1, column=2, padx=5, pady=5)

        self.Abel_methods = [
            'basex', 'direct', 'daun', 'hansenlaw', 'linbasex', 'onion_bordas',
            'onion_peeling', 'rbasex', 'three_point', 'two_point'
        ]
        self.transform_combo = ttk.Combobox(self.top_frame, values=self.Abel_methods, state="readonly", width=13)
        self.transform_combo.current(0)  # default to 'basex'
        self.transform_combo.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Row 1: Speed distribution button
        speed_btn = tk.Button(self.top_frame, text="Speed Distribution",
                              command=self._speed)
        speed_btn.grid(row=1, column=4, padx=5, pady=5)

    # Middle Frame Builders
    def _build_figure_canvas(self):
        # Pleft plot frame
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.annotate("Load an image file", (0.5, 0.6), ha="center")
        self.ax.annotate("via 'Load image file' button above.", (0.5, 0.5), ha="center")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_control_panel(self):
        """
        Right controls:
         - Color Scale Sliders
         - View Controls
         - Energy settings
         - R/KE range
         - Anisotropy button
        """
        # Frame for color scale
        scale_frame = tk.LabelFrame(self.control_frame, text="Color Scale")
        scale_frame.pack(fill="x", padx=5, pady=5)

        # vmin row
        tk.Label(scale_frame, text="vmin:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.vmin_entry = tk.Entry(scale_frame, width=10)
        self.vmin_entry.grid(row=0, column=1, padx=5, pady=2)
        self.vmin_slider = tk.Scale(scale_frame, from_=0, to=255,
                                    orient=tk.HORIZONTAL, command=self._slider_changed)
        self.vmin_slider.grid(row=0, column=2, sticky="we", padx=5, pady=2)
        self.vmin_entry.bind('<Return>', self._entry_changed)

        # vmax row
        tk.Label(scale_frame, text="vmax:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.vmax_entry = tk.Entry(scale_frame, width=10)
        self.vmax_entry.grid(row=1, column=1, padx=5, pady=2)
        self.vmax_slider = tk.Scale(scale_frame, from_=0, to=255,
                                    orient=tk.HORIZONTAL, command=self._slider_changed)
        self.vmax_slider.grid(row=1, column=2, sticky="we", padx=5, pady=2)
        self.vmax_entry.bind('<Return>', self._entry_changed)

        # zoom row
        zoom_pan_frame = tk.LabelFrame(self.control_frame, text="View Control")
        zoom_pan_frame.pack(fill="x", padx=5, pady=5)

        zoom_btn = tk.Button(zoom_pan_frame, text="Zoom", command=self._toggle_zoom)
        zoom_btn.pack(side=tk.LEFT, padx=5, pady=5)

        pan_btn = tk.Button(zoom_pan_frame, text="Pan", command=self._toggle_pan)
        pan_btn.pack(side=tk.LEFT, padx=5, pady=5)

        back_btn = tk.Button(zoom_pan_frame, text="Back", command=self._go_back)
        back_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Energy settings + toggles
        energy_frame = tk.LabelFrame(self.control_frame, text="Energy Conversion")
        energy_frame.pack(fill="x", padx=5, pady=5)

        self.display_in_energy = tk.BooleanVar()
        energy_toggle = tk.Checkbutton(energy_frame, text="Display in Energy",
                                       variable=self.display_in_energy)
        energy_toggle.pack(side=tk.LEFT, padx=5, pady=5)

        settings_button = tk.Button(energy_frame, text="Edit Settings",
                                    command=self._calibration_settings)
        settings_button.pack(side=tk.LEFT, padx=10)

        # Range + Anisotropy frame
        range_frame = tk.LabelFrame(self.control_frame, text="Anisotropy")
        range_frame.pack(fill="x", padx=5, pady=5)

        # Row for "Compute Anisotropy" button
        anisotropy_btn = tk.Button(range_frame, text="Manual Anisotropy", command=self._anisotropy)
        anisotropy_btn.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # R range
        tk.Label(range_frame, text="Px Range:").grid(row=1, column=0, padx=5, pady=2)
        self.rmin = tk.Entry(range_frame, width=5)
        self.rmin.insert(0, "0")
        self.rmin.grid(row=1, column=1, padx=2, pady=2)
        tk.Label(range_frame, text="to").grid(row=1, column=2)
        self.rmax = tk.Entry(range_frame, width=5)
        self.rmax.insert(0, "350")
        self.rmax.grid(row=1, column=3, padx=2, pady=2)

        # KE range
        tk.Label(range_frame, text="KE Range:").grid(row=2, column=0, padx=5, pady=2)
        self.ke_min = tk.Entry(range_frame, width=8)
        self.ke_min.grid(row=2, column=1, padx=2, pady=2)
        tk.Label(range_frame, text="to").grid(row=2, column=2)
        self.ke_max = tk.Entry(range_frame, width=8)
        self.ke_max.grid(row=2, column=3, padx=2, pady=2)

        # Rolling + step
        self.rolling_var = tk.BooleanVar()
        rolling_check = tk.Checkbutton(range_frame, text="Rolling", variable=self.rolling_var)
        rolling_check.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        tk.Label(range_frame, text="Step:").grid(row=3, column=1, sticky="e", padx=5, pady=2)
        self.step_entry = tk.Entry(range_frame, width=5)
        self.step_entry.insert(0, "5")
        self.step_entry.grid(row=3, column=2, padx=2, pady=2)

        # Intensity threshold
        tk.Label(range_frame, text="Intensity threshold:").grid(row=4, column=0, columnspan=2, sticky="e", padx=5)
        self.intensity_threshold_multiplier = tk.Entry(range_frame, width=5)
        self.intensity_threshold_multiplier.insert(0, "1")
        self.intensity_threshold_multiplier.grid(row=4, column=2, padx=5, pady=2, sticky="w")

        # Bind entry events for R / KE
        self.rmin.bind('<KeyRelease>', lambda e: self._update_ke_from_r())
        self.rmax.bind('<KeyRelease>', lambda e: self._update_ke_from_r())
        self.ke_min.bind('<KeyRelease>', lambda e: self._update_r_from_ke())
        self.ke_max.bind('<KeyRelease>', lambda e: self._update_r_from_ke())

    # Bottom Frame Builders
    def _build_bottom_frame(self):
        # Text Box & Save and Quit Buttons
        self.text_box = tk.Text(self.bottom_frame, height=5, fg="blue")
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        quit_btn = tk.Button(self.bottom_frame, text="Quit", command=self._quit)
        quit_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        save_data_btn = tk.Button(self.bottom_frame, text="Save",
                                  command=self._save_current_image_data)
        save_data_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initial text
        self.update_text("To start, load an image data file.\n", clear_first=True)

    def _getfilename(self):
        fn = filedialog.askopenfilename()
        if not fn:
            return

        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, f"Reading image file {fn}\n")
        self.update_text(f"Reading image file {fn}\n", clear_first=True)

        # Load the data
        if ".txt" in fn or ".dat" in fn:
            self.IM = np.loadtxt(fn)
        else:
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, "Error: Unsupported file type")
            self.update_text("Error: Unsupported file type", clear_first=True)

        # Make size odd
        if self.IM.shape[0] % 2 == 0:
            self.text_box.insert(tk.END, "Image is even-sized; shifting to make odd.\n")
            self.update_text("Image is even-sized; shifting to make odd.\n", clear_first=False)
            self.IM = shift(self.IM, (-1, -1))[:-1, :-1]

        # If color (RGB), convert to grayscale
        if len(self.IM.shape) == 3:
            self.IM = np.mean(self.IM, axis=2)

        self.raw_IM = self.IM.copy()
        self.centered_IM = None

        self._update_slider_range(self.IM)
        self._display_raw()

    def _display_raw(self):
        if self.raw_IM is None:
            return
        self.IM = self.raw_IM.copy()
        self._update_slider_range(self.IM)
        self._update_plot()

    def _center(self):
        if self.raw_IM is None:
            self.text_box.insert(tk.END, "No raw image loaded.\n")
            self.update_text("No raw image loaded.\n", clear_first=True)
            return

        method = self.cent_method.get()
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, f"Centering image with {method}\n")
        self.update_text(f"Centering image with {method}\n", clear_first=True)

        self.centered_IM = abel.tools.center.center_image(
            self.raw_IM, method=method, odd_size=True
        )
        self.IM = self.centered_IM.copy()
        self._update_slider_range(self.IM)
        self._update_plot()

    def _display_centered(self):
        if self.centered_IM is not None:
            self.IM = self.centered_IM.copy()
            self._update_slider_range(self.IM)
            self._update_plot()
            self.text_box.insert(tk.END, "Displayed centered raw image.\n")
            self.update_text("Displayed centered raw image.\n", clear_first=False)
        else:
            self.text_box.insert(tk.END, "Please center the image first.\n")
            self.update_text("Please center the image first.\n", clear_first=False)

    def _toggle_zoom(self):
        self.toolbar.zoom()

    def _toggle_pan(self):
        self.toolbar.pan()

    def _go_back(self):
        self.toolbar.home()

    def _transform(self):
        if self.raw_IM is None:
            self.text_box.insert(tk.END, "Sad... No image to transform.\n")
            self.update_text("Sad... No image to transform.\n", clear_first=False)
            return

        self.inverse_method = self.transform_combo.get()
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, f"Inverse Abel transform: {self.inverse_method}\n")
        self.update_text(f"Inverse Abel transform: {self.inverse_method}\n", clear_first=True)
        self.text_box.insert(tk.END, "Transforming...\n")
        self.update_text("Transforming...\n", clear_first=False)

        image_to_transform = self.centered_IM if self.centered_IM is not None else self.raw_IM
        self.AIM = abel.Transform(
            image_to_transform,
            method=self.inverse_method,
            direction="inverse",
            symmetry_axis=None
        )

    def _image_fixer(self):
        if self.AIM is not None:
            self.IM = self.AIM.transform
            center_col = self.IM.shape[1] // 2
            center_row = self.IM.shape[0] // 2
            artifact_width = 10

            # Make an artifact-free image for clean mean and std
            non_artifact = np.delete(self.IM, np.s_[center_col - artifact_width:center_col + artifact_width + 1],
                                     axis=1)
            mean_intensity = np.mean(non_artifact)
            std_intensity = np.std(non_artifact)
            threshold = mean_intensity + 15 * std_intensity

            # Intensity mask for the whole image
            intensity_mask = self.IM > threshold
            # Regional mask for the center column
            region_mask = np.zeros_like(self.IM, dtype=bool)
            region_mask[:, center_col - artifact_width:center_col + artifact_width + 1] = True
            mask = intensity_mask & region_mask
            self.fix_IM = np.copy(self.IM)

            # Define how far to look in each direction
            search_range = 15  # Check a 31x31 area around

            # Radius similarity threshold
            radius_epsilon = 0.1

            for i in range(self.IM.shape[0]):
                for j in range(center_col - artifact_width, center_col + artifact_width + 1):
                    if mask[i, j]:
                        # Calculate radius of this pixel from center
                        y_rel = i - center_row
                        x_rel = j - center_col
                        radius_p = np.sqrt(x_rel ** 2 + y_rel ** 2)

                        # list to store valid neighbors
                        valid_neighbors = []

                        # Check all neighboring pixels within search range
                        for di in range(-search_range, search_range + 1):
                            for dj in range(-search_range, search_range + 1):
                                # Skip the center pixel
                                if di == 0 and dj == 0:
                                    continue

                                # Get neighbor coordinates
                                ni = i + di
                                nj = j + dj

                                # Check if coordinates are within image boundaries
                                if 0 <= ni < self.IM.shape[0] and 0 <= nj < self.IM.shape[1]:
                                    # Calculate radius of this neighbor
                                    ny_rel = ni - center_row
                                    nx_rel = nj - center_col
                                    radius_n = np.sqrt(nx_rel ** 2 + ny_rel ** 2)

                                    # Check if this neighbor is at approximately the same radius
                                    if abs(radius_n - radius_p) < radius_epsilon:
                                        pixel_value = self.IM[ni, nj]

                                        # Only use pixels that are below the threshold
                                        if pixel_value <= 1.5*threshold:
                                            valid_neighbors.append(pixel_value)

                        # If nothing in initial search, then wider search
                        if not valid_neighbors:
                            wider_range = 5

                            for di in range(-wider_range, wider_range + 1):
                                for dj in range(-wider_range, wider_range + 1):
                                    # Skip center
                                    if di == 0 and dj == 0:
                                        continue

                                    ni = i + di
                                    nj = j + dj

                                    if 0 <= ni < self.IM.shape[0] and 0 <= nj < self.IM.shape[1]:
                                        ny_rel = ni - center_row
                                        nx_rel = nj - center_col
                                        radius_n = np.sqrt(nx_rel ** 2 + ny_rel ** 2)

                                        if abs(radius_n - radius_p) < radius_epsilon:
                                            pixel_value = self.IM[ni, nj]
                                            if pixel_value <= 1.5*threshold:
                                                valid_neighbors.append(pixel_value)

                        # Replace pixel value with maximum of valid neighbors
                        if valid_neighbors:
                            self.fix_IM[i, j] = np.max(valid_neighbors)
                        else:
                            self.fix_IM[i, j] = mean_intensity

            return self.fix_IM
        else:
            self.text_box.insert(tk.END, "Please transform the image first.\n")
            self.update_text("Please transform the image first.\n", clear_first=True)

    def _display_transformed(self):
        if self.fix_IM is not None and self.AIM is not None:
            if self.fix_center_artifact.get():
                self.IM = self.fix_IM
                self._update_slider_range(self.fix_IM)
                self._update_plot()
            else:
                self.IM = self.AIM.transform
                self._update_slider_range(self.AIM.transform)
                self._update_plot()

        else:
            self.text_box.insert(tk.END, "Please transform the image first.\n")
            self.update_text("Please transform the image first.\n", clear_first=True)

    def _speed(self):
        self._transform()
        self._image_fixer()
        if self.fix_IM is None:
            return
        if self.AIM is None:
            return

        self.text_box.insert(tk.END, "Computing speed distribution...\n")
        self.update_text("Computing speed distribution...\n", clear_first=False)

        if self.transform_combo.get() == 'linbasex':
            radial, speed = self.AIM.radial, self.AIM.Beta[0]
        elif self.transform_combo.get() == 'rbasex':
            radial, speed, _ = self.AIM.distr.rIbeta()
        else:
            if self.fix_center_artifact.get():
                radial, speed = abel.tools.vmi.angular_integration_3D(self.fix_IM)
            else:
                radial, speed = abel.tools.vmi.angular_integration_3D(self.AIM.transform)

        self.speed_distribution = speed / speed.max()
        self.radial_coords = radial

        # Re-plot
        self.fig.clf()
        ax = self.fig.add_subplot(111)

        peak_ranges = self.find_peak_ranges(radial, self.speed_distribution, threshold=0.5, rel_height=0.5, min_width=5,
                                            prominence_min=0.1)

        if self.display_in_energy.get():
            x_coords = [self._pixel_to_energy(r) for r in radial]
            xlabel = "Energy (cm^-1)"
        else:
            x_coords = radial
            xlabel = "Radial (pixels)"

        for i, (left, right) in enumerate(peak_ranges):
            ax.axvspan(left, right, alpha=0.2, color=f'C{i}')
            if self.display_in_energy.get():
                center_pixel = (left + right) / 2
                center = self._pixel_to_energy(center_pixel)
            else:
                center = (left + right) / 2
            ax.annotate(f"Peak {i + 1}", (center, ax.get_ylim()[1]), ha='center')
            print(f"Peak {i + 1}", center)
            print(f"Peak {i + 1}", left, right)
            self.text_box.insert(tk.END, f"Peak {i + 1} {center}\n")
            self.update_text(f"Peak {i + 1} {center}\n", clear_first=False)

        ax.plot(x_coords, self.speed_distribution)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speed distribution (arb. units)")
        ax.set_title("Speed Distribution")

        self.canvas.draw()

    # Anisotropy
    def _anisotropy(self):
        self._transform()
        self._image_fixer()
        if self.fix_IM is None:
            return
        if self.AIM is None:
            return

        # Prepare Anora
        if self.fix_center_artifact.get():
            img = self.fix_IM
        else:
            img = self.AIM.transform
        height, width = img.shape
        x0, y0 = width / 2, height / 2
        Anni = Anora(img, x0, y0)

        # Inputs
        r_min = int(self.rmin.get())
        r_max = int(self.rmax.get())
        step = int(self.step_entry.get())
        is_rolling = self.rolling_var.get()
        threshold_mult = float(self.intensity_threshold_multiplier.get())
        r_range = [(r_min, r_max)]

        # Generate r ranges
        if is_rolling:
            range_list = self._generate_rolling_ranges(r_range, step)
        else:
            range_list = self._generate_ranges(r_range, step)

        # Lists for storing results
        r_centers = []
        beta2_vals = []
        beta2_errs = []
        intensities = []
        energies = []

        for (start_r, end_r) in range_list:
            rng_intensity = Anni.get_average_intensity_for_range(start_r, end_r)
            if rng_intensity > threshold_mult * Anni.avg_intensity:
                beta2, err, *_ = Anni.calculate_beta2(start_r, end_r)
                center = (start_r + end_r) / 2.0
                r_centers.append(center)
                beta2_vals.append(beta2)
                beta2_errs.append(err)
                intensities.append(rng_intensity)
                # Energy
                energies.append(self._pixel_to_energy(center))

        # Plot results
        self.fig.clf()
        ax = self.fig.add_subplot(111)

        if self.display_in_energy.get():
            x_vals = [self._pixel_to_energy(r) for r in r_centers]
            xlabel = "Energy (cm^-1)"
        else:
            x_vals = r_centers
            xlabel = "Radial Center (pixels)"

        ax.errorbar(x_vals, beta2_vals, yerr=beta2_errs, fmt='o-', capsize=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Anisotropy Parameter β₂")
        ax.set_title(f"Anisotropy (step={step}, rolling={is_rolling})")
        ax.grid(True)
        self.canvas.draw()

        # Show results in the text box
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, f"Average image intensity: {Anni.avg_intensity:.4f}\n")
        self.update_text(f"Average image intensity: {Anni.avg_intensity:.4f}\n", clear_first=True)
        self.text_box.insert(tk.END, "Results for radial slices above threshold:\n")
        self.update_text("Results for radial slices above threshold:\n", clear_first=False)
        if self.display_in_energy.get():
            self.text_box.insert(tk.END, "Energy (cm^-1)\tβ₂\tError\tIntensity\n")
            self.update_text("Energy (cm^-1)\tβ₂\tError\tIntensity\n", clear_first=False)
            for rC, b2, e2, Ival in zip(r_centers, beta2_vals, beta2_errs, intensities):
                E = self._pixel_to_energy(rC)
                self.text_box.insert(tk.END, f"{E:.1f}\t{b2:.4f}\t{e2:.4f}\t{Ival:.4f}\n")
                self.update_text(f"{E:.1f}\t{b2:.4f}\t{e2:.4f}\t{Ival:.4f}\n", clear_first=False)
        else:
            self.text_box.insert(tk.END, "Radial px\tβ₂\tError\tIntensity\n")
            self.update_text("Radial px\tβ₂\tError\tIntensity\n", clear_first=False)
            for rC, b2, e2, Ival in zip(r_centers, beta2_vals, beta2_errs, intensities):
                self.update_text(f"{rC:.1f}\t{b2:.4f}\t{e2:.4f}\t{Ival:.4f}\n", clear_first=False)

            # Save data dictionary for potential saving
        self.anisotropy_data = {
            'r_centers': r_centers,
            'beta2_values': beta2_vals,
            'beta2_errors': beta2_errs,
            'intensities': intensities,
            'energies': energies
        }

        # Create save popup
        self._create_save_popup(self.anisotropy_data)

    # ==========================================================================
    # Utilities
    # ==========================================================================
    def find_peak_ranges(self, x, y, threshold=10, rel_height=0.8, min_width=50, prominence_min=None):
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

    def _generate_ranges(self, ranges, step=3):
        self.r_range_method = 'Non-Rolling'
        self.anni_stepsize = step
        result = []
        if step == 0:
            # Point-by-point analysis (single-pixel analysis)
            for (start_range, end_range) in ranges:
                current = start_range
                while current <= end_range:
                    result.append((current, current))
                    current += 1
        else:
            # Original behavior - analyze in steps
            for (start_range, end_range) in ranges:
                current = start_range
                while current < end_range:
                    result.append((current, current + step))
                    current += step

        return result

    def _generate_rolling_ranges(self, ranges, step=3):
        self.r_range_method = 'Rolling'
        self.anni_stepsize = step
        result = []
        for (start_range, end_range) in ranges:
            current = start_range
            while current < end_range:
                result.append((current, current + step))
                current += 1
        return result

    def _create_save_popup(self, data_dict):
        popup = tk.Toplevel(self)
        popup.title("Save Data")
        popup.geometry("300x150")
        popup.transient(self)
        popup.grab_set()

        label = tk.Label(popup, text="Save the plot data?")
        label.pack(pady=20)

        button_frame = tk.Frame(popup)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Yes",
                  command=lambda: self._save_data(data_dict, popup)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="No",
                  command=popup.destroy).pack(side=tk.LEFT, padx=10)

    def _save_data(self, data_dict, popup):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                # Write header with appropriate inverse abel method, anisotropy method, and units
                f.write(
                    f"Inverse Abel by {self.inverse_method}, {self.r_range_method} Anisotropy with step size of {self.anni_stepsize} pixels\n")
                f.write("# Pixel_Center\t# Energy(cm-1)\tBeta2\tBeta2_Error\tIntensity\n")
                for i in range(len(data_dict['r_centers'])):
                    f.write(
                        f"{data_dict['r_centers'][i]:.1f}\t{data_dict['energies'][i]:.1f}\t{data_dict['beta2_values'][i]:.4f}\t"
                        f"{data_dict['beta2_errors'][i]:.4f}\t{data_dict['intensities'][i]:.4f}\n")
        popup.destroy()

    def _save_current_image_data(self):
        if self.IM is None:
            self.update_text("No image data to save.\n", clear_first=False)
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension='.dat',
            filetypes=[("DAT files", "*.dat"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                np.savetxt(file_path, self.IM)
                self.update_text(f"Image data saved to {file_path}\n", clear_first=False)
            except Exception as e:
                self.update_text(f"Error saving image data: {str(e)}\n", clear_first=False)

    def _update_slider_range(self, data):
        if self.fix_center_artifact.get():
            vmin = np.percentile(data, 0.5)
            if vmin < 0:
                vmin = 0
            vmax = np.percentile(data, 99.99)
            range_size = vmax - vmin

            if range_size <= 2:
                resolution = 0.05
            else:
                resolution = range_size / 100
        else:
            vmin = np.percentile(data, 0.5)
            if vmin < 0:
                vmin = 0
            vmax = np.percentile(data, 99.99)
            range_size = vmax - vmin

            if range_size <= 2:
                resolution = 0.05
            else:
                resolution = range_size / 100

        self.vmin_slider.config(from_=vmin, to=vmax, resolution=resolution)
        self.vmax_slider.config(from_=vmin, to=vmax, resolution=resolution)
        self.vmin_slider.set(vmin)
        self.vmax_slider.set(vmax)

        self.vmin_entry.delete(0, tk.END)
        self.vmin_entry.insert(0, f"{vmin:.2f}")
        self.vmax_entry.delete(0, tk.END)
        self.vmax_entry.insert(0, f"{vmax:.2f}")

    def _slider_changed(self, value):
        vmin = self.vmin_slider.get()
        vmax = self.vmax_slider.get()
        self.vmin_entry.delete(0, tk.END)
        self.vmin_entry.insert(0, f"{vmin:.2f}")
        self.vmax_entry.delete(0, tk.END)
        self.vmax_entry.insert(0, f"{vmax:.2f}")
        self._update_plot()

    def _entry_changed(self, *args):
        try:
            vmin = float(self.vmin_entry.get())
            vmax = float(self.vmax_entry.get())
            self.vmin_slider.set(vmin)
            self.vmax_slider.set(vmax)
            self._update_plot()
        except ValueError:
            pass

    def _update_plot(self, image=None):
        if image is None:
            image = self.IM
        if image is None:
            return

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        vmin = float(self.vmin_entry.get())
        vmax = float(self.vmax_entry.get())
        im = ax.imshow(image, vmin=vmin, vmax=vmax)
        self.fig.colorbar(im)
        self.canvas.draw()

    def update_text(self, text, clear_first=False):
        self.text_box.config(state="normal")
        if clear_first:
            self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, text)
        self.text_box.see(tk.END)
        self.text_box.config(state="disabled")

    def _quit(self):
        self.quit()
        self.destroy()

    # Energy Conversion
    def _calibration_settings(self):
        settings_window = tk.Toplevel(self)
        settings_window.title("Energy Conversion Settings")
        settings_window.geometry("350x200")
        settings_window.transient(self)
        settings_window.grab_set()

        frame = tk.LabelFrame(settings_window, text="Energy Conversion", padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frame, text="1 px =").grid(row=0, column=0, padx=5, pady=5, sticky="e")

        base, exponent = f"{self.cm_per_pixel:e}".split('e')
        base = float(base)
        exponent = int(exponent)

        sci_frame = tk.Frame(frame)
        sci_frame.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        base_entry = tk.Entry(sci_frame, width=8)
        base_entry.pack(side=tk.LEFT)
        base_entry.insert(0, f"{base}")

        tk.Label(sci_frame, text="×10^").pack(side=tk.LEFT)

        exp_entry = tk.Entry(sci_frame, width=4)
        exp_entry.pack(side=tk.LEFT)
        exp_entry.insert(0, str(exponent))

        tk.Label(frame, text=" kg*m/s").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Mass input
        tk.Label(frame, text="Ion Mass:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        mass_entry = tk.Entry(frame, width=10)
        mass_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        mass_entry.insert(0, f"{self.ion_mass:.5f}")
        tk.Label(frame, text="amu").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        def apply_settings():
            try:
                base_val = float(base_entry.get())
                exp_val = int(exp_entry.get())
                self.cm_per_pixel = base_val * (10 ** exp_val)

                self.ion_mass = float(mass_entry.get())

                self._update_ke_from_r()
                settings_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")

        tk.Button(settings_window, text="Apply", command=apply_settings).pack(pady=10)

    def _pixel_to_energy(self, pixel):
        return float(((self.cm_per_pixel * pixel) ** 2) *
                     (5.03411e22) / (2 * (self.ion_mass * 1.6605402e-27)))

    def _energy_to_pixel(self, energy):
        return float(np.sqrt(energy * 2 * (self.ion_mass * 1.6605402e-27) /
                             (5.03411e22)) / self.cm_per_pixel)

    def _update_ke_from_r(self):
        try:
            r_min_val = float(self.rmin.get())
            r_max_val = float(self.rmax.get())
            ke_min_val = self._pixel_to_energy(r_min_val)
            ke_max_val = self._pixel_to_energy(r_max_val)

            self.ke_min.delete(0, tk.END)
            self.ke_max.delete(0, tk.END)
            self.ke_min.insert(0, f"{ke_min_val:.2f}")
            self.ke_max.insert(0, f"{ke_max_val:.2f}")
        except ValueError:
            pass

    def _update_r_from_ke(self):
        try:
            ke_min_val = float(self.ke_min.get())
            ke_max_val = float(self.ke_max.get())

            r_min_val = self._energy_to_pixel(ke_min_val)
            r_max_val = self._energy_to_pixel(ke_max_val)

            self.rmin.delete(0, tk.END)
            self.rmax.delete(0, tk.END)
            self.rmin.insert(0, f"{r_min_val:.0f}")
            self.rmax.insert(0, f"{r_max_val:.0f}")
        except ValueError:
            pass

# Main
if __name__ == "__main__":
    app = MainGUI()
    app.mainloop()
