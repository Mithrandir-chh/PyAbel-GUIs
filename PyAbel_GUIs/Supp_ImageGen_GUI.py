import numpy as np
from scipy.special import legendre
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import abel

try:
    from .gui_core import (
        DEFAULT_SPHERE_DIAMETER_MM,
        DOT_MAX_COUNT,
        DOT_MAX_SUBDIVISIONS,
        DOT_MIN_COUNT,
        RadialDistortionModel,
        distort_image,
        distort_image_radial,
        export_distribution_dots_glb,
        millimetres_per_pixel_for_diameter,
        peak_amplitude,
        validate_peak_parameters,
    )
    from .scrollable_panel import ScrollablePanel
except ImportError:  # Support direct execution
    from gui_core import (
        DEFAULT_SPHERE_DIAMETER_MM,
        DOT_MAX_COUNT,
        DOT_MAX_SUBDIVISIONS,
        DOT_MIN_COUNT,
        RadialDistortionModel,
        distort_image,
        distort_image_radial,
        export_distribution_dots_glb,
        millimetres_per_pixel_for_diameter,
        peak_amplitude,
        validate_peak_parameters,
    )
    from scrollable_panel import ScrollablePanel


class ImageGenerator:
    def __init__(self, n=361, rng=None):
        if not isinstance(n, int) or n < 3:
            raise ValueError("Image size must be an integer of at least 3 pixels.")
        self.size = n
        # A half-pixel center keeps even-sized images geometrically symmetric.
        self.center = (n - 1) / 2
        self.rng = np.random.default_rng() if rng is None else rng

        # Create coordinate grids
        y, x = np.ogrid[:n, :n]
        self.r_pixels = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)
        self.theta = np.arctan2(-(x - self.center), y - self.center)
        self.peaks = []

    def add_peak(self, radius_pixels, intensity, beta2, width_pixels):
        peak = validate_peak_parameters(
            radius_pixels,
            intensity,
            beta2,
            width_pixels,
            image_size=self.size,
        )
        self.peaks.append(list(peak))

    def clear_peaks(self):
        self.peaks = []

    def generate_peak(self, r0_pixels, width_pixels, intensity, beta2):
        r0_pixels, intensity, beta2, width_pixels = validate_peak_parameters(
            r0_pixels,
            intensity,
            beta2,
            width_pixels,
            image_size=self.size,
        )
        radial = np.exp(-(self.r_pixels - r0_pixels) ** 2 / (2 * width_pixels ** 2))
        angular = 1 + beta2 * legendre(2)(np.cos(self.theta))
        peak = radial * angular

        # The amplitude is shared with the volumetric exporter so that the
        # generated slice and the 3D model it represents stay consistent.
        N = peak_amplitude(
            r0_pixels, width_pixels, intensity, image_size=self.size
        )

        return N * peak

    def generate_image(self):
        image = np.zeros((self.size, self.size))
        for r0, intensity, beta2, width in self.peaks:
            image += self.generate_peak(r0, width, intensity, beta2)
        return image

    def add_noise(self, image, noise_level=0.01):
        image = np.asarray(image, dtype=float)
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            raise ValueError("Image must be a finite two-dimensional array.")
        if not np.isfinite(noise_level) or noise_level < 0:
            raise ValueError("Noise level must be a finite nonnegative number.")
        if noise_level == 0:
            return image.copy()

        # Treat noise_level as the intensity represented by one expected
        # count. Multiplying the Poisson draw back by this scale preserves the
        # input expectation instead of forcing every noisy image to the same
        # min/max range.
        expected_counts = np.clip(image, 0, None) / noise_level
        return self.rng.poisson(expected_counts).astype(float) * noise_level

    def distort_image(self, image, horizontal_scale=1.0, vertical_scale=1.0):
        """Apply a specified elliptical detector distortion to an image."""
        return distort_image(
            image,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
        )

    def distort_radially(self, image, angles, radial_scales):
        """Apply an arbitrary smooth radial distortion profile."""
        return distort_image_radial(image, angles, radial_scales)


class DistortionShapeEditor(ttk.Frame):
    """Interactive radial contour editor with selectable mirror symmetry."""

    MIN_SCALE = 0.65
    MAX_SCALE = 1.35
    AXIS_LABELS = {
        "Horizontal": "↔",
        "Diagonal ↗": "↗",
        "Vertical": "↕",
        "Diagonal ↖": "↖",
    }

    def __init__(self, parent, size=280, model=None):
        super().__init__(parent)
        self.size = int(size)
        self.center = self.size / 2
        self.base_radius = self.size * 0.31
        self.axis_radius = self.size * 0.44
        self.model = (
            RadialDistortionModel(point_count=24) if model is None else model
        )
        self.active_handle = None

        self.canvas = tk.Canvas(
            self,
            width=self.size,
            height=self.size,
            background="white",
            highlightthickness=1,
            highlightbackground="#aaaaaa",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, pady=(4, 0))
        self.symmetry_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Symmetry",
            variable=self.symmetry_var,
            command=self._symmetry_changed,
        ).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reset circle", command=self.reset).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, wraplength=self.size).pack(
            fill=tk.X, pady=(3, 0)
        )
        self.redraw()

    def get_profile(self):
        return self.model.profile()

    def reset(self):
        self.model.reset()
        self.redraw()

    def set_ellipse(self, horizontal_scale, vertical_scale):
        horizontal_scale = float(horizontal_scale)
        vertical_scale = float(vertical_scale)
        if not self.MIN_SCALE <= horizontal_scale <= self.MAX_SCALE:
            raise ValueError(
                f"Interactive horizontal scale must be between "
                f"{self.MIN_SCALE} and {self.MAX_SCALE}."
            )
        if not self.MIN_SCALE <= vertical_scale <= self.MAX_SCALE:
            raise ValueError(
                f"Interactive vertical scale must be between "
                f"{self.MIN_SCALE} and {self.MAX_SCALE}."
            )
        self.model.set_ellipse(horizontal_scale, vertical_scale)
        self.redraw()

    def _shape_points(self):
        points = []
        for angle, scale in zip(self.model.angles, self.model.scales):
            radius = self.base_radius * scale
            points.append(
                (
                    self.center + radius * np.cos(angle),
                    self.center - radius * np.sin(angle),
                )
            )
        return points

    def _axis_segment(self, angle):
        dx = self.axis_radius * np.cos(angle)
        dy = self.axis_radius * np.sin(angle)
        return (
            self.center - dx,
            self.center + dy,
            self.center + dx,
            self.center - dy,
        )

    def redraw(self):
        self.canvas.delete("all")
        c = self.center
        r = self.base_radius
        self.canvas.create_oval(
            c - r,
            c - r,
            c + r,
            c + r,
            outline="#888888",
            dash=(4, 3),
            width=2,
        )

        for name, angle in self.model.SYMMETRY_AXES.items():
            selected = self.symmetry_var.get() and name == self.model.selected_axis
            color = "#1769aa" if selected else "#c0c0c0"
            width = 3 if selected else 1
            self.canvas.create_line(
                *self._axis_segment(angle), fill=color, width=width, tags=("axis",)
            )
            label_radius = self.axis_radius + 8
            self.canvas.create_text(
                c + label_radius * np.cos(angle),
                c - label_radius * np.sin(angle),
                text=self.AXIS_LABELS[name],
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
            )

        points = self._shape_points()
        flat_points = [coordinate for point in points + [points[0]] for coordinate in point]
        self.canvas.create_line(
            *flat_points,
            fill="#d32f2f",
            width=2,
            smooth=True,
            splinesteps=12,
        )
        for index, (x, y) in enumerate(points):
            radius = 4
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#ffffff",
                outline="#d32f2f",
                width=2,
                tags=(f"handle-{index}",),
            )
        self.canvas.create_oval(c - 2, c - 2, c + 2, c + 2, fill="#333333")
        self._update_status()

    def _update_status(self):
        if self.symmetry_var.get():
            self.status_var.set(
                f"Symmetry: {self.model.selected_axis}. Click an axis to change it; "
                "drag red handles radially."
            )
        else:
            self.status_var.set(
                "Symmetry off. Drag red handles radially; dashed circle is undistorted."
            )

    def _symmetry_changed(self):
        self.redraw()

    def _nearest_handle(self, x, y):
        points = np.asarray(self._shape_points())
        distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
        index = int(np.argmin(distances))
        return index if distances[index] <= 10 else None

    def _nearest_axis(self, x, y):
        delta_x = x - self.center
        delta_y = self.center - y
        if np.hypot(delta_x, delta_y) > self.axis_radius:
            return None
        candidates = []
        for name, angle in self.model.SYMMETRY_AXES.items():
            distance = abs(-np.sin(angle) * delta_x + np.cos(angle) * delta_y)
            candidates.append((distance, name))
        distance, name = min(candidates)
        return name if distance <= 6 else None

    def _on_press(self, event):
        self.active_handle = self._nearest_handle(event.x, event.y)
        if self.active_handle is not None:
            self._update_active_handle(event.x, event.y)
            return
        if self.symmetry_var.get():
            axis_name = self._nearest_axis(event.x, event.y)
            if axis_name is not None:
                self.model.set_axis(axis_name)
                self.redraw()

    def _on_drag(self, event):
        if self.active_handle is not None:
            self._update_active_handle(event.x, event.y)

    def _on_release(self, event):
        self.active_handle = None

    def _update_active_handle(self, x, y):
        radius = np.hypot(x - self.center, y - self.center)
        scale = np.clip(
            radius / self.base_radius,
            self.MIN_SCALE,
            self.MAX_SCALE,
        )
        self.model.set_scale(
            self.active_handle,
            scale,
            symmetry=self.symmetry_var.get(),
        )
        self.redraw()


class SaveDialog(tk.Toplevel):
    def __init__(self, parent_gui):
        # Initialize with the root window as the parent
        super().__init__(parent_gui.root)

        self.title("Save Images")
        self.parent_gui = parent_gui

        self.transient(parent_gui.root)
        self.grab_set()

        self.selected_images = {}
        self.filename_var = tk.StringVar()

        self.available_images = {
            'current_image': parent_gui.current_image,
            'generated_image': parent_gui.generated_image,
            'abel_transformed': parent_gui.abel_transformed,
            'distorted_image': parent_gui.distorted_image,
            'noise_added': parent_gui.noise_added
        }

        peaks = parent_gui.generator.peaks
        image_size = parent_gui.generator.size

        peak_parts = []
        for i, (radius, intensity, beta2, width) in enumerate(peaks, 1):
            peak_parts.append(f"p{i}_r{radius:.0f}_i{intensity:.0f}_b{beta2:.1f}_w{width:.0f}")

        if peak_parts:
            default_name = f"{'_'.join(peak_parts)}_size{image_size}"
        else:
            default_name = f"no_peaks_size{image_size}"

        self.filename_var.set(default_name)

        self.create_widgets()

        self.geometry("+%d+%d" % (parent_gui.root.winfo_rootx() + 50,
                                  parent_gui.root.winfo_rooty() + 50))

        self.result = None

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main_frame, text="Select images to save:").grid(row=0, column=0, columnspan=2, sticky="w",
                                                                  pady=(0, 10))

        self.checkboxes = {}
        row = 1
        descriptions = {
            'current_image': "The exact image shown on the right",
            'generated_image': "Clean, inverse abel-esque image with added peaks",
            'abel_transformed': "Clean, forward abeled image from added peaks",
            'distorted_image': "Forward image with the specified detector distortion",
            'noise_added': "Noisy forward image (distorted when enabled)"
        }

        for image_name, img in self.available_images.items():
            if img is not None:
                var = tk.BooleanVar(value=True)
                self.checkboxes[image_name] = var
                cb = ttk.Checkbutton(main_frame, variable=var, text=image_name)
                cb.grid(row=row, column=0, sticky="w")
                ttk.Label(main_frame, text=f"({descriptions[image_name]})").grid(row=row, column=1, sticky="w",
                                                                               padx=(10, 0))
                row += 1

        # Add checkbox for log file
        self.save_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, variable=self.save_log_var, text="Save peak parameters logfile").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
        row += 1

        ttk.Label(main_frame, text="Base filename:").grid(row=row, column=0, sticky="w", pady=(20, 0))
        ttk.Entry(main_frame, textvariable=self.filename_var, width=40).grid(row=row + 1, column=0, columnspan=2,
                                                                             sticky="ew")
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row + 2, column=0, columnspan=2, pady=(20, 0))
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)

    def save_log_file(self, save_dir, log_filename):
        # Create a log file to save peak parameters
        full_path = os.path.join(save_dir, log_filename)

        with open(full_path, 'w') as f:
            f.write(f"Image Size: {self.parent_gui.generator.size}x{self.parent_gui.generator.size}\n")
            if self.parent_gui.noise_added is not None:
                f.write(f"Noise Level: {self.parent_gui.noise_var.get()}\n")
            if getattr(self.parent_gui, 'distorted_image', None) is not None:
                profile = getattr(self.parent_gui, 'distortion_profile', None)
                if profile is not None:
                    angles, scales = profile
                    f.write("Distortion Radial Profile (angle_deg, scale):\n")
                    for angle, scale in zip(np.degrees(angles), scales):
                        f.write(f"  {angle:.1f}, {scale:.6f}\n")
            f.write("\nPeak Parameters:\n")

            for i, (radius, intensity, beta2, width) in enumerate(self.parent_gui.generator.peaks, 1):
                f.write(f"\nPeak {i}:\n")
                f.write(f"  Radius (pixels): {radius:.1f}\n")
                f.write(f"  Intensity: {intensity:.1f}\n")
                f.write(f"  Beta2: {beta2:.1f}\n")
                f.write(f"  Width (pixels): {width:.1f}\n")
            f.write("\nAdd Peak Commands\n")
            for radius, intensity, beta2, width in self.parent_gui.generator.peaks:
                f.write(f"\nadd {radius:.1f} {intensity:.1f} {beta2:.1f} {width:.1f}")

        return log_filename

    def save(self):
        selected = {name: var.get() for name, var in self.checkboxes.items()}
        base_filename = self.filename_var.get().strip()
        if not base_filename:
            messagebox.showerror("Invalid filename", "Base filename cannot be empty.", parent=self)
            return
        if os.path.basename(base_filename) != base_filename:
            messagebox.showerror(
                "Invalid filename",
                "Base filename cannot contain directory separators.",
                parent=self,
            )
            return

        # Create a dictionary of filenames for each selected image
        self.result = {}
        for image_name, is_selected in selected.items():
            if is_selected:
                if image_name == 'generated_image':
                    filename = f"{base_filename}.dat"
                elif image_name == 'abel_transformed':
                    filename = f"{base_filename}_f_abel.dat"
                elif image_name == 'distorted_image':
                    filename = f"{base_filename}_f_abel_distorted.dat"
                elif image_name == 'noise_added':
                    filename = f"{base_filename}_f_abel_noise.dat"
                else:  # current_image
                    filename = f"{base_filename}_current.dat"
                self.result[image_name] = filename

        # Add log file to result if selected
        if self.save_log_var.get():
            self.result['log_file'] = f"{base_filename}_parameters.txt"

        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()

class Model3DDialog(tk.Toplevel):
    """Settings for exporting the synthetic distribution as a glTF dot cloud."""

    DEFAULT_DOTS = 25000
    DEFAULT_DIAMETER = 1.0
    MAX_DIAMETER = 50.0
    MAX_SPHERE_DIAMETER = 1000.0
    # Offering the facet count directly avoids asking for a subdivision level
    # and then explaining what it means.
    DETAIL_CHOICES = {
        "20 facets (smallest file)": 0,
        "80 facets": 1,
        "320 facets (smoothest)": 2,
    }

    def __init__(self, parent_gui):
        super().__init__(parent_gui.root)

        self.title("Export 3D Model")
        self.parent_gui = parent_gui
        self.transient(parent_gui.root.winfo_toplevel())
        self.grab_set()

        self.result = None
        self.dots_var = tk.StringVar(value=str(self.DEFAULT_DOTS))
        self.diameter_var = tk.StringVar(value=f"{self.DEFAULT_DIAMETER:g}")
        self.detail_var = tk.StringVar(value=next(iter(self.DETAIL_CHOICES)))
        self.seed_var = tk.StringVar(value="")
        self.sphere_var = tk.StringVar(value=f"{DEFAULT_SPHERE_DIAMETER_MM:g}")
        self.summary_var = tk.StringVar()

        self.create_widgets()
        for variable in (self.dots_var, self.diameter_var, self.detail_var,
                         self.sphere_var):
            variable.trace_add("write", lambda *_: self.update_summary())
        self.update_summary()
        self.geometry("+%d+%d" % (parent_gui.root.winfo_rootx() + 50,
                                  parent_gui.root.winfo_rooty() + 50))

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            main_frame,
            text=(
                "The model is the cylindrically symmetric distribution whose "
                "central slice is the generated image, drawn as a cloud of "
                "small spheres. Dot density follows the intensity, so the "
                "model carries the whole distribution. Detector distortion and "
                "noise act on the 2D projection only and are not part of it."
            ),
            wraplength=460,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(main_frame, text="Number of dots:").grid(
            row=1, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Entry(main_frame, width=8, textvariable=self.dots_var).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Label(
            main_frame, text=f"{DOT_MIN_COUNT}-{DOT_MAX_COUNT}; sets the file size"
        ).grid(row=1, column=2, sticky="w", padx=5, pady=2)

        ttk.Label(main_frame, text="Dot diameter (pixels):").grid(
            row=2, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Entry(main_frame, width=8, textvariable=self.diameter_var).grid(
            row=2, column=1, padx=5, pady=2
        )
        ttk.Label(
            main_frame, text="larger dots read more solidly, and cost nothing"
        ).grid(row=2, column=2, sticky="w", padx=5, pady=2)

        ttk.Label(main_frame, text="Triangles per dot:").grid(
            row=3, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Combobox(
            main_frame,
            width=20,
            textvariable=self.detail_var,
            values=list(self.DETAIL_CHOICES),
            state="readonly",
        ).grid(row=3, column=1, columnspan=2, sticky="w", padx=5, pady=2)

        ttk.Label(main_frame, text="Sphere diameter (mm):").grid(
            row=4, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Entry(main_frame, width=8, textvariable=self.sphere_var).grid(
            row=4, column=1, padx=5, pady=2
        )
        ttk.Label(
            main_frame, text="the whole sphere, whatever its peaks"
        ).grid(row=4, column=2, sticky="w", padx=5, pady=2)

        ttk.Label(main_frame, text="Random seed:").grid(
            row=5, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Entry(main_frame, width=8, textvariable=self.seed_var).grid(
            row=5, column=1, padx=5, pady=2
        )
        ttk.Label(
            main_frame, text="blank for a different cloud each time"
        ).grid(row=5, column=2, sticky="w", padx=5, pady=2)

        ttk.Label(
            main_frame,
            textvariable=self.summary_var,
            wraplength=460,
            justify=tk.LEFT,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=(15, 0))
        ttk.Button(button_frame, text="Export...", command=self.export).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(
            side=tk.LEFT, padx=5
        )

    def update_summary(self):
        """Describe what the current settings will produce."""
        try:
            dots = int(self.dots_var.get())
            diameter = float(self.diameter_var.get())
            detail = self.DETAIL_CHOICES[self.detail_var.get()]
            sphere = float(self.sphere_var.get())
            if not DOT_MIN_COUNT <= dots <= DOT_MAX_COUNT:
                raise ValueError
            if not 0 < diameter <= self.MAX_DIAMETER:
                raise ValueError
            if not 0 < sphere <= self.MAX_SPHERE_DIAMETER:
                raise ValueError
            scale = millimetres_per_pixel_for_diameter(
                self.parent_gui.generator.peaks, sphere
            )
        except (ValueError, KeyError, tk.TclError):
            self.summary_var.set(
                f"Enter {DOT_MIN_COUNT} to {DOT_MAX_COUNT} dots, a dot diameter "
                f"up to {self.MAX_DIAMETER:g} px, and a sphere diameter up to "
                f"{self.MAX_SPHERE_DIAMETER:g} mm."
            )
            return

        per_dot = 20 * 4 ** detail
        # The sphere is written once, so the file tracks the dot count and is
        # almost independent of how detailed each dot is.
        megabytes = (1024 + dots * 55 + per_dot * 20) / 1e6
        self.summary_var.set(
            f"{dots:,} dots of {diameter:g} px ({diameter * scale:.2f} mm), "
            f"{per_dot} triangles each, about {megabytes:,.1f} MB. The sphere "
            f"will be {sphere:g} mm across, at {scale:.4g} mm per pixel. If the "
            "cloud looks too thin, raise the dot diameter: that costs nothing "
            "in glTF."
        )

    def export(self):
        try:
            count = int(self.dots_var.get())
            if not DOT_MIN_COUNT <= count <= DOT_MAX_COUNT:
                raise ValueError(
                    f"Dot count must be between {DOT_MIN_COUNT} and {DOT_MAX_COUNT}."
                )
            diameter = float(self.diameter_var.get())
            if not 0 < diameter <= self.MAX_DIAMETER:
                raise ValueError(
                    f"Dot diameter must be between 0 and {self.MAX_DIAMETER:g} pixels."
                )
            subdivisions = self.DETAIL_CHOICES[self.detail_var.get()]
            sphere = float(self.sphere_var.get())
            if not 0 < sphere <= self.MAX_SPHERE_DIAMETER:
                raise ValueError(
                    f"Sphere diameter must be between 0 and "
                    f"{self.MAX_SPHERE_DIAMETER:g} millimetres."
                )
            millimetres_per_pixel = millimetres_per_pixel_for_diameter(
                self.parent_gui.generator.peaks, sphere
            )
            seed_text = self.seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
        except (ValueError, KeyError, tk.TclError) as exc:
            messagebox.showerror("Invalid model settings", str(exc), parent=self)
            return

        self.result = {
            "count": count,
            "diameter": diameter,
            "subdivisions": subdivisions,
            "seed": seed,
            "millimetres_per_pixel": millimetres_per_pixel,
        }
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class ImageGeneratorGUI:
    """Image-generator workspace that can use a root window or an app tab."""

    def __init__(self, root, configure_window=True):
        self.root = root
        if configure_window:
            self.root.winfo_toplevel().title("Image Generator GUI")

        self.root.grid_columnconfigure(0, minsize=300, weight=0)

        self.root.grid_columnconfigure(1, weight=1)

        self.root.grid_rowconfigure(0, weight=1)  # stretches vertically
################################################################################
        # Initialize image generator
        self.generator = ImageGenerator(n=800)
        self.current_image = None
        self.generated_image = None
        self.abel_transformed = None
        self.distorted_image = None
        self.distortion_profile = None
        self.noise_added = None
        self.distortion_shape_model = RadialDistortionModel(point_count=24)
        self.distortion_shape_model.set_ellipse(1.0, 0.95)
        self.distortion_editor_window = None
        self.distortion_editor = None

        # Create main frames
        self.create_control_frame()
        self.create_canvas_frame()
        self.create_peak_list_frame()

        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)

        # Attach figure to Tk canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add toolbar below the canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.root.bind('<Configure>', self.on_resize)

    def _invalidate_generated_images(self):
        """Discard images that no longer match the edited peak list."""
        self.current_image = None
        self.generated_image = None
        self.abel_transformed = None
        self.distorted_image = None
        self.distortion_profile = None
        self.noise_added = None
        if hasattr(self, 'fig'):
            self.update_plot()

    def create_control_frame(self):
        # The controls, peak list, and command box share one scrolling column
        # so all of them stay reachable in a short window. The peak list takes
        # any spare height while everything fits.
        self.control_panel = ScrollablePanel(self.root, fill_height=True)
        self.control_panel.grid(row=0, column=0, sticky="nsew")
        self.control_panel.content.grid_columnconfigure(0, weight=1)
        self.control_panel.content.grid_rowconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(self.control_panel.content, text="Controls", padding="5 5 5 5")
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Peak parameters
        ttk.Label(control_frame, text="Radius (pixels):").grid(row=0, column=0, padx=5, pady=2)
        self.radius_var = tk.StringVar(value="200")
        ttk.Entry(control_frame, textvariable=self.radius_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(control_frame, text="Intensity:").grid(row=1, column=0, padx=5, pady=2)
        self.intensity_var = tk.StringVar(value="200.0")
        ttk.Entry(control_frame, textvariable=self.intensity_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(control_frame, text="Beta2:").grid(row=2, column=0, padx=5, pady=2)
        self.beta2_var = tk.StringVar(value="2.0")
        ttk.Entry(control_frame, textvariable=self.beta2_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(control_frame, text="Width (pixels):").grid(row=3, column=0, padx=5, pady=2)
        self.width_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.width_var, width=10).grid(row=3, column=1, padx=5, pady=2)

        # Add Peak button
        ttk.Button(control_frame, text="Add Peak", command=self.add_peak_from_inputs).grid(row=4, column=0,
                                                                                           columnspan=2, pady=10)

        # Noise control
        ttk.Label(control_frame, text="Noise Level:").grid(row=5, column=0, padx=5, pady=2)
        self.noise_var = tk.StringVar(value="1.0")
        ttk.Entry(control_frame, textvariable=self.noise_var, width=10).grid(row=5, column=1, padx=5, pady=2)

        # Detector distortion control. Values below 1 compress an axis and
        # values above 1 stretch it while preserving the image dimensions.
        ttk.Label(control_frame, text="Ellipse X scale:").grid(row=6, column=0, padx=5, pady=2)
        self.horizontal_scale_var = tk.StringVar(value="1.00")
        ttk.Entry(control_frame, textvariable=self.horizontal_scale_var, width=10).grid(
            row=6, column=1, padx=5, pady=2
        )

        ttk.Label(control_frame, text="Ellipse Y scale:").grid(row=7, column=0, padx=5, pady=2)
        self.vertical_scale_var = tk.StringVar(value="0.95")
        ttk.Entry(control_frame, textvariable=self.vertical_scale_var, width=10).grid(
            row=7, column=1, padx=5, pady=2
        )

        # Operation buttons
        ttk.Button(control_frame, text="Generate Image", command=self.generate_image).grid(row=8, column=0,
                                                                                           columnspan=2, pady=5)
        ttk.Button(control_frame, text="Abel Transform", command=self.abel_transform).grid(row=9, column=0,
                                                                                           columnspan=2, pady=5)

        ttk.Button(
            control_frame,
            text="Set X/Y Ellipse",
            command=self.set_ellipse_from_inputs,
        ).grid(row=10, column=0, padx=3, pady=5)
        self.edit_distortion_button = ttk.Button(
            control_frame,
            text="Edit Distortion Shape…",
            command=self.open_distortion_editor,
        )
        self.edit_distortion_button.grid(row=10, column=1, padx=3, pady=5)

        ttk.Button(control_frame, text="Distort Projection", command=self.apply_distortion).grid(
            row=11, column=0, columnspan=2, pady=5
        )
        ttk.Button(control_frame, text="Add Noise", command=self.add_noise).grid(row=12, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Save Image", command=self.save_image).grid(row=13, column=0, columnspan=2,
                                                                                   pady=5)
        ttk.Button(
            control_frame,
            text="Save 3D Model (.glb)",
            command=self.export_3d_model,
        ).grid(row=14, column=0, columnspan=2, pady=5)

    def create_canvas_frame(self):
        self.canvas_frame = ttk.LabelFrame(self.root, text="Image Display", padding="5 5 5 5")
        self.canvas_frame.grid(row=0, column=1, padx=(5, 10), pady=5, sticky="nsew")

    def create_peak_list_frame(self):
        # Create separate frames for peaks and commands
        peak_display_frame = ttk.LabelFrame(self.control_panel.content, text="Peak List", padding="5 5 5 5")
        peak_display_frame.grid(row=1, column=0, padx=5, pady=(5, 0), sticky="nsew")

        command_frame = ttk.LabelFrame(self.control_panel.content, text="Commands", padding="5 5 5 5")
        command_frame.grid(row=2, column=0, padx=5, pady=(5, 5), sticky="nsew")

        # Peak List section
        # Create text widget and scrollbar for peak list
        peak_container = ttk.Frame(peak_display_frame)  # Container for peak list and its scrollbar
        peak_container.pack(fill=tk.BOTH, expand=True)

        self.peak_list = tk.Text(peak_container, height=10, width=30)
        peak_scrollbar = ttk.Scrollbar(peak_container, orient="vertical", command=self.peak_list.yview)
        self.peak_list.configure(yscrollcommand=peak_scrollbar.set)

        self.peak_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        peak_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Command section
        cmd_container = ttk.Frame(command_frame)  # Container for command entry and its scrollbar
        cmd_container.pack(fill=tk.BOTH, expand=True)

        self.command_entry = tk.Text(cmd_container, height=4, width=30)
        cmd_scrollbar = ttk.Scrollbar(cmd_container, orient="vertical", command=self.command_entry.yview)
        self.command_entry.configure(yscrollcommand=cmd_scrollbar.set)

        self.command_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cmd_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Execute button in its own container at the bottom of command frame
        button_container = ttk.Frame(command_frame)
        button_container.pack(fill=tk.X, pady=(5, 0))
        execute_button = ttk.Button(button_container, text="Execute Commands", command=self.process_multiline_commands)
        execute_button.pack(side=tk.RIGHT)

        # Add help text
        self.peak_list.insert(tk.END, f"""Available commands:
    add r i b w - Add peak (radius, 
    intensity, beta2, width)
    del n - Delete peak number n
    edit n r i b w - Edit peak n
    clear - Clear all peaks
    list - List all peaks
    help - Show this help
    
Press Enter for new lines. 

Click Execute or Ctrl+Return to run commands.

Maximum allowed radius: {self.generator.size/2} pixels
    """)
        self.peak_list.configure(state='disabled')

        # Bind Ctrl+Return to execute commands
        self.command_entry.bind('<Control-Return>', self.process_multiline_commands)

    def process_multiline_commands(self, event=None):
        # Get all text from the command entry
        commands = self.command_entry.get("1.0", tk.END)
        # Clear the command entry
        self.command_entry.delete("1.0", tk.END)
        # Process the commands
        self.process_command(commands)
        if event is not None:
            return "break"

    def add_peak_from_inputs(self):
        try:
            radius = float(self.radius_var.get())
            intensity = float(self.intensity_var.get())
            beta2 = float(self.beta2_var.get())
            width = float(self.width_var.get())

            self.generator.add_peak(radius, intensity, beta2, width)
            self._invalidate_generated_images()
            self.update_peak_list()

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def process_command(self, event=None):
        if isinstance(event, str):  # If called directly with a command
            command = event
        else:
            command = self.command_entry.get("1.0", tk.END).strip()
            self.command_entry.delete("1.0", tk.END)  # Clear using text indices

        # Split commands by newlines and process each line
        commands = command.split('\n')

        for command in commands:
            command = command.strip().lower()
            if not command:  # Skip empty lines
                continue

            try:
                if command == 'help':
                    self.peak_list.configure(state='normal')
                    self.peak_list.delete("1.0", tk.END)  # Using text indices
                    self.peak_list.insert(tk.END, f"""Available commands:
    add r i b w - Add peak (radius, 
    intensity, beta2, width)
    del n - Delete peak number n
    edit n r i b w - Edit peak n
    clear - Clear all peaks
    list - List all peaks
    help - Show this help
    
Press Enter for new lines. 

Click Execute or Ctrl+Return to run commands.

Maximum allowed radius: {self.generator.size/2} pixels
    """)
                    self.peak_list.configure(state='disabled')

                elif command.split()[0] == 'add':
                    # add peak command
                    parts = command.split()
                    if len(parts) == 5:  # add r i b w
                        _, r, i, b, w = parts

                        peak = validate_peak_parameters(
                            float(r), float(i), float(b), float(w),
                            image_size=self.generator.size,
                        )
                        # Update the input fields
                        self.radius_var.set(r)
                        self.intensity_var.set(i)
                        self.beta2_var.set(b)
                        self.width_var.set(w)
                        # Add the peak
                        self.generator.add_peak(*peak)
                        self._invalidate_generated_images()
                        self.update_peak_list()
                    else:
                        raise ValueError("Add command requires 4 parameters: radius intensity beta2 width")

                elif command.split()[0] == 'del':
                    # delete peak command
                    _, n = command.split()
                    n = int(n) - 1
                    if 0 <= n < len(self.generator.peaks):
                        self.generator.peaks.pop(n)
                        self._invalidate_generated_images()
                        self.update_peak_list()
                    else:
                        raise ValueError("Invalid peak number")

                elif command.split()[0] == 'edit':
                    # edit peak command
                    _, n, r, i, b, w = command.split()
                    n = int(n) - 1  # Convert to 0-based index
                    if 0 <= n < len(self.generator.peaks):
                        peak = validate_peak_parameters(
                            float(r), float(i), float(b), float(w),
                            image_size=self.generator.size,
                        )
                        # Update the input fields
                        self.radius_var.set(r)
                        self.intensity_var.set(i)
                        self.beta2_var.set(b)
                        self.width_var.set(w)
                        # Update the peak
                        self.generator.peaks[n] = list(peak)
                        self._invalidate_generated_images()
                        self.update_peak_list()
                    else:
                        raise ValueError("Invalid peak number")

                elif command == 'clear':
                    self.clear_peaks()

                elif command == 'list':
                    self.update_peak_list()

                elif command:  # Only show error for non-empty commands
                    raise ValueError("Unknown command")

            except (ValueError, IndexError) as e:
                self.peak_list.configure(state='normal')
                self.peak_list.insert(tk.END, f"Error in command '{command}': {str(e)}\n")
                self.peak_list.configure(state='disabled')

    def clear_peaks(self):
        self.generator.clear_peaks()
        self._invalidate_generated_images()
        self.update_peak_list()

    def update_peak_list(self):
        self.peak_list.configure(state='normal')
        self.peak_list.delete(1.0, tk.END)
        for i, (r, intensity, beta2, width) in enumerate(self.generator.peaks, 1):
            self.peak_list.insert(tk.END,
                                  f"Peak {i}:\n  R: {r:.1f}\n  I: {intensity:.1f}\n  β₂: {beta2:.1f}\n  W: {width:.1f}\n\n")
        self.peak_list.configure(state='disabled')

    def generate_image(self):
        if not self.generator.peaks:
            messagebox.showwarning("Warning", "No peaks added yet")
            return

        try:
            self.generated_image = self.generator.generate_image()
        except ValueError as exc:
            messagebox.showerror("Image generation failed", str(exc))
            return
        self.current_image = self.generated_image
        self.abel_transformed = None
        self.distorted_image = None
        self.distortion_profile = None
        self.noise_added = None
        self.update_plot()

    def abel_transform(self):
        if self.generated_image is None:
            messagebox.showwarning("Warning", "Generate an image first")
            return

        try:
            self.abel_transformed = abel.Transform(
                self.generated_image,
                direction='forward',
                method='direct',
                verbose=False,
            ).transform
        except Exception as exc:
            messagebox.showerror("Abel transform failed", str(exc))
            return
        self.current_image = self.abel_transformed
        self.distorted_image = None
        self.distortion_profile = None
        self.noise_added = None
        self.update_plot()

    def set_ellipse_from_inputs(self):
        try:
            horizontal_scale = float(self.horizontal_scale_var.get())
            vertical_scale = float(self.vertical_scale_var.get())
            if not DistortionShapeEditor.MIN_SCALE <= horizontal_scale <= DistortionShapeEditor.MAX_SCALE:
                raise ValueError(
                    f"Interactive horizontal scale must be between "
                    f"{DistortionShapeEditor.MIN_SCALE} and "
                    f"{DistortionShapeEditor.MAX_SCALE}."
                )
            if not DistortionShapeEditor.MIN_SCALE <= vertical_scale <= DistortionShapeEditor.MAX_SCALE:
                raise ValueError(
                    f"Interactive vertical scale must be between "
                    f"{DistortionShapeEditor.MIN_SCALE} and "
                    f"{DistortionShapeEditor.MAX_SCALE}."
                )
            self.distortion_shape_model.set_ellipse(
                horizontal_scale, vertical_scale
            )
            if self.distortion_editor is not None:
                self.distortion_editor.redraw()
        except ValueError as exc:
            messagebox.showerror("Invalid ellipse", str(exc))

    def open_distortion_editor(self):
        if (
            self.distortion_editor_window is not None
            and self.distortion_editor_window.winfo_exists()
        ):
            self.distortion_editor_window.lift()
            self.distortion_editor_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title("Radial Distortion Shape")
        window.transient(self.root)
        editor = DistortionShapeEditor(
            window,
            size=280,
            model=self.distortion_shape_model,
        )
        editor.pack(padx=8, pady=8)
        ttk.Label(
            window,
            text=(
                "The dashed circle is the undistorted reference. The red contour "
                "defines radial detector magnification versus angle."
            ),
            wraplength=280,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(0, 8))

        self.distortion_editor_window = window
        self.distortion_editor = editor
        self.root.update_idletasks()
        x = self.edit_distortion_button.winfo_rootx()
        y = (
            self.edit_distortion_button.winfo_rooty()
            + self.edit_distortion_button.winfo_height()
            + 5
        )
        window.geometry(f"+{x}+{y}")
        window.protocol("WM_DELETE_WINDOW", self._close_distortion_editor)

    def _close_distortion_editor(self):
        if self.distortion_editor_window is not None:
            self.distortion_editor_window.destroy()
        self.distortion_editor_window = None
        self.distortion_editor = None

    def apply_distortion(self):
        if self.abel_transformed is None:
            messagebox.showwarning(
                "Warning", "Generate and forward transform an image first"
            )
            return

        try:
            angles, radial_scales = self.distortion_shape_model.profile()
            self.distorted_image = self.generator.distort_radially(
                self.abel_transformed,
                angles,
                radial_scales,
            )
        except ValueError as exc:
            messagebox.showerror("Distortion failed", str(exc))
            return

        self.current_image = self.distorted_image
        self.distortion_profile = (angles.copy(), radial_scales.copy())
        self.noise_added = None
        self.update_plot()

    def add_noise(self):
        if self.abel_transformed is None:
            messagebox.showwarning("Warning", "Generate and forward transform an image first")
            return

        try:
            noise_level = float(self.noise_var.get())
            noise_source = (
                self.distorted_image
                if self.distorted_image is not None
                else self.abel_transformed
            )
            self.noise_added = self.generator.add_noise(noise_source, noise_level)
            self.current_image = self.noise_added
            self.update_plot()
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))

    def update_plot(self, image=None):
        if image is None:
            image = self.current_image
        self.fig.clear()

        if image is not None:
            ax = self.fig.add_subplot(111)
            im = ax.imshow(image)
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.fig.tight_layout()
        # Redraw the canvas
        self.canvas.draw()

    def on_resize(self, event):
        if hasattr(self, '_resize_timer'):
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(250, self.update_plot)  # 250ms delay

    def save_image(self):
        # Check if there's anything to save
        if self.current_image is None:
            messagebox.showwarning("Warning", "Generate an image first")
            return

        # Create and wait for dialog
        dialog = SaveDialog(self)
        self.root.wait_window(dialog)

        if dialog.result is not None:
            # Get save directory from user
            save_dir = filedialog.askdirectory()
            if not save_dir:
                return

            # Save each selected image
            saved_files = []
            try:
                for image_name, filename in dialog.result.items():
                    if image_name == 'log_file':
                        log_filename = dialog.save_log_file(save_dir, filename)
                        saved_files.append(log_filename)
                    else:
                        full_path = os.path.join(save_dir, filename)
                        np.savetxt(full_path, dialog.available_images[image_name])
                        saved_files.append(filename)
            except OSError as exc:
                messagebox.showerror("Save failed", str(exc))
                return

            # Show success message
            if saved_files:
                message = "Saved files:\n" + "\n".join(saved_files)
                messagebox.showinfo("Success", message)


    def export_3d_model(self):
        """Write the 3D distribution implied by the peak list as a 3D model."""
        if not self.generator.peaks:
            messagebox.showwarning("Warning", "Add at least one peak first")
            return

        dialog = Model3DDialog(self)
        self.root.wait_window(dialog)
        if dialog.result is None:
            return

        settings = dict(dialog.result)

        peak_parts = [
            f"p{index}_r{radius:.0f}_i{intensity:.0f}_b{beta2:.1f}_w{width:.0f}"
            for index, (radius, intensity, beta2, width)
            in enumerate(self.generator.peaks, 1)
        ]
        default_name = f"{'_'.join(peak_parts)}_size{self.generator.size}_3d.glb"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".glb",
            initialfile=default_name,
            filetypes=[("glTF binary", "*.glb")],
        )
        if not file_path:
            return

        window = self.root.winfo_toplevel()
        window.configure(cursor="watch")
        window.update_idletasks()
        try:
            info = export_distribution_dots_glb(
                file_path,
                self.generator.peaks,
                image_size=self.generator.size,
                name=os.path.splitext(os.path.basename(file_path))[0],
                **settings,
            )
        except (ValueError, OSError) as exc:
            messagebox.showerror("3D export failed", str(exc))
            return
        finally:
            window.configure(cursor="")

        low, high = info["bounds"]
        inner, outer = info["radius_range"]
        messagebox.showinfo(
            "3D model saved",
            f"Saved {os.path.basename(file_path)}\n\n"
            f"{info['dots']:,} dots of {info['diameter']:g} px.\n"
            f"The sphere is stored once as {info['stored_triangles']} "
            f"triangles and placed {info['dots']:,} times, standing in for "
            f"{info['facets']:,} triangles of geometry.\n"
            f"Dots span radius {inner:.1f} to {outer:.1f} px, placed by "
            "density so their spacing carries the intensity.\n"
            f"Size: {high[0] - low[0]:.1f} x {high[1] - low[1]:.1f} x "
            f"{high[2] - low[2]:.1f} mm at "
            f"{info['millimetres_per_pixel']:.4g} mm per pixel "
            "(y is the symmetry axis).\n"
            "Written in metres, as the glTF specification requires.",
        )



def main():
    root = tk.Tk()
    ImageGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
