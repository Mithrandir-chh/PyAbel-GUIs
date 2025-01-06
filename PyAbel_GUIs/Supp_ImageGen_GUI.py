import numpy as np
from scipy.special import legendre
import matplotlib;

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
import abel


class ImageGenerator:
    def __init__(self, n=361):
        self.size = n
        self.center = n // 2

        # Create coordinate grids
        y, x = np.ogrid[:n, :n]
        self.r_pixels = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)
        self.theta = np.arctan2(-(x - self.center), y - self.center)
        self.peaks = []

    def add_peak(self, radius_pixels, intensity, beta2, width_pixels):
        self.peaks.append([radius_pixels, intensity, beta2, width_pixels])

    def clear_peaks(self):
        self.peaks = []

    def generate_peak(self, r0_pixels, width_pixels, intensity, beta2):
        radial = np.exp(-(self.r_pixels - r0_pixels) ** 2 / (2 * width_pixels ** 2))
        angular = 1 + beta2 * legendre(2)(np.cos(self.theta))
        peak = radial * angular

        y, x = np.indices((self.size, self.size))
        R = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)
        Theta = np.arctan2(y - self.center, x - self.center)

        radial_norm = np.exp(-(R - r0_pixels) ** 2 / (2 * width_pixels ** 2))
        angular_norm = 1 + beta2 * legendre(2)(np.cos(Theta))

        volume = np.sum(radial_norm * angular_norm)
        N = intensity / volume

        return N * peak

    def generate_image(self):
        image = np.zeros((self.size, self.size))
        for r0, intensity, beta2, width in self.peaks:
            image += self.generate_peak(r0, width, intensity, beta2)
        return image

    def add_noise(self, image, noise_level=0.01):
        orig_min = image.min()
        orig_max = image.max()

        if orig_min < 0:
            image = image - orig_min

        scaled = (image / image.max()) / noise_level
        scaled = np.maximum(scaled, 1e-10)

        noisy = np.random.poisson(scaled)
        noisy = noisy * noise_level

        if orig_min < 0:
            noisy = noisy + orig_min

        noisy_min = noisy.min()
        noisy_max = noisy.max()
        noisy_norm = (noisy - noisy_min) / (noisy_max - noisy_min)
        matched = noisy_norm * (orig_max - orig_min) + orig_min

        return matched

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
            'noise_added': "Noisy, forward abeled image from added peaks"
        }

        for image_name, img in self.available_images.items():
            if img is not None:
                var = tk.BooleanVar()
                self.checkboxes[image_name] = var
                cb = ttk.Checkbutton(main_frame, variable=var, text=image_name)
                cb.grid(row=row, column=0, sticky="w")
                ttk.Label(main_frame, text=f"({descriptions[image_name]})").grid(row=row, column=1, sticky="w",
                                                                               padx=(10, 0))
                row += 1

        ttk.Label(main_frame, text="Base filename:").grid(row=row, column=0, sticky="w", pady=(20, 0))
        ttk.Entry(main_frame, textvariable=self.filename_var, width=40).grid(row=row + 1, column=0, columnspan=2,
                                                                             sticky="ew")
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row + 2, column=0, columnspan=2, pady=(20, 0))
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)

    def save(self):
        selected = {name: var.get() for name, var in self.checkboxes.items()}
        base_filename = self.filename_var.get()

        # Create a dictionary of filenames for each selected image
        self.result = {}
        for image_name, is_selected in selected.items():
            if is_selected:
                if image_name == 'generated_image':
                    filename = f"{base_filename}.dat"
                elif image_name == 'abel_transformed':
                    filename = f"{base_filename}_f_abel.dat"
                elif image_name == 'noise_added':
                    filename = f"{base_filename}_f_abel_noise.dat"
                else:  # current_image
                    filename = f"{base_filename}_current.dat"
                self.result[image_name] = filename

        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()

class ImageGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Generator GUI")

        self.root.grid_columnconfigure(0, minsize=240, weight=0)

        self.root.grid_columnconfigure(1, weight=1)

        self.root.grid_rowconfigure(0, weight=0)  # no vertical stretch
        self.root.grid_rowconfigure(1, weight=1)  # stretches vertically
################################################################################
        # Initialize image generator
        self.generator = ImageGenerator(n=1000)
        self.current_image = None
        self.generated_image = None
        self.abel_transformed = None
        self.noise_added = None

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

    def create_control_frame(self):
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="5 5 5 5")
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
        self.width_var = tk.StringVar(value="30")
        ttk.Entry(control_frame, textvariable=self.width_var, width=10).grid(row=3, column=1, padx=5, pady=2)

        # Add Peak button
        ttk.Button(control_frame, text="Add Peak", command=self.add_peak_from_inputs).grid(row=4, column=0,
                                                                                           columnspan=2, pady=10)

        # Noise control
        ttk.Label(control_frame, text="Noise Level:").grid(row=5, column=0, padx=5, pady=2)
        self.noise_var = tk.StringVar(value="0.1")
        ttk.Entry(control_frame, textvariable=self.noise_var, width=10).grid(row=5, column=1, padx=5, pady=2)

        # Operation buttons
        ttk.Button(control_frame, text="Generate Image", command=self.generate_image).grid(row=6, column=0,
                                                                                           columnspan=2, pady=5)
        ttk.Button(control_frame, text="Abel Transform", command=self.abel_transform).grid(row=7, column=0,
                                                                                           columnspan=2, pady=5)
        ttk.Button(control_frame, text="Add Noise", command=self.add_noise).grid(row=8, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Save Image", command=self.save_image).grid(row=9, column=0, columnspan=2,
                                                                                   pady=5)

    def create_canvas_frame(self):
        self.canvas_frame = ttk.LabelFrame(self.root, text="Image Display", padding="5 5 5 5")
        self.canvas_frame.grid(row=0, column=1, rowspan=2, padx=(5, 10), pady=5, sticky="nsew")

    def create_peak_list_frame(self):
        peak_frame = ttk.LabelFrame(self.root, text="Peak List & Commands", padding="5 5 5 5")
        peak_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Create text widget and scrollbar
        self.peak_list = tk.Text(peak_frame, height=10, width=30)
        scrollbar = ttk.Scrollbar(peak_frame, orient="vertical", command=self.peak_list.yview)
        self.peak_list.configure(yscrollcommand=scrollbar.set)

        # Command entry
        self.command_entry = tk.Entry(peak_frame)
        self.command_entry.bind('<Return>', self.process_command)

        # Pack widgets
        self.peak_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.command_entry.pack(side=tk.BOTTOM, fill=tk.X)

        # Add help text
        self.peak_list.insert(tk.END, """Available commands:
add r i b w - Add peak (radius, intensity, beta2, width)
del n - Delete peak number n
edit n r i b w - Edit peak n
clear - Clear all peaks
list - List all peaks
help - Show this help
""")
        self.peak_list.configure(state='disabled')

    def add_peak_from_inputs(self):
        try:
            radius = float(self.radius_var.get())
            intensity = float(self.intensity_var.get())
            beta2 = float(self.beta2_var.get())
            width = float(self.width_var.get())

            # Add peak through command interface to maintain consistency
            command = f"add {radius} {intensity} {beta2} {width}"
            self.command_entry.insert(0, command)
            self.process_command()

        except ValueError:
            tk.messagebox.showerror("Error", "Please enter valid numbers")

    def process_command(self, event=None):
        if isinstance(event, str):  # If called directly with a command
            command = event
        else:
            command = self.command_entry.get().strip().lower()
        self.command_entry.delete(0, tk.END)

        try:
            if command == 'help':
                self.peak_list.configure(state='normal')
                self.peak_list.delete(1.0, tk.END)
                self.peak_list.insert(tk.END, """Available commands:
add r i b w - Add peak (radius, intensity, beta2, width)
del n - Delete peak number n
edit n r i b w - Edit peak n
clear - Clear all peaks
list - List all peaks
help - Show this help
""")
                self.peak_list.configure(state='disabled')

            elif command.startswith('add'):
                # add peak command
                parts = command.split()
                if len(parts) == 5:  # add r i b w
                    _, r, i, b, w = parts
                    # Update the input fields
                    self.radius_var.set(r)
                    self.intensity_var.set(i)
                    self.beta2_var.set(b)
                    self.width_var.set(w)
                    # Add the peak
                    self.generator.add_peak(float(r), float(i), float(b), float(w))
                    self.update_peak_list()
                else:
                    raise ValueError("Add command requires 4 parameters: radius intensity beta2 width")

            elif command.startswith('del'):
                # delete peak command
                _, n = command.split()
                n = int(n) - 1
                if 0 <= n < len(self.generator.peaks):
                    self.generator.peaks.pop(n)
                    self.update_peak_list()
                else:
                    raise ValueError("Invalid peak number")

            elif command.startswith('edit'):
                # edit peak command
                _, n, r, i, b, w = command.split()
                n = int(n) - 1  # Convert to 0-based index
                if 0 <= n < len(self.generator.peaks):
                    # Update the input fields with the new values
                    self.radius_var.set(r)
                    self.intensity_var.set(i)
                    self.beta2_var.set(b)
                    self.width_var.set(w)
                    # Update the peak
                    self.generator.peaks[n] = [float(r), float(i), float(b), float(w)]
                    self.update_peak_list()
                else:
                    raise ValueError("Invalid peak number")

            elif command == 'clear':
                # u guessed it, clear peak command
                self.clear_peaks()

            elif command == 'list':
                self.update_peak_list()

            else:
                raise ValueError("Unknown command")

        except (ValueError, IndexError) as e:
            self.peak_list.configure(state='normal')
            self.peak_list.insert(tk.END, f"Error: {str(e)}\n")
            self.peak_list.configure(state='disabled')

    def clear_peaks(self):
        self.generator.clear_peaks()
        self.current_image = None
        self.generated_image = None
        self.abel_transformed = None
        self.noise_added = None
        self.update_plot()
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
            tk.messagebox.showwarning("Warning", "No peaks added yet")
            return

        self.generated_image = self.generator.generate_image()
        self.current_image = self.generated_image
        self.abel_transformed = None
        self.noise_added = None
        self.update_plot()

    def abel_transform(self):
        if self.generated_image is None:
            tk.messagebox.showwarning("Warning", "Generate an image first")
            return

        self.abel_transformed = abel.Transform(self.generated_image,
                                               direction='forward',
                                               method='direct',
                                               verbose=True).transform
        self.current_image = self.abel_transformed
        self.update_plot()

    def add_noise(self):
        if self.abel_transformed is None:
            tk.messagebox.showwarning("Warning", "Generate and forward transform an image first")
            return

        try:
            noise_level = float(self.noise_var.get())
            self.noise_added = self.generator.add_noise(self.abel_transformed, noise_level)
            self.current_image = self.noise_added
            self.update_plot()
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter a valid noise level")


    def update_plot(self, image=None):
        if image is None:
            image = self.current_image
        self.fig.clear()

        if image is not None:
            height, width = image.shape
            aspect_ratio = float(height) / float(width)
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
            tk.messagebox.showwarning("Warning", "Generate an image first")
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
            for image_name, filename in dialog.result.items():
                full_path = os.path.join(save_dir, filename)
                np.savetxt(full_path, dialog.available_images[image_name])
                saved_files.append(filename)

            # Show success message
            if saved_files:
                message = "Saved files:\n" + "\n".join(saved_files)
                tk.messagebox.showinfo("Success", message)


def main():
    root = tk.Tk()
    app = ImageGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()