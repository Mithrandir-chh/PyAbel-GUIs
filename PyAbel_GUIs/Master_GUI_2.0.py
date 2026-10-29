# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
from typing import Optional, Any, List, Dict, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import abel
from scipy.signal import find_peaks  # type: ignore[import-untyped]
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
try:
    from .called_functions.anni import Anora
    from .gui_core import (
        TRANSFORM_METHODS,
        circularize_image,
        extract_speed_distribution,
        inverse_transform,
        prepare_image,
    )
    from .scrollable_panel import ScrollablePanel
except ImportError:  # Support direct execution: python PyAbel_GUIs/Master_GUI_2.0.py
    from called_functions.anni import Anora
    from gui_core import (
        TRANSFORM_METHODS,
        circularize_image,
        extract_speed_distribution,
        inverse_transform,
        prepare_image,
    )
    from scrollable_panel import ScrollablePanel

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


class MainGUI(tk.Frame):
    """PyAbel analysis workspace.

    ``master`` may be a root window for the original standalone GUI or any Tk
    container (for example, a notebook tab) for the combined application.
    """

    def __init__(self, master=None, configure_window=True):
        self._owns_root = master is None
        if master is None:
            master = tk.Tk()
        super().__init__(master)

        if self._owns_root:
            self.pack(fill=tk.BOTH, expand=True)
        if configure_window:
            window = self.winfo_toplevel()
            window.title("Master GUI PyAbel")
            window.geometry("1200x700")

        # Initialize function parameters
        self.raw_IM: Optional[NDArray[np.float64]] = None
        self.centered_IM: Optional[NDArray[np.float64]] = None
        self.circularized_IM: Optional[NDArray[np.float64]] = None
        self.circularization_angles: Optional[NDArray[np.float64]] = None
        self.radial_correction: Optional[NDArray[np.float64]] = None
        self.IM: Optional[NDArray[np.float64]] = None
        self.fix_IM: Optional[NDArray[np.float64]] = None
        self.AIM: Optional[Any] = None  # abel.Transform object
        self.inverse_method: Optional[str] = None
        self.speed_distribution: Optional[NDArray[np.float64]] = None
        self.radial_coords: Optional[NDArray[np.float64]] = None

        # Default calibration parameters
        self.cm_per_pixel: float = 6.808455e-24
        self.ion_mass: float = 126.90447

        # Performance optimization: debounce plot updates
        self._pending_plot_update: Optional[str] = None
        self._pending_speed_plot_update: Optional[str] = None
        self._plot_delay_ms: int = 100  # milliseconds

        # State for coordinated image/result plots. A result curve can be
        # inspected by clicking or dragging it; the matching radius is drawn
        # over the inverse-transformed image.
        self._active_result_view: Optional[str] = None
        self._anisotropy_plot_data: Optional[Dict[str, Any]] = None
        self._linked_image_ax: Optional[Any] = None
        self._linked_result_ax: Optional[Any] = None
        self._linked_image: Optional[NDArray[np.float64]] = None
        self._linked_x_mode: str = "radius"
        self._radius_ring: Optional[Circle] = None
        self._radius_readout: Optional[Any] = None
        self._result_cursor_line: Optional[Any] = None
        self._result_dragging: bool = False
        self._selected_radius: Optional[float] = None

        # Current file tracking
        self.current_filename: Optional[str] = None
        self.save_speed_button: Optional[tk.Button] = None

        # Initialize GUI parameters

        # GUI main frame in three rows
        # row 0: top frame (load file, center)
        self.top_frame: tk.Frame = tk.Frame(self)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # row 1: middle frame (plot_frame, control_frame)
        self.middle_frame: tk.Frame = tk.Frame(self)
        self.middle_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # row 2: bottom frame (text output)
        self.bottom_frame: tk.Frame = tk.Frame(self)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Configure row/column weights to resize
        self.grid_rowconfigure(1, weight=1)  # vertically
        self.grid_columnconfigure(1, weight=1)  # horizontally

        # Build top frame
        self._build_top_frame()

        # Build middle frame
        # Create and build Pleft plot sub-frame
        self.plot_frame: tk.Frame = tk.Frame(self.middle_frame)
        self.plot_frame.pack(side=tk.LEFT, fill="both", expand=True)
        self._build_figure_canvas()
        # Create and build Right controls sub-frame
        self.control_frame: tk.Frame = tk.Frame(self.middle_frame)
        self.control_frame.pack(side=tk.RIGHT, fill="y")
        self._build_control_panel()

        # Build bottom frame
        self._build_bottom_frame()

    # Top Frame Builders
    def _build_top_frame(self) -> None:
        # Row 0: "Load image" button
        load_button: tk.Button = tk.Button(self.top_frame, text="Load image file",
                                command=self._getfilename)
        load_button.grid(row=0, column=0, padx=5, pady=5)

        # Row 0: "Center image" button
        self.top_frame.grid_columnconfigure(0, minsize=30)
        center_button: tk.Button = tk.Button(self.top_frame, text="Center Image",
                                  command=self._center)
        center_button.grid(row=0, column=1, padx=5, pady=5)

        # Combobox for centering methods
        self.center_methods: List[str] = ['com', 'convolution', 'gaussian', 'slice']
        self.cent_method: ttk.Combobox = ttk.Combobox(self.top_frame, values=self.center_methods, state="readonly", width=11)
        self.cent_method.current(1)  # default = 'convolution'
        self.cent_method.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Row 0: "Raw Image" button
        self.top_frame.grid_columnconfigure(2, minsize=20)
        display_raw_btn: tk.Button = tk.Button(self.top_frame, text="Raw Image",
                                    command=self._display_raw)
        display_raw_btn.grid(row=0, column=3, padx=5, pady=5)

        # Row 0: "Centered Raw Image" button
        display_centered_btn: tk.Button = tk.Button(self.top_frame, text="Centered Raw Image",
                                         command=self._display_centered)
        display_centered_btn.grid(row=0, column=4, padx=5, pady=5)

        # Row 1: Image fixer toggle
        self.fix_center_artifact: tk.BooleanVar = tk.BooleanVar()
        fixer_toggle: tk.Checkbutton = tk.Checkbutton(self.top_frame, text="Fix Center Artifact",
                                      variable=self.fix_center_artifact)
        fixer_toggle.grid(row=1, column=0, padx=5, pady=5)

        # Row 1: Abel transform combo + button
        abel_btn: tk.Button = tk.Button(self.top_frame, text="Inverse Abel Transform",
                             command=self._transform_and_display)
        abel_btn.grid(row=1, column=1, padx=5, pady=5)

        self.Abel_methods: List[str] = list(TRANSFORM_METHODS)
        self.transform_combo: ttk.Combobox = ttk.Combobox(self.top_frame, values=self.Abel_methods, state="readonly", width=13)
        self.transform_combo.current(0)  # default to 'basex'
        self.transform_combo.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Row 1: Speed distribution button
        speed_btn: tk.Button = tk.Button(self.top_frame, text="Speed Distribution",
                              command=self._speed)
        speed_btn.grid(row=1, column=3, padx=5, pady=5)

        # Row 1: Speed settings button
        speed_settings_btn: tk.Button = tk.Button(self.top_frame, text="Peak Settings",
                                       command=self._speed_settings)
        speed_settings_btn.grid(row=1, column=4, padx=5, pady=5)

        # Row 2: restore circular ring geometry before Abel inversion.
        circularize_btn: tk.Button = tk.Button(
            self.top_frame,
            text="Circularize / Normalize",
            command=self._circularize,
        )
        circularize_btn.grid(row=2, column=0, padx=5, pady=5)

        self.circularization_method: ttk.Combobox = ttk.Combobox(
            self.top_frame,
            values=["lsq", "argmax"],
            state="readonly",
            width=11,
        )
        self.circularization_method.current(0)
        self.circularization_method.grid(row=2, column=1, padx=5, pady=5)

        radial_frame: tk.Frame = tk.Frame(self.top_frame)
        radial_frame.grid(row=2, column=2, padx=5, pady=5)
        tk.Label(radial_frame, text="Radius:").pack(side=tk.LEFT)
        self.circular_rmin: tk.Entry = tk.Entry(radial_frame, width=5)
        self.circular_rmin.pack(side=tk.LEFT, padx=(3, 1))
        tk.Label(radial_frame, text="to").pack(side=tk.LEFT)
        self.circular_rmax: tk.Entry = tk.Entry(radial_frame, width=5)
        self.circular_rmax.pack(side=tk.LEFT, padx=(1, 0))

        tolerance_frame: tk.Frame = tk.Frame(self.top_frame)
        tolerance_frame.grid(row=2, column=3, padx=5, pady=5)
        tk.Label(tolerance_frame, text="Tolerance:").pack(side=tk.LEFT)
        self.circular_tol: tk.Entry = tk.Entry(tolerance_frame, width=6)
        self.circular_tol.insert(0, "0.0")
        self.circular_tol.pack(side=tk.LEFT, padx=(3, 0))

        display_circularized_btn: tk.Button = tk.Button(
            self.top_frame,
            text="Normalized Image",
            command=self._display_circularized,
        )
        display_circularized_btn.grid(row=2, column=4, padx=5, pady=5)

    # Middle Frame Builders
    def _build_figure_canvas(self) -> None:
        # Pleft plot frame
        # A slightly denser canvas keeps labels and tick text crisp on the
        # macOS Retina display while preserving the responsive layout.
        self.fig: Figure = Figure(figsize=(5, 4), dpi=120)
        self.ax: Any = self.fig.add_subplot(111)
        self.ax.annotate("Load an image file", (0.5, 0.6), ha="center")
        self.ax.annotate("via 'Load image file' button above.", (0.5, 0.5), ha="center")

        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar: NavigationToolbar2Tk = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self._on_result_pointer_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_result_pointer_move)
        self.canvas.mpl_connect("button_release_event", self._on_result_pointer_release)

    def _build_control_panel(self) -> None:
        """
        Right controls:
         - Color Scale Sliders
         - View Controls
         - Energy settings
         - R/KE range
         - Anisotropy button
        """
        self.control_panel: ScrollablePanel = ScrollablePanel(
            self.control_frame, width=365
        )
        self.control_panel.pack(fill=tk.BOTH, expand=True)
        self.control_canvas: tk.Canvas = self.control_panel.canvas
        self.control_content: tk.Frame = self.control_panel.content

        # Frame for color scale
        scale_frame: tk.LabelFrame = tk.LabelFrame(self.control_content, text="Color Scale")
        scale_frame.pack(fill="x", padx=5, pady=5)

        # vmin row
        tk.Label(scale_frame, text="vmin:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.vmin_entry: tk.Entry = tk.Entry(scale_frame, width=10)
        self.vmin_entry.grid(row=0, column=1, padx=5, pady=2)
        self.vmin_slider: tk.Scale = tk.Scale(scale_frame, from_=0, to=255,
                                    orient=tk.HORIZONTAL, command=self._slider_changed)
        self.vmin_slider.grid(row=0, column=2, sticky="we", padx=5, pady=2)
        self.vmin_entry.bind('<Return>', self._entry_changed)

        # vmax row
        tk.Label(scale_frame, text="vmax:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.vmax_entry: tk.Entry = tk.Entry(scale_frame, width=10)
        self.vmax_entry.grid(row=1, column=1, padx=5, pady=2)
        self.vmax_slider: tk.Scale = tk.Scale(scale_frame, from_=0, to=255,
                                    orient=tk.HORIZONTAL, command=self._slider_changed)
        self.vmax_slider.grid(row=1, column=2, sticky="we", padx=5, pady=2)
        self.vmax_entry.bind('<Return>', self._entry_changed)

        # zoom row
        zoom_pan_frame: tk.LabelFrame = tk.LabelFrame(self.control_content, text="View Control")
        zoom_pan_frame.pack(fill="x", padx=5, pady=5)

        zoom_btn: tk.Button = tk.Button(zoom_pan_frame, text="Zoom", command=self._toggle_zoom)
        zoom_btn.pack(side=tk.LEFT, padx=5, pady=5)

        pan_btn: tk.Button = tk.Button(zoom_pan_frame, text="Pan", command=self._toggle_pan)
        pan_btn.pack(side=tk.LEFT, padx=5, pady=5)

        back_btn: tk.Button = tk.Button(zoom_pan_frame, text="Back", command=self._go_back)
        back_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Energy settings + toggles
        energy_frame: tk.LabelFrame = tk.LabelFrame(self.control_content, text="Energy Conversion")
        energy_frame.pack(fill="x", padx=5, pady=5)

        self.display_in_energy: tk.BooleanVar = tk.BooleanVar()
        energy_toggle: tk.Checkbutton = tk.Checkbutton(energy_frame, text="Display in Energy",
                                       variable=self.display_in_energy)
        energy_toggle.pack(side=tk.LEFT, padx=5, pady=5)

        settings_button: tk.Button = tk.Button(energy_frame, text="Edit Settings",
                                    command=self._calibration_settings)
        settings_button.pack(side=tk.LEFT, padx=10)

        # Range + Anisotropy frame
        range_frame: tk.LabelFrame = tk.LabelFrame(self.control_content, text="Anisotropy")
        range_frame.pack(fill="x", padx=5, pady=5)

        # Row for "Compute Anisotropy" button
        anisotropy_btn: tk.Button = tk.Button(range_frame, text="Manual Anisotropy", command=self._anisotropy)
        anisotropy_btn.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # R range
        self.r_range_left_label: tk.Label = tk.Label(range_frame, text="Px Range:")
        self.r_range_left_label.grid(row=1, column=0, padx=5, pady=2)
        self.rmin: tk.Entry = tk.Entry(range_frame, width=5)
        self.rmin.insert(0, "0")
        self.rmin.grid(row=1, column=1, padx=2, pady=2)
        self.r_range_right_label: tk.Label = tk.Label(range_frame, text="to")
        self.r_range_right_label.grid(row=1, column=2)
        self.rmax: tk.Entry = tk.Entry(range_frame, width=5)
        self.rmax.insert(0, "350")
        self.rmax.grid(row=1, column=3, padx=2, pady=2)

        # KE range
        self.KE_range_left_label: tk.Label = tk.Label(range_frame, text="KE Range:")
        self.KE_range_left_label.grid(row=2, column=0, padx=5, pady=2)
        self.ke_min: tk.Entry = tk.Entry(range_frame, width=8)
        self.ke_min.grid(row=2, column=1, padx=2, pady=2)
        self.KE_range_right_label: tk.Label = tk.Label(range_frame, text="to")
        self.KE_range_right_label.grid(row=2, column=2)
        self.ke_max: tk.Entry = tk.Entry(range_frame, width=8)
        self.ke_max.grid(row=2, column=3, padx=2, pady=2)

        # Rolling + step
        self.rolling_var: tk.BooleanVar = tk.BooleanVar()
        rolling_check: tk.Checkbutton = tk.Checkbutton(range_frame, text="Rolling", variable=self.rolling_var)
        rolling_check.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        tk.Label(range_frame, text="Step:").grid(row=3, column=1, sticky="e", padx=5, pady=2)
        self.step_entry: tk.Entry = tk.Entry(range_frame, width=5)
        self.step_entry.insert(0, "5")
        self.step_entry.grid(row=3, column=2, padx=2, pady=2)

        # Intensity threshold
        tk.Label(range_frame, text="Intensity threshold:").grid(row=4, column=0, columnspan=2, sticky="e", padx=5)
        self.intensity_threshold_multiplier: tk.Entry = tk.Entry(range_frame, width=5)
        self.intensity_threshold_multiplier.insert(0, "1")
        self.intensity_threshold_multiplier.grid(row=4, column=2, padx=5, pady=2, sticky="w")

        # New: Anisotropy fitting method
        self.fit_methods_label: tk.Label = tk.Label(range_frame, text="Fit Method:")
        self.fit_methods_label.grid(row=5, column=0, padx=5, pady=2, sticky="e")
        self.fit_methods: List[str] = ['robust', 'RANSAC']
        self.fit_method_combo: ttk.Combobox = ttk.Combobox(range_frame, values=self.fit_methods, state="readonly", width=8)
        self.fit_method_combo.current(0)  # default to 'robust'
        self.fit_method_combo.grid(row=5, column=1, columnspan=2, padx=2, pady=2, sticky="w")

        # New: Show fit plot toggle
        self.show_fit_plot: tk.BooleanVar = tk.BooleanVar(value=False)
        tk.Checkbutton(
            range_frame,
            text="Show representative fit",
            variable=self.show_fit_plot,
        ).grid(row=6, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # New: Advanced RANSAC Parameters frame
        self.ransac_frame: tk.LabelFrame = tk.LabelFrame(self.control_content, text="RANSAC Parameters")
        self.ransac_frame.pack(fill="x", padx=5, pady=5)

        # RANSAC trials
        tk.Label(self.ransac_frame, text="Max trials:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.ransac_trials: tk.Entry = tk.Entry(self.ransac_frame, width=5)
        self.ransac_trials.insert(0, "250")
        self.ransac_trials.grid(row=0, column=1, padx=2, pady=2)

        # RANSAC min samples
        tk.Label(self.ransac_frame, text="Min samples:").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.ransac_min_samples: tk.Entry = tk.Entry(self.ransac_frame, width=5)
        self.ransac_min_samples.insert(0, "25")
        self.ransac_min_samples.grid(row=1, column=1, padx=2, pady=2)

        # RANSAC global threshold
        tk.Label(self.ransac_frame, text="Global threshold:").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        self.ransac_global_thresh: tk.Entry = tk.Entry(self.ransac_frame, width=5)
        self.ransac_global_thresh.insert(0, "3.5")
        self.ransac_global_thresh.grid(row=2, column=1, padx=2, pady=2)

        # RANSAC special threshold
        tk.Label(self.ransac_frame, text="Special threshold:").grid(row=3, column=0, padx=5, pady=2, sticky="e")
        self.ransac_special_thresh: tk.Entry = tk.Entry(self.ransac_frame, width=5)
        self.ransac_special_thresh.insert(0, "2.0")
        self.ransac_special_thresh.grid(row=3, column=1, padx=2, pady=2)

        # RANSAC random seed
        tk.Label(self.ransac_frame, text="Random seed:").grid(
            row=4, column=0, padx=5, pady=2, sticky="e"
        )
        self.ransac_seed: tk.Entry = tk.Entry(self.ransac_frame, width=5)
        self.ransac_seed.insert(0, "0")
        self.ransac_seed.grid(row=4, column=1, padx=2, pady=2)

        # Show/hide RANSAC parameters based on fitting method selection
        self.fit_method_combo.bind("<<ComboboxSelected>>", self._toggle_ransac_frame)

        # Initial state - hide RANSAC frame if not selected
        if self.fit_method_combo.get() != 'RANSAC':
            self.ransac_frame.pack_forget()

        # Bind entry events for R / KE
        self.rmin.bind('<KeyRelease>', lambda e: self._update_ke_from_r())
        self.rmax.bind('<KeyRelease>', lambda e: self._update_ke_from_r())
        self.ke_min.bind('<KeyRelease>', lambda e: self._update_r_from_ke())
        self.ke_max.bind('<KeyRelease>', lambda e: self._update_r_from_ke())

    def _toggle_ransac_frame(self, event: Optional[tk.Event] = None) -> None:
        # Show or hide RANSAC parameters frame based on selection
        if self.fit_method_combo.get() == 'RANSAC':
            self.ransac_frame.pack(fill="x", padx=5, pady=5)
            # The RANSAC panel is the last control, so reveal it.
            self.after_idle(self.control_panel.scroll_to_end)
        else:
            self.ransac_frame.pack_forget()

    # Bottom Frame Builders
    def _build_bottom_frame(self) -> None:
        # Text Box & Save and Quit Buttons
        self.text_box: tk.Text = tk.Text(self.bottom_frame, height=5, fg="blue")
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a frame on the right side of the text display
        right_side_frame: tk.Frame = tk.Frame(self.bottom_frame)
        right_side_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Button row
        button_row: tk.Frame = tk.Frame(right_side_frame)
        button_row.pack(side=tk.TOP)

        save_data_btn: tk.Button = tk.Button(button_row, text="Save",
                                  command=self._save_current_image_data)
        save_data_btn.pack(side=tk.LEFT, padx=5)

        quit_btn: tk.Button = tk.Button(button_row, text="Quit", command=self._quit)
        quit_btn.pack(side=tk.LEFT, padx=5)

        # Version row
        version_row: tk.Frame = tk.Frame(right_side_frame)
        version_row.pack(side=tk.TOP, pady=(5, 0))

        version_label: tk.Label = tk.Label(version_row, text="ver 2.1.02", foreground='#808080')
        version_label.pack()

        # Initial text
        self.update_text("To start, load an image data file.\n", clear_first=True)

    def _getfilename(self) -> None:
        fn: str = filedialog.askopenfilename(
            title="Select Image Data File",
            filetypes=[("Data files", "*.txt *.dat"), ("All files", "*.*")]
        )
        if not fn:
            return

        # Load and validate the data
        if not fn.lower().endswith((".txt", ".dat")):
            self.update_text("Error: Unsupported file type", clear_first=True)
            return

        try:
            loaded = np.loadtxt(fn)
        except OSError as exc:
            self.update_text(f"Error loading image: {exc}\n", clear_first=True)
            return

        self.load_image_array(loaded, source_name=fn)

    def load_image_array(self, image, source_name="In-memory image") -> bool:
        """Load an image array into the analysis workflow.

        This is also the bridge used by PyAbel Studio to pass the image that is
        currently displayed in the generator directly to the analysis tab.
        """
        try:
            loaded = np.asarray(image, dtype=float)
            original_shape = loaded.shape
            prepared = prepare_image(loaded, make_odd=True)
        except (TypeError, ValueError) as exc:
            self.update_text(f"Error loading image: {exc}\n", clear_first=True)
            return False

        self.current_filename = str(source_name)
        self.IM = prepared
        self.update_text(
            f"Loaded image from {self.current_filename}\n", clear_first=True
        )

        if self.IM.shape != original_shape:
            self.update_text(
                f"Cropped even image dimensions from {original_shape} to "
                f"{self.IM.shape} for a centered odd grid.\n",
                clear_first=False,
            )

        self.raw_IM = self.IM.copy()
        self._clear_derived_results()

        self._update_slider_range(self.IM)
        self._display_raw()
        return True

    def _clear_derived_results(self) -> None:
        """Clear results that are no longer valid for the current image."""
        self.centered_IM = None
        self.circularized_IM = None
        self.circularization_angles = None
        self.radial_correction = None
        self._clear_analysis_results()

    def _clear_analysis_results(self) -> None:
        """Clear transforms and plots after source-image preprocessing changes."""
        self.fix_IM = None
        self.AIM = None
        self.inverse_method = None
        self.speed_distribution = None
        self.radial_coords = None
        self._anisotropy_plot_data = None
        self._reset_linked_plot_state()
        if self.save_speed_button is not None:
            self.save_speed_button.destroy()
            self.save_speed_button = None

    def _display_raw(self) -> None:
        if self.raw_IM is None:
            return
        self.IM = self.raw_IM.copy()
        self._update_slider_range(self.IM)
        self._update_plot()

    def _center(self) -> None:
        if self.raw_IM is None:
            self.update_text("No raw image loaded.\n", clear_first=True)
            return

        method: str = self.cent_method.get()
        self.text_box.delete(1.0, tk.END)
        self.update_text(f"Centering image with {method}\n", clear_first=True)

        try:
            centered = abel.tools.center.center_image(
                self.raw_IM, method=method, odd_size=True, square=True
            )
            self.centered_IM = prepare_image(centered, make_odd=True)
        except (ValueError, RuntimeError) as exc:
            self.update_text(f"Centering failed: {exc}\n", clear_first=False)
            return

        self.circularized_IM = None
        self.circularization_angles = None
        self.radial_correction = None
        self._clear_analysis_results()
        self.update_text(f"Finished centering image with {method}\n", clear_first=False)
        self.IM = self.centered_IM.copy()
        self._update_slider_range(self.IM)
        self._update_plot()

    def _display_centered(self) -> None:
        if self.centered_IM is not None:
            self.IM = self.centered_IM.copy()
            self._update_slider_range(self.IM)
            self._update_plot()
            self.update_text("Displayed centered raw image.\n", clear_first=False)
        else:
            self.update_text("Please center the image first.\n", clear_first=False)

    def _circularize(self) -> None:
        """Estimate and remove radial detector distortion from the loaded image."""
        if self.raw_IM is None:
            self.update_text("No raw image loaded.\n", clear_first=True)
            return

        method = self.circularization_method.get()
        try:
            tol = float(self.circular_tol.get())
            rmin_text = self.circular_rmin.get().strip()
            rmax_text = self.circular_rmax.get().strip()
            if bool(rmin_text) != bool(rmax_text):
                raise ValueError("Enter both circularization radii or leave both blank.")
            radial_range = (
                (float(rmin_text), float(rmax_text))
                if rmin_text and rmax_text
                else None
            )

            source = self.centered_IM
            if source is None:
                centering_method = self.cent_method.get()
                self.update_text(
                    f"Centering with {centering_method} before circularization.\n",
                    clear_first=True,
                )
                centered = abel.tools.center.center_image(
                    self.raw_IM,
                    method=centering_method,
                    odd_size=True,
                    square=True,
                )
                self.centered_IM = prepare_image(centered, make_odd=True)
                source = self.centered_IM
            else:
                self.update_text("Circularizing centered image.\n", clear_first=True)

            corrected, angles, factors, _ = circularize_image(
                source,
                method=method,
                radial_range=radial_range,
                tol=tol,
                return_correction=True,
            )
        except Exception as exc:
            self.update_text(f"Circularization failed: {exc}\n", clear_first=False)
            messagebox.showerror("Circularization failed", str(exc))
            return

        self.circularized_IM = corrected
        self.circularization_angles = angles
        self.radial_correction = factors
        self._clear_analysis_results()
        self.IM = corrected.copy()
        self._update_slider_range(self.IM)
        self._update_plot()
        self.update_text(
            f"Finished circularizing with {method}; radial correction range "
            f"{np.min(factors):.4f} to {np.max(factors):.4f}.\n",
            clear_first=False,
        )

    def _display_circularized(self) -> None:
        if self.circularized_IM is None:
            self.update_text(
                "Please circularize/normalize the image first.\n",
                clear_first=False,
            )
            return
        self.IM = self.circularized_IM.copy()
        self._update_slider_range(self.IM)
        self._update_plot()
        self.update_text("Displayed normalized image.\n", clear_first=False)

    def _analysis_source_image(self) -> Optional[NDArray[np.float64]]:
        """Return the most processed valid source image for Abel inversion."""
        if self.circularized_IM is not None:
            return self.circularized_IM
        if self.centered_IM is not None:
            return self.centered_IM
        return self.raw_IM

    def _toggle_zoom(self) -> None:
        self.toolbar.zoom()

    def _toggle_pan(self) -> None:
        self.toolbar.pan()

    def _go_back(self) -> None:
        self.toolbar.home()

    def _transform_and_display(self) -> None:
        if not self._transform():
            return
        self._run_conditional_image_fixer()
        self._display_transformed()

    def _transform(self) -> bool:
        if self.raw_IM is None:
            self.update_text("Sad... No image to transform.\n", clear_first=False)
            return False

        self.inverse_method = self.transform_combo.get()
        self.text_box.delete(1.0, tk.END)
        self.update_text(f"Inverse Abel transform: {self.inverse_method}\n", clear_first=True)
        self.update_text("Transforming...\n", clear_first=False)
        # Enable Anora method entry as default
        self._update_button_state(self.fit_method_combo, disable=False)
        self._update_button_state(self.fit_methods_label, disable=False)
        image_to_transform = self._analysis_source_image()
        if self.inverse_method == "rbasex":
            self._update_button_state(self.fit_method_combo, disable=True)
            self._update_button_state(self.fit_methods_label, disable=True)
            self.fit_method_combo.current(0)
            self.ransac_frame.pack_forget()
        self.fix_IM = None
        try:
            self.AIM = inverse_transform(image_to_transform, self.inverse_method)
        except Exception as exc:
            self.AIM = None
            self.update_text(f"Transform failed: {exc}\n", clear_first=False)
            messagebox.showerror("Transform failed", str(exc))
            return False

        self.update_text("Finished transforming\n", clear_first=False)
        return True

    def _image_fixer(self) -> Optional[NDArray[np.float64]]:
        """Optimized image fixer with vectorized operations"""
        if self.AIM is None:
            self.update_text("Please transform the image first.\n", clear_first=True)
            return None

        self.IM = self.AIM.transform
        center_col: int = self.IM.shape[1] // 2
        center_row: int = self.IM.shape[0] // 2
        artifact_width: int = 20

        # Make an artifact-free image for clean mean and std
        non_artifact: NDArray[np.float64] = np.delete(
            self.IM,
            np.s_[center_col - artifact_width:center_col + artifact_width + 1],
            axis=1
        )
        mean_intensity: float = float(np.mean(non_artifact))
        std_intensity: float = float(np.std(non_artifact))
        threshold: float = mean_intensity + 10 * std_intensity

        # Intensity mask for the whole image
        intensity_mask: NDArray[np.bool_] = self.IM > threshold
        # Regional mask for the center column
        region_mask: NDArray[np.bool_] = np.zeros_like(self.IM, dtype=bool)
        region_mask[:, center_col - artifact_width:center_col + artifact_width + 1] = True
        mask: NDArray[np.bool_] = intensity_mask & region_mask

        self.fix_IM = np.copy(self.IM)

        # Optimized: Use vectorized operations where possible
        # Create coordinate grids
        rows, cols = np.where(mask)

        if len(rows) == 0:
            return self.fix_IM

        # Calculate radii for all pixels at once
        y_grid, x_grid = np.meshgrid(
            np.arange(self.IM.shape[0]) - center_row,
            np.arange(self.IM.shape[1]) - center_col,
            indexing='ij'
        )
        radius_grid: NDArray[np.float64] = np.sqrt(x_grid**2 + y_grid**2)

        search_range: int = 15
        radius_epsilon: float = 0.1

        # Process each artifact pixel (still needs loop but optimized inside)
        for i, j in zip(rows, cols):  # noqa: B905 - NumPy outputs match
            radius_p: float = radius_grid[i, j]

            # Define search window
            i_min: int = max(0, i - search_range)
            i_max: int = min(self.IM.shape[0], i + search_range + 1)
            j_min: int = max(0, j - search_range)
            j_max: int = min(self.IM.shape[1], j + search_range + 1)

            # Get window of radii and intensities
            radius_window: NDArray[np.float64] = radius_grid[i_min:i_max, j_min:j_max]
            intensity_window: NDArray[np.float64] = self.IM[i_min:i_max, j_min:j_max]

            # Find pixels at similar radius
            similar_radius_mask: NDArray[np.bool_] = np.abs(radius_window - radius_p) < radius_epsilon
            below_threshold_mask: NDArray[np.bool_] = intensity_window <= 1.5 * threshold
            valid_mask: NDArray[np.bool_] = similar_radius_mask & below_threshold_mask

            # Exclude center pixel
            center_i: int = i - i_min
            center_j: int = j - j_min
            if 0 <= center_i < valid_mask.shape[0] and 0 <= center_j < valid_mask.shape[1]:
                valid_mask[center_i, center_j] = False

            valid_neighbors: NDArray[np.float64] = intensity_window[valid_mask]

            if len(valid_neighbors) > 0:
                self.fix_IM[i, j] = np.mean(valid_neighbors)
            else:
                # Wider search if needed
                wider_range: int = 30
                i_min_wide: int = max(0, i - wider_range)
                i_max_wide: int = min(self.IM.shape[0], i + wider_range + 1)
                j_min_wide: int = max(0, j - wider_range)
                j_max_wide: int = min(self.IM.shape[1], j + wider_range + 1)

                radius_window_wide: NDArray[np.float64] = radius_grid[i_min_wide:i_max_wide, j_min_wide:j_max_wide]
                intensity_window_wide: NDArray[np.float64] = self.IM[i_min_wide:i_max_wide, j_min_wide:j_max_wide]

                similar_radius_mask_wide: NDArray[np.bool_] = np.abs(radius_window_wide - radius_p) < radius_epsilon
                below_threshold_mask_wide: NDArray[np.bool_] = intensity_window_wide <= 1.5 * threshold
                valid_mask_wide: NDArray[np.bool_] = similar_radius_mask_wide & below_threshold_mask_wide

                center_i_wide: int = i - i_min_wide
                center_j_wide: int = j - j_min_wide
                if (0 <= center_i_wide < valid_mask_wide.shape[0]
                        and 0 <= center_j_wide < valid_mask_wide.shape[1]):
                    valid_mask_wide[center_i_wide, center_j_wide] = False

                valid_neighbors_wide: NDArray[np.float64] = intensity_window_wide[valid_mask_wide]

                if len(valid_neighbors_wide) > 0:
                    self.fix_IM[i, j] = np.mean(valid_neighbors_wide)
                else:
                    self.fix_IM[i, j] = mean_intensity

        return self.fix_IM

    def _display_transformed(self) -> None:
        if self.AIM is not None:
            if self.fix_IM is not None and self.fix_center_artifact.get():
                self.IM = self.fix_IM
                self._update_slider_range(self.fix_IM)
                self._update_plot()
            else:
                self.IM = self.AIM.transform
                self._update_slider_range(self.AIM.transform)
                self._update_plot()

        else:
            self.update_text("Please transform the image first.\n", clear_first=True)

    def _speed(self) -> None:
        if not self._transform() or self.AIM is None:
            return

        speed_image: Optional[NDArray[np.float64]] = None
        if self.fix_center_artifact.get():
            speed_image = self._image_fixer()
            if speed_image is None:
                return

        self.update_text("Computing speed distribution...\n", clear_first=False)
        try:
            self.radial_coords, self.speed_distribution = extract_speed_distribution(
                self.AIM,
                self.transform_combo.get(),
                image=speed_image,
            )
        except ValueError as exc:
            self.update_text(f"Speed-distribution calculation failed: {exc}\n", clear_first=False)
            messagebox.showerror("Speed distribution failed", str(exc))
            return

        # Store radial coordinates and speed
        self.speed_data: Dict[str, Any] = {
            'radial_coords': self.radial_coords,
            'speed_distribution': self.speed_distribution,
            'energies': [self._pixel_to_energy(r) for r in self.radial_coords],
            'transform_method': self.inverse_method
        }

        # Initialize peak finding parameters if not already set
        if not hasattr(self, 'peak_threshold'):
            self.peak_threshold: float = 0.5
        if not hasattr(self, 'peak_rel_height'):
            self.peak_rel_height: float = 0.5
        if not hasattr(self, 'peak_min_width'):
            self.peak_min_width: int = 5
        if not hasattr(self, 'peak_prominence'):
            self.peak_prominence: float = 0.1

        result_image = self._result_image()
        if result_image is not None:
            self._update_slider_range(result_image)

        # Update the plot
        self._pending_speed_plot_update = None
        self._update_speed_plot()

        # Add a button for saving speed distribution data
        if self.save_speed_button is None:
            self.save_speed_button = tk.Button(
                self.plot_frame,
                text="Save Speed Distribution",
                command=self._save_speed_data,
            )
            self.save_speed_button.pack(side=tk.BOTTOM, padx=5, pady=5)

    # Anisotropy
    def _anisotropy(self) -> None:
        if not self._transform() or self.AIM is None:
            return

        if self.fix_center_artifact.get() and self._image_fixer() is None:
            return

        # Get parameters from GUI
        try:
            r_min: int = int(self.rmin.get())
            r_max: int = int(self.rmax.get())
            step: int = int(self.step_entry.get())
            is_rolling: bool = self.rolling_var.get()
            threshold_mult: float = float(self.intensity_threshold_multiplier.get())
            max_radius = min(self.AIM.transform.shape) // 2
            if r_min < 0 or r_max <= r_min or r_max > max_radius:
                raise ValueError(
                    f"Radius range must satisfy 0 <= minimum < maximum <= {max_radius}."
                )
            if step < 0:
                raise ValueError("Step cannot be negative.")
            if threshold_mult < 0 or not np.isfinite(threshold_mult):
                raise ValueError("Intensity threshold must be a finite nonnegative number.")
        except ValueError as exc:
            messagebox.showerror("Invalid anisotropy settings", str(exc))
            self.update_text(f"Invalid anisotropy settings: {exc}\n", clear_first=False)
            return

        # Prepare data structure for results
        r_centers: List[float] = []
        beta2_vals: List[float] = []
        beta2_errs: List[float] = []
        intensities: List[float] = []
        energies: List[float] = []
        a_vals: List[float] = []
        a_errs: List[float] = []
        rejected_beta_count: int = 0

        # Check if using rbasex
        if self.transform_combo.get() == "rbasex":
            # Use Distributions.Ibeta for rbasex
            self.update_text("Using rbasex Ibeta method for anisotropy...\n", clear_first=False)

            # Get radial and beta data from distr
            # Use step as window parameter for smoothing
            radial: NDArray[np.float64]
            intensity: NDArray[np.float64]
            beta2: NDArray[np.float64]
            radial, intensity, beta2 = self.AIM.distr.rIbeta(window=max(1, step))

            # Filter by radius range
            mask: NDArray[np.bool_] = (radial >= r_min) & (radial <= r_max)

            # Get the average intensity for the full radius range
            # for threshold calculation
            avg_intensity: float = np.mean(intensity)
            self.avg_intensity: float = avg_intensity

            if is_rolling:
                # For rolling, we use all points in the range
                filtered_radii: NDArray[np.float64] = radial[mask]
                filtered_beta2: NDArray[np.float64] = beta2[mask]
                filtered_intensity: NDArray[np.float64] = intensity[mask]
                self.r_range_method: str = 'Rolling'
                self.anni_stepsize: int = step

                # Store all data points
                for i, r in enumerate(filtered_radii):
                    if filtered_intensity[i] > threshold_mult * self.avg_intensity:
                        if not np.isfinite(filtered_beta2[i]) or not -1 <= filtered_beta2[i] <= 2:
                            rejected_beta_count += 1
                            continue
                        r_centers.append(r)
                        beta2_vals.append(filtered_beta2[i])
                        beta2_errs.append(0.0)  # No error calculation in rbasex method
                        intensities.append(filtered_intensity[i])
                        energies.append(self._pixel_to_energy(r))
                        a_vals.append(0.0)  # Not provided by rbasex
                        a_errs.append(0.0)  # Not provided by rbasex
            elif step == 0:
                # Point-by-point analysis without radial averaging.
                self.r_range_method = 'Point-by-Point'
                self.anni_stepsize = 0
                for r, range_intensity, range_beta in zip(  # noqa: B905
                    radial[mask], intensity[mask], beta2[mask]
                ):
                    if range_intensity > threshold_mult * self.avg_intensity:
                        if not np.isfinite(range_beta) or not -1 <= range_beta <= 2:
                            rejected_beta_count += 1
                            continue
                        r_centers.append(float(r))
                        beta2_vals.append(float(range_beta))
                        beta2_errs.append(0.0)
                        intensities.append(float(range_intensity))
                        energies.append(self._pixel_to_energy(float(r)))
                        a_vals.append(0.0)
                        a_errs.append(0.0)
            else:
                # For non-rolling, we use steps
                self.r_range_method: str = 'Non-Rolling'
                self.anni_stepsize: int = step
                for r_start in range(r_min, r_max, step):
                    r_end: int = min(r_start + step, r_max)
                    r_center: float = (r_start + r_end) / 2.0

                    # Find indices in the range
                    range_mask: NDArray[np.bool_] = (radial >= r_start) & (radial <= r_end)
                    if not np.any(range_mask):
                        continue

                    # Calculate average intensity and beta in the range
                    range_intensity: float = np.mean(intensity[range_mask])
                    if range_intensity > threshold_mult * self.avg_intensity:
                        range_beta: float = np.mean(beta2[range_mask])
                        if not np.isfinite(range_beta) or not -1 <= range_beta <= 2:
                            rejected_beta_count += 1
                            continue

                        r_centers.append(r_center)
                        beta2_vals.append(range_beta)
                        beta2_errs.append(0.0)  # No error calculation in rbasex
                        intensities.append(range_intensity)
                        energies.append(self._pixel_to_energy(r_center))
                        a_vals.append(0.0)  # Not provided by rbasex
                        a_errs.append(0.0)  # Not provided by rbasex
        else:
            # Use Anora for other transform algorithms
            # Prepare Anora
            img: Optional[NDArray[np.float64]]
            if self.fix_center_artifact.get():
                img = self.fix_IM
            else:
                img = self.AIM.transform
            height: int
            width: int
            height, width = img.shape
            x0: float = (width - 1) / 2
            y0: float = (height - 1) / 2
            Anni: Any = Anora(img, x0, y0)

            # Inputs
            r_range: List[Tuple[int, int]] = [(r_min, r_max)]

            # Get fitting method parameters
            fit_method: str = self.fit_method_combo.get()
            show_fit: bool = self.show_fit_plot.get()

            # RANSAC parameters if needed
            ransac_trials: int
            ransac_min_samples: int
            ransac_global_thresh: float
            ransac_special_thresh: float
            ransac_seed: Optional[int]
            if fit_method == 'RANSAC':
                try:
                    ransac_trials = int(self.ransac_trials.get())
                    ransac_min_samples = int(self.ransac_min_samples.get())
                    ransac_global_thresh = float(self.ransac_global_thresh.get())
                    ransac_special_thresh = float(self.ransac_special_thresh.get())
                    seed_text = self.ransac_seed.get().strip()
                    ransac_seed = int(seed_text) if seed_text else None
                    if ransac_trials < 1:
                        raise ValueError("RANSAC trials must be at least 1.")
                    if not 2 <= ransac_min_samples <= 360:
                        raise ValueError("RANSAC minimum samples must be between 2 and 360.")
                    if ransac_global_thresh <= 0 or ransac_special_thresh <= 0:
                        raise ValueError("RANSAC thresholds must be greater than zero.")
                except ValueError as exc:
                    messagebox.showerror("Invalid RANSAC settings", str(exc))
                    self.update_text(f"Invalid RANSAC settings: {exc}\n", clear_first=False)
                    return
            else:
                # Default values if not using RANSAC
                ransac_trials = 250
                ransac_min_samples = 25
                ransac_global_thresh = 3.5
                ransac_special_thresh = 2.0
                ransac_seed = None

            # Generate r ranges
            range_list: List[Tuple[int, int]]
            if is_rolling:
                range_list = self._generate_rolling_ranges(r_range, step)
            else:
                range_list = self._generate_ranges(r_range, step)

            # Lists for storing results for fit plots
            theta_data: List[Any] = []
            w_theta_data: List[Any] = []
            theta_fit_data: List[Any] = []
            w_fit_data: List[Any] = []
            inliers_data: List[Any] = []

            # Get average intensity for entire image
            self.avg_intensity: float = np.mean(img)

            # Calculate initial intensity for the entire range
            rough_range_intensity: float = self.get_average_intensity_for_range(Anni, r_min, r_max)

            for (start_r, end_r) in range_list:
                rng_intensity: float = self.get_average_intensity_for_range(Anni, start_r, end_r)
                if rng_intensity > threshold_mult * self.avg_intensity:
                    # Call the updated calculate_beta2 method with appropriate parameters
                    try:
                        result: Dict[str, Any] = Anni.calculate_beta2(
                            start_r, end_r,
                            method=fit_method,
                            RANSAC_trials=ransac_trials,
                            RANSAC_min_samples=ransac_min_samples,
                            RANSAC_global_thresh=ransac_global_thresh,
                            RANSAC_special_thresh=ransac_special_thresh,
                            random_seed=ransac_seed,
                            show=False
                        )
                    except (RuntimeError, ValueError) as exc:
                        self.update_text(
                            f"Skipped radial range {start_r}-{end_r}: {exc}\n",
                            clear_first=False,
                        )
                        continue

                    # Extract results from the dictionary returned by calculate_beta2
                    a_fit: float
                    a_err: float
                    beta2_fit: float
                    beta2_err: float
                    (a_fit, a_err) = result["A"]
                    (beta2_fit, beta2_err) = result["beta2"]

                    # Store all data
                    center: float = (start_r + end_r) / 2.0
                    r_centers.append(center)
                    beta2_vals.append(beta2_fit)
                    beta2_errs.append(beta2_err)
                    a_vals.append(a_fit)
                    a_errs.append(a_err)
                    intensities.append(rng_intensity)
                    # Energy
                    energies.append(self._pixel_to_energy(center))

                    # Store angular distribution data for potential plotting
                    theta_data.append(result["theta_deg"])
                    w_theta_data.append(result["W_theta"])
                    theta_fit_data.append(result["theta_fit_deg"])
                    w_fit_data.append(result["W_fit"])
                    inliers_data.append(result["inliers"])

            # Show fit plot if requested (for Anora only) not really relevant in this version
            if show_fit and r_centers and self.transform_combo.get() != 'rbasex':
                # Get the index of the max intensity
                max_idx: int = intensities.index(max(intensities))

                # Create a new window for the fit plot
                fit_window: tk.Toplevel = tk.Toplevel(self)
                fit_window.title("Angular Distribution Fit")
                fit_window.geometry("800x600")

                # Create the figure and canvas for the fit plot
                fit_fig: Figure = Figure(figsize=(8, 6), dpi=120)
                fit_ax: Any = fit_fig.add_subplot(111)

                # Plot the data and fit
                fit_ax.plot(theta_data[max_idx], w_theta_data[max_idx], "o", ms=5, alpha=0.6, label="data (norm.)")
                fit_ax.plot(theta_fit_data[max_idx], w_fit_data[max_idx], "-", lw=2, label="fit (norm.)")

                # Highlight inliers if using RANSAC
                if fit_method == 'RANSAC':
                    fit_ax.scatter(
                        np.array(theta_data[max_idx])[inliers_data[max_idx]],
                        np.array(w_theta_data[max_idx])[inliers_data[max_idx]],
                        s=70, facecolors="none", edgecolors="lime", linewidths=1.4, label="inliers"
                    )

                # Set labels and title
                fit_ax.set_xlabel("θ (deg)")
                fit_ax.set_ylabel("probability per bin")
                r_center: float = r_centers[max_idx]
                fit_ax.set_title(
                    f"r = {r_center:.1f} px  |  β₂ = {beta2_vals[max_idx]:.3f} ± {beta2_errs[max_idx]:.3f}  |  A = {a_vals[max_idx]:.3g}"
                )
                fit_ax.grid(alpha=0.25)
                fit_ax.legend(framealpha=0.9)

                # Create and pack the canvas
                fit_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(fit_fig, master=fit_window)
                fit_canvas.draw()
                fit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # Add a navigation toolbar
                fit_toolbar: NavigationToolbar2Tk = NavigationToolbar2Tk(fit_canvas, fit_window)
                fit_toolbar.update()
                fit_canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        # Keep the inverse image beside the anisotropy curve. Plot data are
        # retained so color-scale changes can redraw this coordinated view.
        self._anisotropy_plot_data = {
            "r_centers": list(r_centers),
            "beta2_values": list(beta2_vals),
            "beta2_errors": list(beta2_errs),
            "step": step,
            "rolling": is_rolling,
            "rbasex": self.transform_combo.get() == "rbasex",
        }
        result_image = self._result_image()
        if result_image is not None:
            self._update_slider_range(result_image)
        self._update_anisotropy_plot()

        # Show results in the text box
        self.text_box.delete(1.0, tk.END)

        if self.transform_combo.get() == 'rbasex':
            self.update_text(f"Average intensity: {self.avg_intensity:.4f}\n", clear_first=True)
        else:
            self.update_text(f"Average image intensity: {self.avg_intensity:.4f}\n", clear_first=True)
            self.update_text(f"Average rough range intensity: {rough_range_intensity:.4f}\n", clear_first=False)

        self.update_text("Results for radial slices above threshold:\n", clear_first=False)
        if rejected_beta_count:
            self.update_text(
                f"Rejected {rejected_beta_count} rbasex values outside the physical "
                "range -1 <= beta2 <= 2.\n",
                clear_first=False,
            )

        if self.display_in_energy.get():
            # With errors for non-rbasex methods
            if self.transform_combo.get() == 'rbasex':
                self.update_text("Energy (cm^-1)\tβ₂\tIntensity\n", clear_first=False)
                for i, rC in enumerate(r_centers):
                    E = self._pixel_to_energy(rC)
                    self.update_text(
                        f"{E:.1f}\t{beta2_vals[i]:.4f}\t{intensities[i]:.4f}\n",
                        clear_first=False)
            else:
                self.update_text("Energy (cm^-1)\tβ₂\tError\tIntensity\tA\tA Error\n", clear_first=False)
                for i, rC in enumerate(r_centers):
                    E = self._pixel_to_energy(rC)
                    self.update_text(
                        f"{E:.1f}\t{beta2_vals[i]:.4f}\t{beta2_errs[i]:.4f}\t{intensities[i]:.4f}\t{a_vals[i]:.4f}\t{a_errs[i]:.4f}\n",
                        clear_first=False)
        else:
            if self.transform_combo.get() == 'rbasex':
                self.update_text("Radial px\tβ₂\tIntensity\n", clear_first=False)
                for i, rC in enumerate(r_centers):
                    self.update_text(
                        f"{rC:.1f}\t{beta2_vals[i]:.4f}\t{intensities[i]:.4f}\n",
                        clear_first=False)
            else:
                self.update_text("Radial px\tβ₂\tError\tIntensity\tA\tA Error\n", clear_first=False)
                for i, rC in enumerate(r_centers):
                    self.update_text(
                        f"{rC:.1f}\t{beta2_vals[i]:.4f}\t{beta2_errs[i]:.4f}\t{intensities[i]:.4f}\t{a_vals[i]:.4f}\t{a_errs[i]:.4f}\n",
                        clear_first=False)

        # Save data dictionary for saving
        method_name: str
        if self.transform_combo.get() == 'rbasex':
            method_name = 'rbasex-Ibeta'
        else:
            method_name = fit_method

        self.anisotropy_data: Dict[str, Any] = {
            'r_centers': r_centers,
            'beta2_values': beta2_vals,
            'beta2_errors': beta2_errs,
            'a_values': a_vals,
            'a_errors': a_errs,
            'intensities': intensities,
            'energies': energies,
            'fit_method': method_name,
            'rolling': is_rolling,
            'step_size': step,
            'ransac_settings': {
                'trials': ransac_trials,
                'min_samples': ransac_min_samples,
                'global_threshold': ransac_global_thresh,
                'special_threshold': ransac_special_thresh,
                'random_seed': ransac_seed,
            } if method_name == 'RANSAC' else None,
        }

        if not r_centers:
            messagebox.showwarning(
                "No anisotropy results",
                "No radial slices passed the intensity threshold.",
            )
            return

        # Create save popup
        self._create_save_anisotropy_popup(self.anisotropy_data)

    # ==========================================================================
    # Utilities
    # ==========================================================================
    def find_peak_ranges(self, x: NDArray[np.float64], y: NDArray[np.float64], threshold: float = 10, rel_height: float = 0.8, min_width: int = 50, prominence_min: Optional[float] = None) -> List[Tuple[float, float]]:
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("Peak coordinates and values must be matching 1D arrays.")
        if not 0 <= rel_height <= 1:
            raise ValueError("Relative peak height must be between 0 and 1.")
        if min_width < 1:
            raise ValueError("Minimum peak width must be at least one sample.")

        # Find peaks
        peaks: NDArray[np.int_]
        peak_properties: Dict[str, NDArray[np.float64]]
        peaks, peak_properties = find_peaks(
            y,
            height=threshold,
            prominence=0.0 if prominence_min is None else prominence_min,
        )

        # Calculate prominences
        prominences: NDArray[np.float64] = peak_properties['prominences']

        peak_ranges: List[Tuple[float, float]] = []
        for peak, prominence in zip(peaks, prominences):  # noqa: B905
            # Define a threshold based on the peak height and prominence
            height_threshold: float = y[peak] - prominence * rel_height

            # Find left boundary
            left: int = peak
            while left > 0 and y[left] > height_threshold:
                left -= 1

            # Find right boundary
            right: int = peak
            while right < len(y) - 1 and y[right] > height_threshold:
                right += 1

            # Ensure minimum width
            if right - left < min_width:
                padding: int = (min_width - (right - left)) // 2
                left = max(0, left - padding)
                right = min(len(y) - 1, right + padding)

            peak_ranges.append((x[left], x[right]))

        return peak_ranges

    def _generate_ranges(self, ranges: List[Tuple[int, int]], step: int = 3) -> List[Tuple[int, int]]:
        self.r_range_method: str = 'Non-Rolling'
        self.anni_stepsize: int = step
        result: List[Tuple[int, int]] = []
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
                    result.append((current, min(current + step, end_range)))
                    current += step

        return result

    def _generate_rolling_ranges(self, ranges: List[Tuple[int, int]], step: int = 3) -> List[Tuple[int, int]]:
        self.r_range_method: str = 'Rolling'
        self.anni_stepsize: int = step
        result: List[Tuple[int, int]] = []
        for (start_range, end_range) in ranges:
            current = start_range
            while current < end_range:
                result.append((current, min(current + step, end_range)))
                current += 1
        return result

    def get_average_intensity_for_range(self, Anni: Any, r_min: float, r_max: float) -> float:
        # Get average intensity for a range using the Anora
        return Anni.average_intensity(r_min, r_max)

    def _save_speed_data(self) -> None:
        if self.speed_distribution is None or self.radial_coords is None:
            messagebox.showinfo("Information", "Please calculate speed distribution first.")
            return

        file_path: str = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'w') as f:
                # Write header
                f.write(f"Speed Distribution Data - Inverse Abel by {self.inverse_method}\n")
                f.write("# Pixel_Center\tEnergy(cm-1)\tSpeed_Distribution\n")

                # Write data
                energies: List[float] = [self._pixel_to_energy(r) for r in self.radial_coords]
                for i in range(len(self.radial_coords)):
                    f.write(
                        f"{self.radial_coords[i]:.1f}\t{energies[i]:.1f}\t"
                        f"{self.speed_distribution[i]:.6f}\n")

            self.update_text(f"Speed distribution data saved to {file_path}\n", clear_first=False)

    def _create_save_anisotropy_popup(self, data_dict: Dict[str, Any]) -> None:
        popup: tk.Toplevel = tk.Toplevel(self)
        popup.title("Save Data")
        popup.geometry("300x150")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        label: tk.Label = tk.Label(popup, text="Save the plot data?")
        label.pack(pady=20)

        button_frame: tk.Frame = tk.Frame(popup)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Yes",
                  command=lambda: self._save_anni_data(data_dict, popup)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="No",
                  command=popup.destroy).pack(side=tk.LEFT, padx=10)

    def _save_anni_data(self, data_dict: Dict[str, Any], popup: tk.Toplevel) -> None:
        file_path: str = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                # Write header with appropriate inverse abel method, anisotropy method, and units
                f.write(
                    f"Inverse Abel by {self.inverse_method}, {self.r_range_method} Anisotropy with step size of {self.anni_stepsize} pixels\n")
                f.write(f"Fitting method: {data_dict['fit_method']}\n")
                if data_dict.get('ransac_settings') is not None:
                    settings = data_dict['ransac_settings']
                    f.write(
                        "RANSAC settings: "
                        f"trials={settings['trials']}, "
                        f"min_samples={settings['min_samples']}, "
                        f"global_threshold={settings['global_threshold']}, "
                        f"special_threshold={settings['special_threshold']}, "
                        f"random_seed={settings['random_seed']}\n"
                    )

                # Different headers for rbasex vs Anora
                if 'rbasex' in data_dict['fit_method']:
                    f.write("# Pixel_Center\t# Energy(cm-1)\tBeta2\tIntensity\n")
                    for i in range(len(data_dict['r_centers'])):
                        f.write(
                            f"{data_dict['r_centers'][i]:.1f}\t{data_dict['energies'][i]:.1f}\t{data_dict['beta2_values'][i]:.4f}\t"
                            f"{data_dict['intensities'][i]:.4f}\n")
                else:
                    f.write(
                        "# Pixel_Center\t# Energy(cm-1)\tBeta2\tBeta2_Error"
                        "\tIntensity\tA\tA_Error\n"
                    )
                    for i in range(len(data_dict['r_centers'])):
                        f.write(
                            f"{data_dict['r_centers'][i]:.1f}\t{data_dict['energies'][i]:.1f}\t{data_dict['beta2_values'][i]:.4f}\t"
                            f"{data_dict['beta2_errors'][i]:.4f}\t{data_dict['intensities'][i]:.4f}\t"
                            f"{data_dict['a_values'][i]:.4f}\t{data_dict['a_errors'][i]:.4f}\n")
        popup.destroy()

    def _save_current_image_data(self) -> None:
        if self.IM is None:
            self.update_text("No image data to save.\n", clear_first=False)
            return

        file_path: str = filedialog.asksaveasfilename(
            defaultextension='.dat',
            filetypes=[("DAT files", "*.dat"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                np.savetxt(file_path, self.IM)
                self.update_text(f"Image data saved to {file_path}\n", clear_first=False)
            except Exception as e:
                self.update_text(f"Error saving image data: {str(e)}\n", clear_first=False)

    def _update_button_state(self, button: Union[tk.Widget, ttk.Combobox], disable: bool = True) -> None:
        if isinstance(button, ttk.Combobox):
            # For Combobox
            if disable:
                button.configure(state='disabled', foreground='#808080')
            else:
                button.configure(state='readonly', foreground='black')
        else:
            # For other widgets like Labels, Buttons, etc.
            if disable:
                button.configure(state=tk.DISABLED, foreground='#808080')
            else:
                button.configure(state=tk.NORMAL, foreground='black')

    def _update_slider_range(self, data: Optional[NDArray[np.float64]]) -> None:
        if data is None:
            return
        finite_data = np.asarray(data)[np.isfinite(data)]
        if finite_data.size == 0:
            return
        vmin = max(0.0, float(np.percentile(finite_data, 0.5)))
        vmax = float(np.percentile(finite_data, 99.99))
        if vmax <= vmin:
            vmax = vmin + max(0.01, abs(vmin) * 0.01)
        range_size = vmax - vmin
        resolution = 0.01 if range_size <= 2 else range_size / 100

        self.vmin_slider.config(from_=vmin, to=vmax, resolution=resolution)
        self.vmax_slider.config(from_=vmin, to=vmax, resolution=resolution)
        self.vmin_slider.set(vmin)
        self.vmax_slider.set(vmax)

        self.vmin_entry.delete(0, tk.END)
        self.vmin_entry.insert(0, f"{vmin:.2f}")
        self.vmax_entry.delete(0, tk.END)
        self.vmax_entry.insert(0, f"{vmax:.2f}")

    def _slider_changed(self, value: str) -> None:
        vmin: float = self.vmin_slider.get()
        vmax: float = self.vmax_slider.get()
        self.vmin_entry.delete(0, tk.END)
        self.vmin_entry.insert(0, f"{vmin:.2f}")
        self.vmax_entry.delete(0, tk.END)
        self.vmax_entry.insert(0, f"{vmax:.2f}")
        if vmin < vmax:
            self._schedule_plot_update()

    def _entry_changed(self, event: Optional[tk.Event] = None) -> None:
        try:
            vmin: float = float(self.vmin_entry.get())
            vmax: float = float(self.vmax_entry.get())
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                raise ValueError
            self.vmin_slider.set(vmin)
            self.vmax_slider.set(vmax)
            self._schedule_plot_update()
        except ValueError:
            pass

    def _schedule_plot_update(self) -> None:
        """Debounce plot updates to avoid excessive redraws"""
        # Cancel pending update if exists
        if self._pending_plot_update is not None:
            self.after_cancel(self._pending_plot_update)

        # Schedule new update
        self._pending_plot_update = self.after(self._plot_delay_ms, self._execute_plot_update)

    def _execute_plot_update(self) -> None:
        """Execute the actual plot update"""
        self._pending_plot_update = None
        if (self._active_result_view == "speed"
                and self.speed_distribution is not None
                and self.radial_coords is not None):
            self._update_speed_plot()
        elif (self._active_result_view == "anisotropy"
              and self._anisotropy_plot_data is not None):
            self._update_anisotropy_plot()
        else:
            self._update_plot()

    def _peak_slider_changed(self, param_name: str, value: str) -> None:
        # Handle changes to peak finding parameter sliders
        value_float: float = float(value)

        # Update corresponding entry
        if param_name == 'threshold':
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, f"{value_float:.2f}")
            self.peak_threshold = value_float
        elif param_name == 'rel_height':
            self.rel_height_entry.delete(0, tk.END)
            self.rel_height_entry.insert(0, f"{value_float:.2f}")
            self.peak_rel_height = value_float
        elif param_name == 'min_width':
            self.min_width_entry.delete(0, tk.END)
            self.min_width_entry.insert(0, f"{int(value_float)}")
            self.peak_min_width = int(value_float)
        elif param_name == 'prominence':
            self.prominence_entry.delete(0, tk.END)
            self.prominence_entry.insert(0, f"{value_float:.2f}")
            self.peak_prominence = value_float

        # Schedule update instead of immediate
        self._schedule_speed_plot_update()

    def _peak_entry_changed(self, param_name: str, *args: Any) -> None:
        # Handle changes to peak finding parameter entries
        try:
            value: Union[float, int]
            if param_name == 'threshold':
                value = float(self.threshold_entry.get())
                if not 0 <= value <= 1:
                    raise ValueError
                self.threshold_slider.set(value)
                self.peak_threshold = value
            elif param_name == 'rel_height':
                value = float(self.rel_height_entry.get())
                if not 0 <= value <= 1:
                    raise ValueError
                self.rel_height_slider.set(value)
                self.peak_rel_height = value
            elif param_name == 'min_width':
                value = int(self.min_width_entry.get())
                if not 1 <= value <= 50:
                    raise ValueError
                self.min_width_slider.set(value)
                self.peak_min_width = value
            elif param_name == 'prominence':
                value = float(self.prominence_entry.get())
                if not 0 <= value <= 1:
                    raise ValueError
                self.prominence_slider.set(value)
                self.peak_prominence = value

            # Schedule update
            self._schedule_speed_plot_update()
        except ValueError:
            pass

    def _schedule_speed_plot_update(self) -> None:
        """Debounce speed plot updates"""
        if self._pending_speed_plot_update is not None:
            self.after_cancel(self._pending_speed_plot_update)

        self._pending_speed_plot_update = self.after(self._plot_delay_ms, self._update_speed_plot)

    def _reset_linked_plot_state(self, clear_selection: bool = True) -> None:
        """Forget artists belonging to a coordinated result view."""
        self._active_result_view = None
        self._linked_image_ax = None
        self._linked_result_ax = None
        self._linked_image = None
        self._radius_ring = None
        self._radius_readout = None
        self._result_cursor_line = None
        self._result_dragging = False
        if clear_selection:
            self._selected_radius = None

    def _result_image(self) -> Optional[NDArray[np.float64]]:
        """Return the inverse-transformed image used for result inspection."""
        if (self.fix_center_artifact.get()
                and self.fix_IM is not None):
            return np.asarray(self.fix_IM)
        if self.AIM is not None:
            return np.asarray(self.AIM.transform)
        return None

    def _color_limits(self, image: NDArray[np.float64]) -> Tuple[float, float]:
        """Return valid display limits, falling back to robust image limits."""
        try:
            vmin = float(self.vmin_entry.get())
            vmax = float(self.vmax_entry.get())
        except ValueError:
            vmin = float("nan")
            vmax = float("nan")

        if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
            return vmin, vmax

        finite = np.asarray(image)[np.isfinite(image)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(finite, 0.5))
        vmax = float(np.percentile(finite, 99.5))
        if vmax <= vmin:
            vmax = vmin + max(0.01, abs(vmin) * 0.01)
        return vmin, vmax

    def _build_linked_result_axes(
        self,
        view_name: str,
        result_title: str,
        xlabel: str,
        ylabel: str,
        x_mode: str,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Build image and curve panels that share an interactive radius."""
        image = self._result_image()
        if image is None:
            return None, None

        self._reset_linked_plot_state(clear_selection=False)
        self._active_result_view = view_name
        self._linked_x_mode = x_mode
        self._linked_image = image

        self.fig.clear()
        self.fig.patch.set_facecolor("#f4f6f8")
        grid = self.fig.add_gridspec(
            1, 2,
            width_ratios=(1.0, 1.28),
            left=0.06,
            right=0.97,
            bottom=0.13,
            top=0.87,
            wspace=0.32,
        )
        image_ax = self.fig.add_subplot(grid[0, 0])
        result_ax = self.fig.add_subplot(grid[0, 1])
        self._linked_image_ax = image_ax
        self._linked_result_ax = result_ax
        self.ax = result_ax

        vmin, vmax = self._color_limits(image)
        displayed = image_ax.imshow(
            image,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        image_ax.set_xlabel("x (pixels)", fontsize=11)
        image_ax.set_ylabel("y (pixels)", fontsize=11)
        image_ax.tick_params(labelsize=10)
        for spine in image_ax.spines.values():
            spine.set_color("#4f5968")
            spine.set_linewidth(0.8)
        colorbar = self.fig.colorbar(
            displayed, ax=image_ax, fraction=0.046, pad=0.035
        )
        colorbar.set_label("Intensity", color="#4a5260", fontsize=10.5)
        colorbar.ax.tick_params(colors="#4a5260", labelsize=9.5)

        rows, columns = image.shape
        center = ((columns - 1) / 2.0, (rows - 1) / 2.0)
        self._radius_ring = Circle(
            center,
            radius=0.0,
            fill=False,
            edgecolor="#62e6ff",
            linewidth=2.2,
            visible=False,
            zorder=5,
        )
        self._radius_ring.set_path_effects([
            path_effects.Stroke(linewidth=4.2, foreground="#111722", alpha=0.8),
            path_effects.Normal(),
        ])
        image_ax.add_patch(self._radius_ring)
        self._radius_readout = image_ax.text(
            0.035,
            0.965,
            "",
            transform=image_ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            visible=False,
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "#111722",
                "edgecolor": "#62e6ff",
                "alpha": 0.84,
            },
        )

        result_ax.set_facecolor("white")
        if result_title:
            result_ax.set_title(
                result_title,
                loc="left",
                fontsize=13,
                fontweight="semibold",
                pad=10,
            )
        result_ax.set_xlabel(xlabel, fontsize=11)
        result_ax.set_ylabel(ylabel, fontsize=11)
        result_ax.grid(
            axis="y", color="#c8d0da", linestyle="--", linewidth=0.8, alpha=0.65
        )
        result_ax.spines["top"].set_visible(False)
        result_ax.spines["right"].set_visible(False)
        result_ax.spines["left"].set_color("#7b8491")
        result_ax.spines["bottom"].set_color("#7b8491")
        result_ax.tick_params(colors="#3f4752", labelsize=10)
        self._result_cursor_line = result_ax.axvline(
            0.0,
            color="#e95464",
            linewidth=1.5,
            linestyle=(0, (4, 3)),
            alpha=0.9,
            visible=False,
            zorder=6,
        )
        return image_ax, result_ax

    def _radius_from_result_x(self, x_value: float) -> Optional[float]:
        if not np.isfinite(x_value) or self._linked_image is None:
            return None
        try:
            radius = (
                self._energy_to_pixel(float(x_value))
                if self._linked_x_mode == "energy"
                else float(x_value)
            )
        except ValueError:
            return None

        maximum = (min(self._linked_image.shape) - 1) / 2.0
        return float(np.clip(radius, 0.0, maximum))

    def _update_radius_highlight(self, radius: float) -> None:
        """Move the image ring and curve guide to a pixel radius."""
        if (self._linked_image is None
                or self._radius_ring is None
                or self._result_cursor_line is None):
            return
        maximum = (min(self._linked_image.shape) - 1) / 2.0
        radius = float(np.clip(radius, 0.0, maximum))
        self._selected_radius = radius
        energy = self._pixel_to_energy(radius)
        x_value = energy if self._linked_x_mode == "energy" else radius

        self._radius_ring.set_radius(radius)
        self._radius_ring.set_visible(True)
        self._result_cursor_line.set_xdata([x_value, x_value])
        self._result_cursor_line.set_visible(True)
        if self._radius_readout is not None:
            self._radius_readout.set_text(
                f"Radius  {radius:.2f} px\nEnergy  {energy:.2f} cm⁻¹"
            )
            self._radius_readout.set_visible(True)
        self.canvas.draw_idle()

    def _update_radius_from_event(self, event: Any) -> None:
        if (event.inaxes is not self._linked_result_ax
                or event.xdata is None):
            return
        radius = self._radius_from_result_x(float(event.xdata))
        if radius is not None:
            self._update_radius_highlight(radius)

    def _on_result_pointer_press(self, event: Any) -> None:
        if (event.inaxes is not self._linked_result_ax
                or event.button != 1
                or bool(getattr(self.toolbar, "mode", ""))):
            return
        self._result_dragging = True
        self._update_radius_from_event(event)

    def _on_result_pointer_move(self, event: Any) -> None:
        if self._result_dragging:
            self._update_radius_from_event(event)

    def _on_result_pointer_release(self, event: Any) -> None:
        if not self._result_dragging:
            return
        self._update_radius_from_event(event)
        self._result_dragging = False

    def _update_plot(self, image: Optional[NDArray[np.float64]] = None) -> None:
        if image is None:
            image = self.IM
        if image is None:
            return

        self._reset_linked_plot_state()
        self.fig.clear()
        self.fig.patch.set_facecolor("#f4f6f8")
        ax: Any = self.fig.add_subplot(111)
        self.ax = ax
        vmin, vmax = self._color_limits(image)
        displayed = ax.imshow(
            image,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("x (pixels)", fontsize=11)
        ax.set_ylabel("y (pixels)", fontsize=11)
        ax.tick_params(labelsize=10)
        colorbar = self.fig.colorbar(displayed, ax=ax, fraction=0.046, pad=0.035)
        colorbar.set_label("Intensity", fontsize=10.5)
        colorbar.ax.tick_params(labelsize=9.5)
        self.fig.subplots_adjust(left=0.09, right=0.91, bottom=0.1, top=0.9)
        self.canvas.draw()

    def _update_anisotropy_plot(self) -> None:
        data = self._anisotropy_plot_data
        if data is None:
            return

        display_energy = bool(self.display_in_energy.get())
        radii = np.asarray(data["r_centers"], dtype=float)
        beta2 = np.asarray(data["beta2_values"], dtype=float)
        errors = np.asarray(data["beta2_errors"], dtype=float)
        x_values = (
            np.asarray([self._pixel_to_energy(r) for r in radii])
            if display_energy else radii
        )
        xlabel = "Energy (cm⁻¹)" if display_energy else "Radius (pixels)"
        _, ax = self._build_linked_result_axes(
            "anisotropy",
            "",
            xlabel,
            "Anisotropy parameter β₂",
            "energy" if display_energy else "radius",
        )
        if ax is None:
            return

        ax.axhspan(-1.0, 2.0, color="#edf7f0", alpha=0.75, zorder=0)
        ax.axhline(0.0, color="#8993a1", linewidth=0.9, zorder=1)
        ax.axhline(-1.0, color="#67a678", linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(2.0, color="#67a678", linewidth=0.8, linestyle=":", zorder=1)
        if data["rbasex"]:
            ax.plot(
                x_values,
                beta2,
                color="#315cab",
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.3,
                zorder=3,
            )
        else:
            ax.errorbar(
                x_values,
                beta2,
                yerr=errors,
                color="#315cab",
                ecolor="#7895c7",
                linewidth=1.9,
                elinewidth=1.0,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=3,
                zorder=3,
            )
        ax.text(
            0.02,
            0.97,
            "physical β₂ range  −1 to 2",
            transform=ax.transAxes,
            va="top",
            color="#568565",
            fontsize=8.5,
        )
        ax.margins(x=0.025)
        if self._selected_radius is not None:
            self._update_radius_highlight(self._selected_radius)
        self.canvas.draw()

    def _update_speed_plot(self) -> None:
        # Update speed distribution plot with current peak finding parameters
        self._pending_speed_plot_update = None
        if self.speed_distribution is None or self.radial_coords is None:
            return

        try:
            peak_ranges: List[Tuple[float, float]] = self.find_peak_ranges(
                self.radial_coords,
                self.speed_distribution,
                threshold=self.peak_threshold,
                rel_height=self.peak_rel_height,
                min_width=self.peak_min_width,
                prominence_min=self.peak_prominence
            )
        except ValueError as exc:
            self.update_text(f"Invalid peak settings: {exc}\n", clear_first=False)
            return

        display_energy = bool(self.display_in_energy.get())
        if display_energy:
            x_coords = np.asarray([
                self._pixel_to_energy(r) for r in self.radial_coords
            ])
            xlabel = "Energy (cm⁻¹)"
        else:
            x_coords = np.asarray(self.radial_coords)
            xlabel = "Radius (pixels)"

        _, ax = self._build_linked_result_axes(
            "speed",
            "Speed distribution",
            xlabel,
            "Intensity (arb. units)",
            "energy" if display_energy else "radius",
        )
        if ax is None:
            return

        distribution = np.asarray(self.speed_distribution)
        ax.plot(x_coords, distribution, color="#2369a8", linewidth=2.1, zorder=3)
        ax.fill_between(
            x_coords, 0.0, distribution, color="#5ca4d6", alpha=0.17, zorder=2
        )

        # Clear previous peak information
        self.text_box.delete(1.0, tk.END)
        self.update_text("Peak information:\n", clear_first=True)

        peak_colors = ("#f0a04b", "#9b72cf", "#42a88a", "#dd6b7b")
        for i, (left, right) in enumerate(peak_ranges):
            if display_energy:
                plot_left = self._pixel_to_energy(left)
                plot_right = self._pixel_to_energy(right)
                center_pixel = (left + right) / 2
                center = self._pixel_to_energy(center_pixel)
                self.update_text(
                    f"Peak {i + 1}: {center:.2f} cm^-1\n", clear_first=False
                )
            else:
                plot_left, plot_right = left, right
                center = (left + right) / 2
                self.update_text(
                    f"Peak {i + 1}: {center:.2f} pixels\n", clear_first=False
                )

            color = peak_colors[i % len(peak_colors)]
            ax.axvspan(plot_left, plot_right, alpha=0.14, color=color, zorder=1)
            ax.annotate(
                f"Peak {i + 1}",
                (center, 0.92),
                ha="center",
                va="top",
                color=color,
                fontsize=8.5,
                fontweight="semibold",
                xycoords=("data", "axes fraction"),
            )

        ax.axhline(0.0, color="#8993a1", linewidth=0.9, zorder=1)
        ax.margins(x=0.025)
        if self._selected_radius is not None:
            self._update_radius_highlight(self._selected_radius)
        self.canvas.draw()

    def update_text(self, text: str, clear_first: bool = False) -> None:
        self.text_box.config(state="normal")
        if clear_first:
            self.text_box.delete(1.0, tk.END)
            # Always display current filename first if it exists
            if self.current_filename:
                self.text_box.insert(tk.END, f"File: {self.current_filename}\n")
                self.text_box.insert(tk.END, "-" * 50 + "\n")
        self.text_box.insert(tk.END, text)
        self.text_box.see(tk.END)
        self.text_box.config(state="disabled")

    def _run_conditional_image_fixer(self) -> None:
        # Only run image fixer if the fix_center_artifact option is selected
        if self.fix_center_artifact.get():
            self._image_fixer()

    def _quit(self) -> None:
        window = self.winfo_toplevel()
        window.quit()
        window.destroy()

    # ==========================================================================
    # Speed and Energy Settings
    # ==========================================================================
    def _speed_settings(self) -> None:
        # Check if we have speed distribution data
        if self.speed_distribution is None or self.radial_coords is None:
            messagebox.showinfo("Information", "Please calculate speed distribution first.")
            return
        else:
            settings_window: tk.Toplevel = tk.Toplevel(self)
            settings_window.title("Peak Finding Settings")
            settings_window.geometry("400x250")
            settings_window.transient(self.winfo_toplevel())
            settings_window.grab_set()

            # Create frame for settings
            speed_frame: tk.LabelFrame = tk.LabelFrame(settings_window, text="Peak Parameters", padx=10, pady=10)
            speed_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Store current settings
            self.peak_threshold: float = getattr(self, 'peak_threshold', 0.5)
            self.peak_rel_height: float = getattr(self, 'peak_rel_height', 0.5)
            self.peak_min_width: int = getattr(self, 'peak_min_width', 5)
            self.peak_prominence: float = getattr(self, 'peak_prominence', 0.1)

            # Threshold row
            tk.Label(speed_frame, text="Threshold:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
            self.threshold_entry: tk.Entry = tk.Entry(speed_frame, width=10)
            self.threshold_entry.grid(row=0, column=1, padx=5, pady=2)
            self.threshold_entry.insert(0, f"{self.peak_threshold:.2f}")
            self.threshold_slider: tk.Scale = tk.Scale(speed_frame, from_=0, to=1.0, resolution=0.01,
                                             orient=tk.HORIZONTAL,
                                             command=lambda v: self._peak_slider_changed('threshold', v))
            self.threshold_slider.set(self.peak_threshold)
            self.threshold_slider.grid(row=0, column=2, sticky="we", padx=5, pady=2)
            self.threshold_entry.bind('<Return>', lambda e: self._peak_entry_changed('threshold'))

            # Relative height row
            tk.Label(speed_frame, text="Rel. Height:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
            self.rel_height_entry: tk.Entry = tk.Entry(speed_frame, width=10)
            self.rel_height_entry.grid(row=1, column=1, padx=5, pady=2)
            self.rel_height_entry.insert(0, f"{self.peak_rel_height:.2f}")
            self.rel_height_slider: tk.Scale = tk.Scale(speed_frame, from_=0, to=1.0, resolution=0.01,
                                              orient=tk.HORIZONTAL,
                                              command=lambda v: self._peak_slider_changed('rel_height', v))
            self.rel_height_slider.set(self.peak_rel_height)
            self.rel_height_slider.grid(row=1, column=2, sticky="we", padx=5, pady=2)
            self.rel_height_entry.bind('<Return>', lambda e: self._peak_entry_changed('rel_height'))

            # Min width row
            tk.Label(speed_frame, text="Min Width:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
            self.min_width_entry: tk.Entry = tk.Entry(speed_frame, width=10)
            self.min_width_entry.grid(row=2, column=1, padx=5, pady=2)
            self.min_width_entry.insert(0, f"{self.peak_min_width}")
            self.min_width_slider: tk.Scale = tk.Scale(speed_frame, from_=1, to=50, resolution=1,
                                             orient=tk.HORIZONTAL,
                                             command=lambda v: self._peak_slider_changed('min_width', v))
            self.min_width_slider.set(self.peak_min_width)
            self.min_width_slider.grid(row=2, column=2, sticky="we", padx=5, pady=2)
            self.min_width_entry.bind('<Return>', lambda e: self._peak_entry_changed('min_width'))

            # Prominence row
            tk.Label(speed_frame, text="Prominence:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
            self.prominence_entry: tk.Entry = tk.Entry(speed_frame, width=10)
            self.prominence_entry.grid(row=3, column=1, padx=5, pady=2)
            self.prominence_entry.insert(0, f"{self.peak_prominence:.2f}")
            self.prominence_slider: tk.Scale = tk.Scale(speed_frame, from_=0, to=1.0, resolution=0.01,
                                              orient=tk.HORIZONTAL,
                                              command=lambda v: self._peak_slider_changed('prominence', v))
            self.prominence_slider.set(self.peak_prominence)
            self.prominence_slider.grid(row=3, column=2, sticky="we", padx=5, pady=2)
            self.prominence_entry.bind('<Return>', lambda e: self._peak_entry_changed('prominence'))

    # Energy Conversion
    def _calibration_settings(self) -> None:
        settings_window: tk.Toplevel = tk.Toplevel(self)
        settings_window.title("Energy Conversion Settings")
        settings_window.geometry("350x200")
        settings_window.transient(self.winfo_toplevel())
        settings_window.grab_set()

        frame: tk.LabelFrame = tk.LabelFrame(settings_window, text="Energy Conversion", padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frame, text="1 px =").grid(row=0, column=0, padx=5, pady=5, sticky="e")

        base_str: str
        exponent_str: str
        base_str, exponent_str = f"{self.cm_per_pixel:e}".split('e')
        base: float = float(base_str)
        exponent: int = int(exponent_str)

        sci_frame: tk.Frame = tk.Frame(frame)
        sci_frame.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        base_entry: tk.Entry = tk.Entry(sci_frame, width=8)
        base_entry.pack(side=tk.LEFT)
        base_entry.insert(0, f"{base}")

        tk.Label(sci_frame, text="×10^").pack(side=tk.LEFT)

        exp_entry: tk.Entry = tk.Entry(sci_frame, width=4)
        exp_entry.pack(side=tk.LEFT)
        exp_entry.insert(0, str(exponent))

        tk.Label(frame, text=" kg*m/s").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Mass input
        tk.Label(frame, text="Ion Mass:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        mass_entry: tk.Entry = tk.Entry(frame, width=10)
        mass_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        mass_entry.insert(0, f"{self.ion_mass:.5f}")
        tk.Label(frame, text="amu").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        def apply_settings() -> None:
            try:
                base_val: float = float(base_entry.get())
                exp_val: int = int(exp_entry.get())
                self.cm_per_pixel = base_val * (10 ** exp_val)
                self.ion_mass = float(mass_entry.get())
                if self.cm_per_pixel <= 0 or self.ion_mass <= 0:
                    raise ValueError

                self._update_ke_from_r()
                settings_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")

        tk.Button(settings_window, text="Apply", command=apply_settings).pack(pady=10)

    def _pixel_to_energy(self, pixel: float) -> float:
        if pixel < 0 or self.cm_per_pixel <= 0 or self.ion_mass <= 0:
            raise ValueError("Pixel radius and calibration values must be positive.")
        return float(((self.cm_per_pixel * pixel) ** 2) *
                     (5.03411e22) / (2 * (self.ion_mass * 1.6605402e-27)))

    def _energy_to_pixel(self, energy: float) -> float:
        if energy < 0 or self.cm_per_pixel <= 0 or self.ion_mass <= 0:
            raise ValueError("Energy must be nonnegative and calibration values positive.")
        return float(np.sqrt(energy * 2 * (self.ion_mass * 1.6605402e-27) /
                             (5.03411e22)) / self.cm_per_pixel)

    def _update_ke_from_r(self) -> None:
        try:
            r_min_val: float = float(self.rmin.get())
            r_max_val: float = float(self.rmax.get())
            ke_min_val: float = self._pixel_to_energy(r_min_val)
            ke_max_val: float = self._pixel_to_energy(r_max_val)

            self.ke_min.delete(0, tk.END)
            self.ke_max.delete(0, tk.END)
            self.ke_min.insert(0, f"{ke_min_val:.2f}")
            self.ke_max.insert(0, f"{ke_max_val:.2f}")
        except ValueError:
            pass

    def _update_r_from_ke(self) -> None:
        try:
            ke_min_val: float = float(self.ke_min.get())
            ke_max_val: float = float(self.ke_max.get())

            r_min_val: float = self._energy_to_pixel(ke_min_val)
            r_max_val: float = self._energy_to_pixel(ke_max_val)

            self.rmin.delete(0, tk.END)
            self.rmax.delete(0, tk.END)
            self.rmin.insert(0, f"{r_min_val:.0f}")
            self.rmax.insert(0, f"{r_max_val:.0f}")
        except ValueError:
            pass


# Main
if __name__ == "__main__":
    app: MainGUI = MainGUI()
    app.mainloop()
