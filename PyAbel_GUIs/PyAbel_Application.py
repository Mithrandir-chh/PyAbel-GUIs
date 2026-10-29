"""Combined desktop application for the PyAbel analysis and image GUIs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk
import traceback
from types import SimpleNamespace

import numpy as np

import abel
import abel.direct

try:
    from .Supp_ImageGen_GUI import ImageGeneratorGUI, Model3DDialog
    from .gui_core import (
        DEFAULT_SPHERE_DIAMETER_MM,
        TRANSFORM_METHODS,
        distribution_reach,
        export_distribution_dots_glb,
        inverse_transform,
    )
except ImportError:  # Support direct execution from the PyAbel_GUIs directory.
    from Supp_ImageGen_GUI import ImageGeneratorGUI, Model3DDialog
    from gui_core import (
        DEFAULT_SPHERE_DIAMETER_MM,
        TRANSFORM_METHODS,
        distribution_reach,
        export_distribution_dots_glb,
        inverse_transform,
    )


APP_NAME = "PyAbel Studio"
MASTER_MODULE_NAME = "PyAbel_GUIs._master_gui_runtime"


def _load_master_module():
    """Load the legacy-named master module (its filename contains a dot)."""
    loaded = sys.modules.get(MASTER_MODULE_NAME)
    if loaded is not None:
        return loaded

    source_path = Path(__file__).with_name("Master_GUI_2.0.py")
    spec = importlib.util.spec_from_file_location(MASTER_MODULE_NAME, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the analysis GUI from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MASTER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(MASTER_MODULE_NAME, None)
        raise
    return module


def verify_cython_backend() -> tuple[bool, str]:
    """Exercise the compiled direct backend and compare it with Python."""
    if not abel.direct.cython_ext:
        return False, "compiled direct backend was not loaded"

    radius = np.arange(32, dtype=float)
    profile = np.exp(-((radius - 12.0) / 4.0) ** 2)
    compiled = abel.direct.direct_transform(profile, direction="forward", backend="C")
    reference = abel.direct.direct_transform(
        profile, direction="forward", backend="python"
    )
    if not np.all(np.isfinite(compiled)):
        return False, "compiled direct backend returned non-finite values"
    if not np.allclose(compiled, reference, rtol=1e-11, atol=1e-11):
        return False, "compiled direct backend disagrees with the Python backend"
    return True, "compiled direct backend operational"


def verify_transform_methods() -> tuple[bool, str]:
    """Run every transform exposed by the analysis GUI on a small image."""
    coordinates = np.arange(51, dtype=float) - 25
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x, y)
    image = np.exp(-((radius - 13.0) / 3.5) ** 2)

    previous_basis_dir = abel.transform.get_basis_dir(make=False)
    with tempfile.TemporaryDirectory(prefix="pyabel-studio-basis-") as basis_dir:
        abel.transform.set_basis_dir(basis_dir, make=True)
        try:
            for method in TRANSFORM_METHODS:
                try:
                    result = inverse_transform(image, method)
                except Exception as exc:
                    return False, f"{method} failed: {exc}"
                transformed = np.asarray(result.transform)
                if transformed.shape != image.shape or not np.all(
                    np.isfinite(transformed)
                ):
                    return False, f"{method} returned an invalid transform"
        finally:
            abel.transform.set_basis_dir(previous_basis_dir, make=False)

    return True, f"all {len(TRANSFORM_METHODS)} Abel transform methods operational"


class PyAbelStudio(tk.Tk):
    """One macOS-friendly window containing both PyAbel GUI workflows."""

    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1280x860")
        self.minsize(1020, 720)

        self._build_menu()
        self._build_workspace()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(
            label="Open Image in Analysis…", command=self._open_analysis_image
        )
        file_menu.add_command(
            label="Save Generator Image…", command=self._save_generator_image
        )
        file_menu.add_command(
            label="Export Generator 3D Model…", command=self._export_generator_model
        )
        file_menu.add_separator()
        file_menu.add_command(label="Quit PyAbel Studio", command=self.destroy)
        menu_bar.add_cascade(label="File", menu=file_menu)

        workflow_menu = tk.Menu(menu_bar, tearoff=False)
        workflow_menu.add_command(
            label="Send Current Generator Image to Analysis",
            command=self.send_generator_image_to_analysis,
        )
        workflow_menu.add_separator()
        workflow_menu.add_command(
            label="Show Image Generator", command=lambda: self.notebook.select(0)
        )
        workflow_menu.add_command(
            label="Show Analysis", command=lambda: self.notebook.select(1)
        )
        menu_bar.add_cascade(label="Workflow", menu=workflow_menu)

        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="About PyAbel Studio", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menu_bar)

    def _build_workspace(self) -> None:
        bridge_bar = ttk.Frame(self, padding=(8, 6))
        bridge_bar.pack(fill=tk.X)

        ttk.Button(
            bridge_bar,
            text="Send Current Image to Analysis  →",
            command=self.send_generator_image_to_analysis,
        ).pack(side=tk.LEFT)

        backend_ok, backend_detail = verify_cython_backend()
        self.backend_ok = backend_ok
        self.backend_detail = backend_detail
        backend_text = (
            "Direct backend: Cython ✓"
            if backend_ok
            else "Direct backend: Python fallback"
        )
        ttk.Label(bridge_bar, text=backend_text).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        generator_tab = ttk.Frame(self.notebook)
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(generator_tab, text="Image Generator")
        self.notebook.add(analysis_tab, text="Analysis & Abel Inversion")

        self.generator = ImageGeneratorGUI(generator_tab, configure_window=False)
        master_module = _load_master_module()
        self.analysis = master_module.MainGUI(
            analysis_tab, configure_window=False
        )
        self.analysis.pack(fill=tk.BOTH, expand=True)

    def _open_analysis_image(self) -> None:
        self.notebook.select(1)
        self.analysis._getfilename()

    def _save_generator_image(self) -> None:
        self.notebook.select(0)
        self.generator.save_image()

    def _export_generator_model(self) -> None:
        self.notebook.select(0)
        self.generator.export_3d_model()

    def _generator_stage_name(self) -> str:
        image = self.generator.current_image
        for candidate, label in (
            (self.generator.noise_added, "noisy projection"),
            (self.generator.distorted_image, "distorted projection"),
            (self.generator.abel_transformed, "forward Abel projection"),
            (self.generator.generated_image, "generated slice"),
        ):
            if image is candidate and candidate is not None:
                return label
        return "current generator image"

    def send_generator_image_to_analysis(self) -> None:
        image = self.generator.current_image
        if image is None:
            messagebox.showinfo(
                "No generated image",
                "Generate or select an image in the Image Generator first.",
                parent=self,
            )
            self.notebook.select(0)
            return

        stage = self._generator_stage_name()
        if self.analysis.load_image_array(image.copy(), f"Image Generator — {stage}"):
            self.notebook.select(1)
            self.analysis.update_text(
                "Ready for centering, normalization, or Abel inversion.\n",
                clear_first=False,
            )

    def _show_about(self) -> None:
        messagebox.showinfo(
            f"About {APP_NAME}",
            f"{APP_NAME}\n"
            f"PyAbel {abel.__version__}\n\n"
            f"{self.backend_detail}\n\n"
            "Includes image generation, detector-shape distortion, centering, "
            "circularization, Abel transforms, and distribution analysis.",
            parent=self,
        )


def _run_gui_smoke_test() -> None:
    app = PyAbelStudio()
    try:
        app.geometry("1020x620")
        app.update_idletasks()
        app.analysis.fit_method_combo.set("RANSAC")
        app.analysis._toggle_ransac_frame()
        app.update()
        if app.analysis.ransac_frame.winfo_manager() != "pack":
            raise RuntimeError("RANSAC controls did not open")
        if app.analysis.control_canvas.yview()[0] <= 0:
            raise RuntimeError("analysis control column did not scroll to RANSAC")

        # Both control columns must scroll with the wheel over any control,
        # without the wheel changing the control it passes over.
        for tab, panel, control, name in (
            (0, app.generator.control_panel,
             app.generator.edit_distortion_button, "image generator"),
            (1, app.analysis.control_panel,
             app.analysis.fit_method_combo, "analysis"),
        ):
            app.notebook.select(tab)
            app.update()
            panel.canvas.yview_moveto(0.0)
            control.event_generate("<MouseWheel>", delta=-120)
            app.update()
            if panel.canvas.yview()[0] <= 0:
                raise RuntimeError(f"{name} control column did not scroll")
        if app.analysis.fit_method_combo.get() != "RANSAC":
            raise RuntimeError("wheel scrolling changed the fit method")

        # The 3D model export runs while the window is still mapped, because the
        # settings dialog takes a grab.
        app.generator.generator.add_peak(12.0, 100.0, 1.0, 2.0)
        model_dialog = Model3DDialog(app.generator)
        try:
            model_dialog.dots_var.set("500")
            model_dialog.detail_var.set("320 facets (smoothest)")
            model_dialog.export()
        finally:
            if model_dialog.winfo_exists():
                model_dialog.destroy()
        if model_dialog.result is None:
            raise RuntimeError("model dialog rejected its own settings")

        settings = dict(model_dialog.result)
        # Left at its default, the dialog sizes the whole sphere to match the
        # reference model however small this peak is.
        sphere = 2 * distribution_reach(app.generator.generator.peaks) * settings[
            "millimetres_per_pixel"
        ]
        if abs(sphere - DEFAULT_SPHERE_DIAMETER_MM) > 1e-6:
            raise RuntimeError("model dialog did not apply the default sphere size")
        with tempfile.TemporaryDirectory(prefix="pyabel-studio-model-") as folder:
            model_path = Path(folder) / "smoke.glb"
            summary = export_distribution_dots_glb(
                model_path,
                app.generator.generator.peaks,
                image_size=app.generator.generator.size,
                **settings,
            )
            if summary["stored_triangles"] != 320:
                raise RuntimeError("glTF export stored the wrong mesh")
            if summary["dots"] != 500 or summary["facets"] != 500 * 320:
                raise RuntimeError("glTF export produced an unexpected cloud")
            # Instancing stores the sphere once, so the file must stay far
            # below what 160,000 separately written triangles would need.
            if model_path.stat().st_size * 10 > 84 + 50 * summary["facets"]:
                raise RuntimeError("glTF export did not instance the sphere")
        app.generator.generator.clear_peaks()

        app.withdraw()
        test_image = np.eye(21, dtype=float)
        app.generator.current_image = test_image
        app.generator.generated_image = test_image
        app.send_generator_image_to_analysis()
        app.update_idletasks()
        if app.analysis.raw_IM is None or app.analysis.raw_IM.shape != (21, 21):
            raise RuntimeError("generator-to-analysis bridge failed")

        # Exercise both coordinated result layouts and their click/drag radius
        # inspection without running a costly transform in the GUI smoke test.
        radius = np.arange(11, dtype=float)
        distribution = np.exp(-((radius - 5.0) / 1.8) ** 2)
        app.analysis.AIM = SimpleNamespace(transform=test_image)
        app.analysis.inverse_method = "smoke-test"
        app.analysis.radial_coords = radius
        app.analysis.speed_distribution = distribution
        app.analysis.peak_threshold = 0.2
        app.analysis.peak_rel_height = 0.5
        app.analysis.peak_min_width = 1
        app.analysis.peak_prominence = 0.05
        app.analysis._update_speed_plot()
        if (app.analysis._linked_image_ax is None
                or app.analysis._linked_result_ax is None):
            raise RuntimeError("speed result did not retain the inverse image")
        speed_event = SimpleNamespace(
            inaxes=app.analysis._linked_result_ax,
            xdata=5.0,
            button=1,
        )
        app.analysis._on_result_pointer_press(speed_event)
        speed_event.xdata = 7.0
        app.analysis._on_result_pointer_move(speed_event)
        app.analysis._on_result_pointer_release(speed_event)
        if (app.analysis._radius_ring is None
                or not app.analysis._radius_ring.get_visible()
                or not np.isclose(app.analysis._radius_ring.get_radius(), 7.0)):
            raise RuntimeError("speed plot radius drag did not update the image ring")

        app.analysis.display_in_energy.set(True)
        app.analysis._anisotropy_plot_data = {
            "r_centers": [3.0, 5.0, 7.0],
            "beta2_values": [-0.4, 0.2, 1.1],
            "beta2_errors": [0.1, 0.08, 0.12],
            "step": 2,
            "rolling": False,
            "rbasex": False,
        }
        app.analysis._update_anisotropy_plot()
        if app.analysis._linked_result_ax.get_title():
            raise RuntimeError("anisotropy plot title was not removed")
        anisotropy_event = SimpleNamespace(
            inaxes=app.analysis._linked_result_ax,
            xdata=app.analysis._pixel_to_energy(5.0),
            button=1,
        )
        app.analysis._on_result_pointer_press(anisotropy_event)
        app.analysis._on_result_pointer_release(anisotropy_event)
        if (app.analysis._radius_ring is None
                or not np.isclose(app.analysis._radius_ring.get_radius(), 5.0)):
            raise RuntimeError("energy-axis anisotropy click selected the wrong radius")
    finally:
        app.destroy()


def _gui_smoke_test() -> tuple[bool, str]:
    _run_gui_smoke_test()
    return True, "combined GUI smoke test passed"


def _run_check(check) -> int:
    """Run a packaging check and report its outcome as the exit status.

    Packaged builds are windowed programs. An uncaught exception there opens
    a dialog that waits for someone to close it, which would stall a build.
    """
    try:
        ok, detail = check()
    except Exception:
        traceback.print_exc()
        return 1
    print(detail)
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify the bundled compiled direct-transform backend and exit",
    )
    parser.add_argument(
        "--gui-smoke-test",
        action="store_true",
        help="construct both GUI workspaces, test their bridge, and exit",
    )
    parser.add_argument(
        "--transform-smoke-test",
        action="store_true",
        help="run every Abel inversion method included in the analysis GUI",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_check(verify_cython_backend)
    if args.gui_smoke_test:
        return _run_check(_gui_smoke_test)
    if args.transform_smoke_test:
        return _run_check(verify_transform_methods)

    app = PyAbelStudio()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
