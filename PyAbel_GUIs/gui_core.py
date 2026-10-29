"""Testable numerical helpers shared by the PyAbel GUIs."""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import abel


FloatArray = NDArray[np.float64]

# NumPy 2.0 renamed trapz to trapezoid. The packaged builds pin NumPy 1.26, so
# both spellings have to work.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Public methods supported by the bundled PyAbel code, including the
# Nestor-Olsen method introduced on the upstream 0.10 development branch.
TRANSFORM_METHODS: Tuple[str, ...] = (
    "basex",
    "daun",
    "direct",
    "hansenlaw",
    "linbasex",
    "nestorolsen",
    "onion_bordas",
    "onion_peeling",
    "rbasex",
    "three_point",
    "two_point",
)

# Only these methods implement regularization. Other methods must not receive
# a misleading ``reg`` option, even when their implementation accepts **kwargs.
DEFAULT_TRANSFORM_OPTIONS: Dict[str, Dict[str, Any]] = {
    "basex": {"reg": 100.0},
    "daun": {"reg": 100.0},
    "rbasex": {"reg": ("diff", 100.0)},
}


class RadialDistortionModel:
    """Editable periodic radial-scale profile used by the ImageGen canvas."""

    SYMMETRY_AXES: Dict[str, float] = {
        "Horizontal": 0.0,
        "Diagonal ↗": np.pi / 4,
        "Vertical": np.pi / 2,
        "Diagonal ↖": 3 * np.pi / 4,
    }

    def __init__(self, point_count: int = 24) -> None:
        if point_count < 8 or point_count % 8:
            raise ValueError("Distortion point count must be a multiple of 8.")
        self.angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)
        self.scales = np.ones(point_count, dtype=float)
        self.selected_axis = "Vertical"

    @property
    def point_count(self) -> int:
        return int(self.angles.size)

    def reset(self) -> None:
        self.scales.fill(1.0)

    def set_axis(self, name: str) -> None:
        if name not in self.SYMMETRY_AXES:
            raise ValueError(f"Unknown symmetry axis {name!r}.")
        self.selected_axis = name

    def mirrored_index(self, index: int, axis_name: Optional[str] = None) -> int:
        index = int(index) % self.point_count
        name = self.selected_axis if axis_name is None else axis_name
        if name not in self.SYMMETRY_AXES:
            raise ValueError(f"Unknown symmetry axis {name!r}.")
        reflected_angle = 2 * self.SYMMETRY_AXES[name] - self.angles[index]
        step = 2 * np.pi / self.point_count
        return int(np.rint(np.mod(reflected_angle, 2 * np.pi) / step)) % self.point_count

    def set_scale(
        self,
        index: int,
        scale: float,
        *,
        symmetry: bool = False,
        axis_name: Optional[str] = None,
    ) -> None:
        scale = float(scale)
        if not np.isfinite(scale) or not 0.5 <= scale <= 2.0:
            raise ValueError("Distortion scales must be between 0.5 and 2.0.")
        index = int(index) % self.point_count
        self.scales[index] = scale
        if symmetry:
            self.scales[self.mirrored_index(index, axis_name)] = scale

    def set_ellipse(self, horizontal_scale: float, vertical_scale: float) -> None:
        scales = np.asarray([horizontal_scale, vertical_scale], dtype=float)
        if not np.all(np.isfinite(scales)) or np.any(scales < 0.5) or np.any(scales > 2.0):
            raise ValueError("Distortion scales must be between 0.5 and 2.0.")
        cosine = np.cos(self.angles)
        sine = np.sin(self.angles)
        self.scales = 1 / np.sqrt(
            (cosine / horizontal_scale) ** 2
            + (sine / vertical_scale) ** 2
        )

    def profile(self) -> Tuple[FloatArray, FloatArray]:
        return self.angles.copy(), self.scales.copy()


def prepare_image(image: NDArray[Any], *, make_odd: bool = True) -> FloatArray:
    """Validate a loaded image and optionally crop even dimensions to odd."""
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = np.mean(array, axis=2)
    if array.ndim != 2:
        raise ValueError("Image data must be a two-dimensional numeric array.")
    if min(array.shape) < 3:
        raise ValueError("Image data must be at least 3 x 3 pixels.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Image data contains NaN or infinite values.")

    if make_odd:
        rows = array.shape[0] - (array.shape[0] % 2 == 0)
        cols = array.shape[1] - (array.shape[1] % 2 == 0)
        array = array[:rows, :cols]

    return np.array(array, dtype=float, copy=True)


def distort_image(
    image: NDArray[Any],
    *,
    horizontal_scale: float = 1.0,
    vertical_scale: float = 1.0,
    order: int = 3,
) -> FloatArray:
    """Apply a center-preserving elliptical detector distortion.

    A scale below one compresses the apparent image along that axis; a scale
    above one stretches it. The image shape and geometric center are retained.
    """
    array = prepare_image(image, make_odd=False)
    scales = np.asarray([horizontal_scale, vertical_scale], dtype=float)
    if not np.all(np.isfinite(scales)):
        raise ValueError("Distortion scales must be finite numbers.")
    if np.any(scales < 0.5) or np.any(scales > 2.0):
        raise ValueError("Distortion scales must be between 0.5 and 2.0.")
    valid_order = (
        not isinstance(order, bool)
        and isinstance(order, (int, np.integer))
        and 0 <= order <= 5
    )
    if not valid_order:
        raise ValueError("Interpolation order must be an integer from 0 to 5.")

    rows, cols = array.shape
    center_y = (rows - 1) / 2
    center_x = (cols - 1) / 2
    output_y, output_x = np.indices(array.shape, dtype=float)
    source_y = (output_y - center_y) / vertical_scale + center_y
    source_x = (output_x - center_x) / horizontal_scale + center_x
    distorted = map_coordinates(
        array,
        (source_y, source_x),
        order=int(order),
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    if np.min(array) >= 0:
        distorted = np.maximum(distorted, 0)
    return np.asarray(distorted, dtype=float)


def distort_image_radial(
    image: NDArray[Any],
    angles: NDArray[Any],
    radial_scales: NDArray[Any],
    *,
    order: int = 3,
) -> FloatArray:
    """Distort an image using a smooth periodic radial scale versus angle."""
    array = prepare_image(image, make_odd=False)
    angles = np.asarray(angles, dtype=float)
    scales = np.asarray(radial_scales, dtype=float)
    if angles.ndim != 1 or scales.ndim != 1 or angles.shape != scales.shape:
        raise ValueError("Distortion angles and scales must be matching 1D arrays.")
    if angles.size < 4:
        raise ValueError("At least four angular distortion points are required.")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(scales)):
        raise ValueError("Distortion angles and scales must be finite.")
    if np.any(scales < 0.5) or np.any(scales > 2.0):
        raise ValueError("Distortion scales must be between 0.5 and 2.0.")
    valid_order = (
        not isinstance(order, bool)
        and isinstance(order, (int, np.integer))
        and 0 <= order <= 5
    )
    if not valid_order:
        raise ValueError("Interpolation order must be an integer from 0 to 5.")

    periodic_angles = np.mod(angles, 2 * np.pi)
    sort_order = np.argsort(periodic_angles)
    periodic_angles = periodic_angles[sort_order]
    scales = scales[sort_order]
    if np.any(np.diff(periodic_angles) < 1e-12):
        raise ValueError("Distortion profile angles must be unique modulo 2π.")

    spline_angles = np.append(periodic_angles, periodic_angles[0] + 2 * np.pi)
    spline_scales = np.append(scales, scales[0])
    profile = CubicSpline(spline_angles, spline_scales, bc_type="periodic")

    rows, cols = array.shape
    center_y = (rows - 1) / 2
    center_x = (cols - 1) / 2
    output_y, output_x = np.indices(array.shape, dtype=float)
    delta_x = output_x - center_x
    delta_y = output_y - center_y
    output_angle = np.mod(np.arctan2(-delta_y, delta_x), 2 * np.pi)
    output_angle = np.where(
        output_angle < periodic_angles[0],
        output_angle + 2 * np.pi,
        output_angle,
    )
    scale_grid = np.clip(profile(output_angle), 0.5, 2.0)
    source_x = delta_x / scale_grid + center_x
    source_y = delta_y / scale_grid + center_y
    distorted = map_coordinates(
        array,
        (source_y, source_x),
        order=int(order),
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    if np.min(array) >= 0:
        distorted = np.maximum(distorted, 0)
    return np.asarray(distorted, dtype=float)


def circularize_image(
    image: NDArray[Any],
    *,
    method: str = "lsq",
    radial_range: Optional[Tuple[float, float]] = None,
    tol: float = 0.0,
    dr: float = 0.5,
    dt: float = 0.1,
    return_correction: bool = False,
) -> Any:
    """Estimate and correct radial distortion in an already centered image."""
    array = prepare_image(image, make_odd=False)
    if method not in {"argmax", "lsq"}:
        raise ValueError("Circularization method must be 'argmax' or 'lsq'.")

    numeric = np.asarray([tol, dr, dt], dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Circularization tolerance and grid steps must be finite.")
    if tol < 0:
        raise ValueError("Circularization tolerance cannot be negative.")
    if dr <= 0 or dt <= 0:
        raise ValueError("Circularization grid steps must be greater than zero.")

    clean_range: Optional[Tuple[float, float]] = None
    if radial_range is not None:
        if len(radial_range) != 2:
            raise ValueError("Radial range must contain minimum and maximum radii.")
        rmin, rmax = (float(value) for value in radial_range)
        if not np.isfinite(rmin) or not np.isfinite(rmax):
            raise ValueError("Circularization radii must be finite.")
        if rmin < 0 or rmax <= rmin:
            raise ValueError("Circularization requires 0 <= rmin < rmax.")
        if rmax > min(array.shape) / 2:
            raise ValueError(
                f"Circularization rmax cannot exceed {min(array.shape) / 2:g} pixels."
            )
        clean_range = (rmin, rmax)

    result = abel.tools.circularize.circularize_image(
        array,
        method=method,
        radial_range=clean_range,
        dr=dr,
        dt=dt,
        tol=tol,
        return_correction=return_correction,
    )
    if not return_correction:
        return prepare_image(result, make_odd=False)

    corrected, angles, factors, correction_function = result
    angles = np.asarray(angles, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(factors)):
        raise ValueError("PyAbel returned a non-finite circularization correction.")
    if np.any(factors <= 0):
        raise ValueError("PyAbel returned a non-positive circularization correction.")
    return (
        prepare_image(corrected, make_odd=False),
        angles,
        factors,
        correction_function,
    )


def inverse_transform(
    image: NDArray[Any],
    method: str,
    *,
    symmetry_axis: Tuple[int, int] = (0, 1),
) -> Any:
    """Run one supported inverse Abel transform with valid method options."""
    if method not in TRANSFORM_METHODS:
        raise ValueError(
            f"Unsupported Abel method {method!r}. "
            f"Choose one of: {', '.join(TRANSFORM_METHODS)}."
        )

    array = prepare_image(image, make_odd=False)
    options = dict(DEFAULT_TRANSFORM_OPTIONS.get(method, {}))
    return abel.Transform(
        array,
        method=method,
        direction="inverse",
        symmetry_axis=symmetry_axis,
        transform_options=options,
    )


def extract_speed_distribution(
    transform: Any,
    method: str,
    *,
    image: Optional[FloatArray] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Extract and safely normalize a radial speed distribution."""
    if method == "linbasex":
        radial = np.asarray(transform.radial, dtype=float)
        speed = np.asarray(transform.Beta[0], dtype=float)
    elif method == "rbasex":
        radial, speed, _ = transform.distr.rIbeta()
        radial = np.asarray(radial, dtype=float)
        speed = np.asarray(speed, dtype=float)
    else:
        source = transform.transform if image is None else image
        radial, speed = abel.tools.vmi.angular_integration_3D(source)
        radial = np.asarray(radial, dtype=float)
        speed = np.asarray(speed, dtype=float)

    if radial.ndim != 1 or speed.ndim != 1 or radial.shape != speed.shape:
        raise ValueError("PyAbel returned an invalid speed-distribution shape.")
    if not np.all(np.isfinite(radial)) or not np.all(np.isfinite(speed)):
        raise ValueError("PyAbel returned NaN or infinite speed data.")

    scale = float(np.max(speed)) if speed.size else 0.0
    if scale <= 0 and speed.size:
        scale = float(np.max(np.abs(speed)))
    if scale > 0:
        speed = speed / scale
    else:
        speed = np.zeros_like(speed)
    return radial, speed


def validate_peak_parameters(
    radius: float,
    intensity: float,
    beta2: float,
    width: float,
    *,
    image_size: int,
) -> Tuple[float, float, float, float]:
    """Validate one physical synthetic peak and return normalized floats."""
    values = np.asarray([radius, intensity, beta2, width], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Peak parameters must all be finite numbers.")
    if radius < 0 or radius >= image_size / 2:
        raise ValueError(
            f"Radius must be at least 0 and less than {image_size / 2:g} pixels."
        )
    if intensity < 0:
        raise ValueError("Intensity cannot be negative.")
    if not -1 <= beta2 <= 2:
        raise ValueError("Beta2 must be between -1 and 2 for a physical P2 distribution.")
    if width <= 0:
        raise ValueError("Peak width must be greater than zero.")
    return float(radius), float(intensity), float(beta2), float(width)


# ---------------------------------------------------------------------------
# Three-dimensional model export
#
# The generated image is the central slice of a cylindrically symmetric 3D
# distribution, so re-evaluating the peak model in three dimensions rebuilds
# the Newton sphere that the synthetic data represents.
#
# The model is written as a cloud of small spheres rather than as a surface.
# A surface can only show one intensity and has to discard the rest, whereas
# dots are drawn from the distribution as a probability density, so the number
# of dots per unit volume is the intensity everywhere. Each dot is one particle
# of the Newton sphere, which is also how a detector records the experiment.
# ---------------------------------------------------------------------------


def peak_amplitude(
    radius_pixels: float,
    width_pixels: float,
    intensity: float,
    *,
    image_size: int,
) -> float:
    """Return the scale factor giving a peak a radius-independent 3D yield.

    The extra radial weight converts the sum over a 2D slice into the radial
    weighting needed for equal particle yield at different radii, so that the
    same factor applies to the slice and to the volume it represents.
    """
    center = (image_size - 1) / 2
    y, x = np.indices((image_size, image_size))
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    radial_norm = np.exp(-(distance - radius_pixels) ** 2 / (2 * width_pixels ** 2))
    volume = np.sum(radial_norm * distance)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Peak normalization failed; check radius and width.")
    return float(intensity * 2000 / volume)


def _prepare_peaks(
    peaks: Sequence[Sequence[float]], image_size: int
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """Validate a peak list and return it with each peak's amplitude."""
    if not len(peaks):
        raise ValueError("Add at least one peak before exporting a 3D model.")
    validated = [
        validate_peak_parameters(
            radius, intensity, beta2, width, image_size=image_size
        )
        for radius, intensity, beta2, width in peaks
    ]
    amplitudes = [
        peak_amplitude(radius, width, intensity, image_size=image_size)
        for radius, intensity, _, width in validated
    ]
    return validated, amplitudes


def evaluate_peak_field(
    points: NDArray[Any],
    peaks: Sequence[Sequence[float]],
    amplitudes: Sequence[float],
) -> FloatArray:
    """Evaluate the 3D distribution at arbitrary points rather than on a grid.

    ``points`` is an N x 3 array of x, y, z coordinates in pixels about the
    distribution center, with y along the cylindrical symmetry axis.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Field sample points must be an N x 3 array.")

    radius = np.linalg.norm(points, axis=1)
    cosine = np.divide(
        points[:, 1], radius, out=np.ones_like(radius), where=radius > 0
    )
    legendre_p2 = 0.5 * (3.0 * cosine ** 2 - 1.0)
    values = np.zeros(radius.shape, dtype=float)
    for (peak_radius, _, beta2, width), amplitude in zip(peaks, amplitudes):
        values += (
            amplitude
            * np.exp(-((radius - peak_radius) ** 2) / (2 * width ** 2))
            * (1 + beta2 * legendre_p2)
        )
    return values


def count_mesh_surfaces(vertices: NDArray[Any], faces: NDArray[Any]) -> int:
    """Return how many separate closed surfaces a mesh contains.

    A dot cloud that has not been merged reports one surface per dot.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return 0
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    columns = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adjacency = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)),
        shape=(len(vertices), len(vertices)),
    )
    _, labels = connected_components(adjacency, directed=False)
    # Count only labels reached by a facet, so any unreferenced vertex passed in
    # by a caller does not register as a surface of its own.
    return int(np.unique(labels[np.unique(faces)]).size)


def facet_normals(vertices: NDArray[Any], faces: NDArray[Any]) -> FloatArray:
    """Return unit normals for each triangle, using its winding order."""
    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)


def write_stl(
    path: Any,
    vertices: NDArray[Any],
    faces: NDArray[Any],
    *,
    binary: bool = True,
    name: str = "pyabel_distribution",
) -> int:
    """Write a triangle mesh as an STL file and return the facet count."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be an N x 3 array of coordinates.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be an N x 3 array of vertex indices.")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Mesh faces reference vertices outside the vertex array.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices contain NaN or infinite values.")
    if not faces.size:
        raise ValueError("There are no triangles to write.")

    triangles = vertices[faces]
    normals = facet_normals(vertices, faces)

    if binary:
        record = np.zeros(
            len(faces),
            dtype=np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attributes", "<u2"),
                ]
            ),
        )
        if record.dtype.itemsize != 50:
            raise RuntimeError("Binary STL facet records must be 50 bytes.")
        record["normal"] = normals
        record["vertices"] = triangles
        # A binary header must not begin with "solid" or readers treat the file
        # as ASCII.
        header = f"PyAbel Studio 3D distribution: {name}".encode("ascii", "replace")
        with open(path, "wb") as handle:
            handle.write(header[:80].ljust(80, b"\0"))
            handle.write(np.asarray(len(faces), dtype="<u4").tobytes())
            handle.write(record.tobytes())
        return int(len(faces))

    lines = [f"solid {name}"]
    for normal, triangle in zip(normals, triangles):
        lines.append(
            f"facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}"
        )
        lines.append("  outer loop")
        for vertex in triangle:
            lines.append(
                f"    vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}"
            )
        lines.append("  endloop")
        lines.append("endfacet")
    lines.append(f"endsolid {name}")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return int(len(faces))


DOT_MAX_COUNT = 400_000
DOT_MIN_COUNT = 100
DOT_MAX_SUBDIVISIONS = 2

# Image pixels are not a physical size, so an exported model needs a scale. One
# tenth of a millimetre per pixel turns a 490 px distribution into a 49 mm
# object, which is a sensible thing to print or hold.
MILLIMETRES_PER_PIXEL = 0.1
# STL carries no unit at all and is read as millimetres by every slicer, while
# glTF is defined in metres. Writing each in its own unit makes both open at
# the same real-world size.
_METRES_PER_MILLIMETRE = 0.001


def _validate_scale(millimetres_per_pixel: float) -> float:
    """Check the pixel-to-millimetre scale of an exported model."""
    scale = float(millimetres_per_pixel)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be a positive number of millimetres per pixel.")
    if scale > 100:
        raise ValueError("Scale must be at most 100 millimetres per pixel.")
    return scale


# A fixed pixel scale makes the model grow and shrink with its peaks, so the
# export dialog asks for the size of the whole sphere instead. The default is
# taken from the reference model TestPrint.glb: its outermost peak sits at
# 200 px with a 10 px width, and the cloud is 0.1059 mm per pixel, which makes
# the sphere 48.7 mm across.
DEFAULT_SPHERE_DIAMETER_MM = 48.7


def distribution_reach(peaks: Sequence[Sequence[float]]) -> float:
    """Return the radius in pixels that the dot cloud is taken to fill.

    Three widths past the outermost peak holds all but about one dot in a
    thousand of that shell, so this is the visible edge of the sphere.
    """
    if not len(peaks):
        raise ValueError("Add at least one peak before exporting a 3D model.")
    return max(float(radius) + 3 * float(width) for radius, _, _, width in peaks)


def millimetres_per_pixel_for_diameter(
    peaks: Sequence[Sequence[float]], sphere_diameter_mm: float
) -> float:
    """Return the scale that makes the whole distribution a given diameter."""
    diameter = float(sphere_diameter_mm)
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Sphere diameter must be a positive number of millimetres.")
    return _validate_scale(diameter / (2 * distribution_reach(peaks)))

# Vertices and outward-wound faces of a regular icosahedron, the coarsest
# sphere whose facets are all the same size.
_ICOSAHEDRON_VERTICES = np.array(
    [
        [-1, (1 + 5 ** 0.5) / 2, 0], [1, (1 + 5 ** 0.5) / 2, 0],
        [-1, -(1 + 5 ** 0.5) / 2, 0], [1, -(1 + 5 ** 0.5) / 2, 0],
        [0, -1, (1 + 5 ** 0.5) / 2], [0, 1, (1 + 5 ** 0.5) / 2],
        [0, -1, -(1 + 5 ** 0.5) / 2], [0, 1, -(1 + 5 ** 0.5) / 2],
        [(1 + 5 ** 0.5) / 2, 0, -1], [(1 + 5 ** 0.5) / 2, 0, 1],
        [-(1 + 5 ** 0.5) / 2, 0, -1], [-(1 + 5 ** 0.5) / 2, 0, 1],
    ],
    dtype=float,
)
_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)


def unit_icosphere(subdivisions: int = 0) -> Tuple[FloatArray, NDArray[np.int64]]:
    """Return a unit sphere mesh with 20 * 4**subdivisions triangles."""
    valid = (
        not isinstance(subdivisions, bool)
        and isinstance(subdivisions, (int, np.integer))
        and 0 <= subdivisions <= DOT_MAX_SUBDIVISIONS
    )
    if not valid:
        raise ValueError(
            f"Sphere subdivisions must be an integer from 0 to "
            f"{DOT_MAX_SUBDIVISIONS}."
        )

    vertices = [
        vertex / np.linalg.norm(vertex) for vertex in _ICOSAHEDRON_VERTICES
    ]
    faces = _ICOSAHEDRON_FACES
    for _ in range(int(subdivisions)):
        midpoints: Dict[Tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            key = (min(first, second), max(first, second))
            if key not in midpoints:
                middle = vertices[first] + vertices[second]
                vertices.append(middle / np.linalg.norm(middle))
                midpoints[key] = len(vertices) - 1
            return midpoints[key]

        divided = np.empty((len(faces) * 4, 3), dtype=np.int64)
        for index, (first, second, third) in enumerate(faces):
            a = midpoint(first, second)
            b = midpoint(second, third)
            c = midpoint(third, first)
            divided[4 * index] = (first, a, c)
            divided[4 * index + 1] = (second, b, a)
            divided[4 * index + 2] = (third, c, b)
            divided[4 * index + 3] = (a, b, c)
        faces = divided

    return np.asarray(vertices, dtype=float), faces


def _sample_from_density(
    coordinate: FloatArray,
    density: FloatArray,
    count: int,
    generator: np.random.Generator,
) -> FloatArray:
    """Draw samples from a tabulated 1D density by inverting its integral."""
    cumulative = np.cumsum(density)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot sample from a density that integrates to zero.")
    return np.interp(generator.random(count), cumulative / total, coordinate)


def sample_distribution_points(
    peaks: Sequence[Sequence[float]],
    *,
    image_size: int,
    count: int = 25_000,
    seed: Optional[int] = None,
    resolution: int = 4096,
) -> FloatArray:
    """Draw points from the 3D distribution, treating it as a probability density.

    Sampling directly in spherical coordinates gives positions with no
    underlying grid at all, so the result carries no lattice to hide. Returns
    an N x 3 array in pixels about the center, with y along the symmetry axis.
    """
    validated, amplitudes = _prepare_peaks(peaks, image_size)
    valid_count = (
        not isinstance(count, bool)
        and isinstance(count, (int, np.integer))
        and DOT_MIN_COUNT <= count <= DOT_MAX_COUNT
    )
    if not valid_count:
        raise ValueError(
            f"Dot count must be an integer from {DOT_MIN_COUNT} to {DOT_MAX_COUNT}."
        )

    generator = np.random.default_rng(seed)
    count = int(count)

    # The angular factor integrates to the same value whatever beta2 is, because
    # the second Legendre polynomial averages to zero over the sphere. Relative
    # peak yield therefore depends only on the radial part.
    radial_integrals = []
    radial_grids = []
    for (radius, _, _, width), amplitude in zip(validated, amplitudes):
        low = max(0.0, radius - 6 * width)
        grid = np.linspace(low, radius + 6 * width, resolution)
        # The volume element in three dimensions carries a factor of R squared.
        density = grid ** 2 * np.exp(-((grid - radius) ** 2) / (2 * width ** 2))
        radial_grids.append((grid, density))
        radial_integrals.append(amplitude * float(_trapezoid(density, grid)))

    weights = np.asarray(radial_integrals, dtype=float)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("The peak list carries no three-dimensional yield.")
    per_peak = generator.multinomial(count, weights / weights.sum())

    cosine_grid = np.linspace(-1.0, 1.0, resolution)
    legendre_p2 = 0.5 * (3.0 * cosine_grid ** 2 - 1.0)

    blocks = []
    for (_, _, beta2, _), (grid, density), share in zip(
        validated, radial_grids, per_peak
    ):
        if share == 0:
            continue
        radius = _sample_from_density(grid, density, int(share), generator)
        # Sampling cos(theta) rather than theta already includes sin(theta) from
        # the solid angle, so the remaining density is just the angular factor.
        cosine = _sample_from_density(
            cosine_grid,
            np.clip(1 + beta2 * legendre_p2, 0.0, None),
            int(share),
            generator,
        )
        azimuth = generator.random(int(share)) * 2 * np.pi
        perpendicular = radius * np.sqrt(np.clip(1 - cosine ** 2, 0.0, None))
        blocks.append(
            np.stack(
                [
                    perpendicular * np.cos(azimuth),
                    radius * cosine,
                    perpendicular * np.sin(azimuth),
                ],
                axis=1,
            )
        )

    points = np.concatenate(blocks) if blocks else np.zeros((0, 3), dtype=float)
    generator.shuffle(points)
    return points


def dot_cloud_mesh(
    points: NDArray[Any],
    *,
    diameter: float = 1.0,
    subdivisions: int = 0,
) -> Tuple[FloatArray, NDArray[np.int64]]:
    """Place one small sphere at every point and return the combined mesh."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Dot positions must be an N x 3 array.")
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Dot diameter must be a positive number of pixels.")
    if not len(points):
        raise ValueError("There are no dots to write.")

    sphere_vertices, sphere_faces = unit_icosphere(subdivisions)
    sphere_vertices = sphere_vertices * (float(diameter) / 2)

    vertices = (points[:, None, :] + sphere_vertices[None, :, :]).reshape(-1, 3)
    offsets = (np.arange(len(points), dtype=np.int64) * len(sphere_vertices))
    faces = (sphere_faces[None, :, :] + offsets[:, None, None]).reshape(-1, 3)
    return vertices, faces


def export_distribution_dots_stl(
    path: Any,
    peaks: Sequence[Sequence[float]],
    *,
    image_size: int,
    count: int = 25_000,
    diameter: float = 1.0,
    subdivisions: int = 0,
    binary: bool = True,
    seed: Optional[int] = None,
    millimetres_per_pixel: float = MILLIMETRES_PER_PIXEL,
    name: str = "pyabel_distribution",
) -> Dict[str, Any]:
    """Write the distribution as a cloud of small spheres, in millimetres.

    Dots are drawn from the distribution itself, so where it is twice as intense
    there are twice as many dots, and the model carries the density rather than
    a single intensity level. The spheres are separate solids that overlap where
    the density is high, so the file is a valid collection of closed bodies
    rather than one manifold shell.
    """
    scale = _validate_scale(millimetres_per_pixel)
    points = sample_distribution_points(
        peaks, image_size=image_size, count=count, seed=seed
    )
    vertices, faces = dot_cloud_mesh(
        points, diameter=diameter, subdivisions=subdivisions
    )
    # STL has no unit field, so the numbers themselves are the millimetres.
    facets = write_stl(path, vertices * scale, faces, binary=binary, name=name)

    radii = np.linalg.norm(points, axis=1)
    return {
        "path": str(path),
        "mode": "dots",
        "format": "stl",
        "units": "mm",
        "millimetres_per_pixel": scale,
        "dots": int(len(points)),
        "facets": facets,
        "vertices": int(len(vertices)),
        "facets_per_dot": int(len(faces) // len(points)),
        "diameter": float(diameter),
        "subdivisions": int(subdivisions),
        "seed": seed,
        "radius_range": (float(radii.min()), float(radii.max())),
        "bounds": (
            tuple(float(value) * scale for value in vertices.min(axis=0)),
            tuple(float(value) * scale for value in vertices.max(axis=0)),
        ),
        "binary": bool(binary),
    }


# ---------------------------------------------------------------------------
# glTF export
#
# STL stores every triangle of every sphere in full, so a cloud costs its dot
# count times its facet count. glTF separates geometry from placement: the
# sphere is stored once as a mesh and each dot is a node that refers to it. The
# facet count then barely affects the file at all, because only one sphere is
# ever written.
# ---------------------------------------------------------------------------

_GLB_MAGIC = 0x46546C67
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK = 0x004E4942
_GLTF_FLOAT = 5126
_GLTF_UNSIGNED_SHORT = 5123
_GLTF_UNSIGNED_INT = 5125
_GLTF_ARRAY_BUFFER = 34962
_GLTF_ELEMENT_ARRAY_BUFFER = 34963

DEFAULT_DOT_COLOR = (0.35, 0.60, 0.85, 1.0)


def vertex_normals(vertices: NDArray[Any], faces: NDArray[Any]) -> FloatArray:
    """Return smooth per-vertex normals by area-weighted facet averaging."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices)
    triangles = vertices[faces]
    # The un-normalized cross product is twice the facet area, which weights
    # each facet by how much of the surface it represents.
    weighted = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], weighted)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)


def _pad_to_four(payload: bytes, filler: bytes) -> bytes:
    remainder = len(payload) % 4
    return payload if remainder == 0 else payload + filler * (4 - remainder)


def write_glb(
    path: Any,
    vertices: NDArray[Any],
    faces: NDArray[Any],
    placements: NDArray[Any],
    *,
    name: str = "pyabel_distribution",
    color: Sequence[float] = DEFAULT_DOT_COLOR,
    precision: Optional[float] = None,
) -> Dict[str, Any]:
    """Write one mesh instanced at many positions as a binary glTF file.

    The mesh is stored a single time and every placement becomes a node that
    points at it, so the file size depends on how many instances there are and
    barely on how detailed each one is.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    placements = np.asarray(placements, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be an N x 3 array of coordinates.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be an N x 3 array of vertex indices.")
    if placements.ndim != 2 or placements.shape[1] != 3:
        raise ValueError("Instance placements must be an N x 3 array.")
    if not len(faces) or not len(placements):
        raise ValueError("There is nothing to write.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh faces reference vertices outside the vertex array.")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(placements)):
        raise ValueError("Mesh data contains NaN or infinite values.")
    if len(color) != 4 or not all(0 <= float(value) <= 1 for value in color):
        raise ValueError("Colour must be four values between 0 and 1.")
    if precision is not None and (not np.isfinite(precision) or precision <= 0):
        raise ValueError("Placement precision must be a positive distance.")

    normals = vertex_normals(vertices, faces).astype(np.float32)
    small = len(vertices) <= 65535
    index_type = _GLTF_UNSIGNED_SHORT if small else _GLTF_UNSIGNED_INT
    # glTF buffers are always little-endian, so ask for that explicitly rather
    # than relying on the byte order of the machine writing the file.
    indices = faces.astype("<u2" if small else "<u4").ravel()

    blocks = [
        vertices.astype("<f4").tobytes(),
        normals.astype("<f4").tobytes(),
        indices.tobytes(),
    ]
    views = []
    buffer = bytearray()
    for block, target in zip(
        blocks,
        (_GLTF_ARRAY_BUFFER, _GLTF_ARRAY_BUFFER, _GLTF_ELEMENT_ARRAY_BUFFER),
    ):
        while len(buffer) % 4:
            buffer.append(0)
        views.append(
            {
                "buffer": 0,
                "byteOffset": len(buffer),
                "byteLength": len(block),
                "target": target,
            }
        )
        buffer.extend(block)

    # Placements are rounded to keep the node list compact, but the number of
    # decimals has to follow the units the caller is writing in. A fixed count
    # would quantise the positions onto a grid whenever the model is small in
    # its own units, which is precisely the lattice a sampled cloud avoids.
    decimals = 6 if precision is None else max(
        0, int(np.ceil(-np.log10(float(precision))))
    )

    # Every instance hangs off one root node, so a viewer shows a single object
    # that can be selected and moved as a whole, with each dot still a
    # separately identified child underneath it. Listing the instances as roots
    # instead would fill the scene tree with thousands of top-level entries.
    nodes = [{"name": name, "children": list(range(1, len(placements) + 1))}]
    nodes += [
        {
            "mesh": 0,
            "translation": [round(float(value), decimals) for value in position],
        }
        for position in placements
    ]
    document = {
        "asset": {"version": "2.0", "generator": f"PyAbel Studio ({name})"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [
            {
                # Naming the mesh separately from the root keeps an outliner
                # readable: the group takes the model name and the instances
                # under it are numbered from the mesh name.
                "name": "dot",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "dot",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [float(value) for value in color],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.55,
                },
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _GLTF_FLOAT,
                "count": int(len(vertices)),
                "type": "VEC3",
                # glTF requires bounds on a position accessor.
                "min": [float(value) for value in vertices.min(axis=0)],
                "max": [float(value) for value in vertices.max(axis=0)],
            },
            {
                "bufferView": 1,
                "componentType": _GLTF_FLOAT,
                "count": int(len(vertices)),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": index_type,
                "count": int(indices.size),
                "type": "SCALAR",
            },
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(buffer)}],
    }

    json_chunk = _pad_to_four(
        json.dumps(document, separators=(",", ":")).encode("utf-8"), b" "
    )
    binary_chunk = _pad_to_four(bytes(buffer), b"\0")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)

    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        handle.write(struct.pack("<II", len(json_chunk), _GLB_JSON_CHUNK))
        handle.write(json_chunk)
        handle.write(struct.pack("<II", len(binary_chunk), _GLB_BIN_CHUNK))
        handle.write(binary_chunk)

    return {
        "instances": int(len(placements)),
        "mesh_triangles": int(len(faces)),
        "total_triangles": int(len(faces) * len(placements)),
        "bytes": total,
    }


def export_distribution_dots_glb(
    path: Any,
    peaks: Sequence[Sequence[float]],
    *,
    image_size: int,
    count: int = 25_000,
    diameter: float = 1.0,
    subdivisions: int = 0,
    seed: Optional[int] = None,
    millimetres_per_pixel: float = MILLIMETRES_PER_PIXEL,
    color: Sequence[float] = DEFAULT_DOT_COLOR,
    name: str = "pyabel_distribution",
) -> Dict[str, Any]:
    """Write the dot cloud as instanced glTF, storing the sphere only once."""
    scale = _validate_scale(millimetres_per_pixel)
    points = sample_distribution_points(
        peaks, image_size=image_size, count=count, seed=seed
    )
    sphere_vertices, sphere_faces = unit_icosphere(subdivisions)
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Dot diameter must be a positive number of pixels.")
    sphere_vertices = sphere_vertices * (float(diameter) / 2)

    # glTF is defined in metres, so the same physical model is a thousand times
    # smaller in its own numbers than it is in the millimetre-based STL.
    to_metres = scale * _METRES_PER_MILLIMETRE
    written = write_glb(
        path,
        sphere_vertices * to_metres,
        sphere_faces,
        points * to_metres,
        name=name,
        color=color,
        # Hold a hundredth of a pixel, so rounding never becomes visible.
        precision=0.01 * to_metres,
    )
    radii = np.linalg.norm(points, axis=1)
    low = (points.min(axis=0) - diameter / 2) * scale
    high = (points.max(axis=0) + diameter / 2) * scale
    return {
        "path": str(path),
        "mode": "dots",
        "format": "glb",
        "units": "m",
        "millimetres_per_pixel": scale,
        "dots": int(len(points)),
        "facets": written["total_triangles"],
        "facets_per_dot": written["mesh_triangles"],
        "stored_triangles": written["mesh_triangles"],
        "bytes": written["bytes"],
        "diameter": float(diameter),
        "subdivisions": int(subdivisions),
        "seed": seed,
        "radius_range": (float(radii.min()), float(radii.max())),
        "bounds": (
            tuple(float(value) for value in low),
            tuple(float(value) for value in high),
        ),
        "binary": True,
    }
