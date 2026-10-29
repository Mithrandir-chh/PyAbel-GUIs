import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import abel
from PyAbel_GUIs.Supp_ImageGen_GUI import ImageGenerator, SaveDialog
from PyAbel_GUIs.called_functions.anni import Anora
from PyAbel_GUIs.gui_core import (
    RadialDistortionModel,
    DEFAULT_SPHERE_DIAMETER_MM,
    DOT_MAX_COUNT,
    DOT_MAX_SUBDIVISIONS,
    DOT_MIN_COUNT,
    MILLIMETRES_PER_PIXEL,
    TRANSFORM_METHODS,
    _prepare_peaks,
    _trapezoid,
    circularize_image,
    count_mesh_surfaces,
    distort_image,
    distort_image_radial,
    distribution_reach,
    dot_cloud_mesh,
    evaluate_peak_field,
    export_distribution_dots_glb,
    export_distribution_dots_stl,
    extract_speed_distribution,
    inverse_transform,
    millimetres_per_pixel_for_diameter,
    prepare_image,
    sample_distribution_points,
    unit_icosphere,
    vertex_normals,
    validate_peak_parameters,
)


STL_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attributes", "<u2")]
)


def read_glb(path):
    """Parse a binary glTF file back into its JSON document and buffer."""
    payload = Path(path).read_bytes()
    magic, version, length = struct.unpack("<III", payload[:12])
    assert magic == 0x46546C67, "not a glTF binary file"
    assert version == 2
    assert length == len(payload), "declared length does not match the file"
    offset, chunks = 12, {}
    while offset < length:
        chunk_length, chunk_type = struct.unpack("<II", payload[offset:offset + 8])
        offset += 8
        # Every chunk must be padded to a four byte boundary.
        assert chunk_length % 4 == 0
        chunks[chunk_type] = payload[offset:offset + chunk_length]
        offset += chunk_length
    return json.loads(chunks[0x4E4F534A].decode("utf-8")), chunks[0x004E4942]


def edge_use_counts(faces):
    """Return how many triangles use each undirected mesh edge."""
    counts = {}
    for first, second in ((0, 1), (1, 2), (2, 0)):
        for low, high in zip(faces[:, first], faces[:, second]):
            counts[(min(low, high), max(low, high))] = (
                counts.get((min(low, high), max(low, high)), 0) + 1
            )
    return np.asarray(sorted(counts.values()))


EXPECTED_METHODS = {
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
}


@pytest.fixture(scope="session")
def example_image(tmp_path_factory):
    path = next(Path("PyAbel_GUIs/Test Images").glob("*.dat"))
    # Preserve the entire field of view while keeping the all-method test fast.
    return prepare_image(np.loadtxt(path)[::6, ::6])


@pytest.fixture(scope="session")
def known_truth_projection():
    generator = ImageGenerator(201)
    generator.add_peak(40, 200, -0.5, 5)
    generator.add_peak(70, 150, 1.5, 7)
    truth = generator.generate_image()
    projection = abel.Transform(
        truth,
        direction="forward",
        method="direct",
        verbose=False,
    ).transform
    return truth, projection


@pytest.fixture
def transform_basis(tmp_path_factory):
    previous = abel.transform.get_basis_dir()
    basis_dir = tmp_path_factory.getbasetemp() / "gui-basis"
    abel.transform.set_basis_dir(str(basis_dir), make=True)
    try:
        yield
    finally:
        abel.transform.set_basis_dir(previous, make=False)


def test_transform_inventory_matches_current_pyabel():
    assert set(TRANSFORM_METHODS) == EXPECTED_METHODS


@pytest.mark.parametrize("method", TRANSFORM_METHODS)
def test_every_transform_on_supplied_example_data(example_image, transform_basis, method):
    result = inverse_transform(example_image, method)
    assert result.transform.shape == example_image.shape
    assert np.all(np.isfinite(result.transform))

    radial, speed = extract_speed_distribution(result, method)
    assert radial.shape == speed.shape
    assert radial.ndim == 1
    assert np.all(np.isfinite(radial))
    assert np.all(np.isfinite(speed))
    assert np.max(np.abs(speed)) <= 1 + 1e-12


@pytest.mark.parametrize("method", TRANSFORM_METHODS)
def test_every_transform_recovers_known_signal_region(
    known_truth_projection, transform_basis, method
):
    truth, projection = known_truth_projection
    reconstructed = inverse_transform(projection, method).transform
    signal = truth > truth.max() * 0.005
    correlation = np.corrcoef(truth[signal], reconstructed[signal])[0, 1]
    assert correlation > 0.99


def test_prepare_image_validates_and_crops_each_even_axis():
    assert prepare_image(np.zeros((100, 102))).shape == (99, 101)
    with pytest.raises(ValueError, match="two-dimensional"):
        prepare_image(np.zeros(10))
    bad = np.zeros((5, 5))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        prepare_image(bad)


@pytest.mark.parametrize(
    "peak",
    [
        (-1, 1, 0, 1),
        (50, 1, 0, 1),
        (10, -1, 0, 1),
        (10, 1, -1.01, 1),
        (10, 1, 2.01, 1),
        (10, 1, 0, 0),
    ],
)
def test_invalid_synthetic_peaks_are_rejected(peak):
    with pytest.raises(ValueError):
        validate_peak_parameters(*peak, image_size=100)


def test_even_synthetic_image_is_centered_between_pixels():
    generator = ImageGenerator(100)
    peak = generator.generate_peak(25, 4, 200, 1)
    assert generator.center == 49.5
    assert np.allclose(peak, peak[::-1, ::-1], atol=1e-12)


def test_peak_intensity_has_radius_independent_weighted_yield():
    generator = ImageGenerator(201)
    yields = []
    for radius in (25, 50, 75):
        peak = generator.generate_peak(radius, 4, 200, 0)
        yields.append(float(np.sum(peak * generator.r_pixels)))
    assert np.allclose(yields, 400_000, rtol=1e-12)


@pytest.mark.parametrize("beta2", [-1.0, 0.0, 2.0])
def test_anora_recovers_physical_beta2_boundaries(beta2):
    generator = ImageGenerator(101)
    image = generator.generate_peak(25, 4, 200, beta2)
    fitted, error = Anora(image).calculate_beta2(20, 30, show=False)["beta2"]
    assert fitted == pytest.approx(beta2, abs=2e-3)
    assert np.isfinite(error)


def test_anora_supports_single_pixel_radial_range():
    generator = ImageGenerator(101)
    image = generator.generate_peak(25, 4, 200, 0.5)
    result = Anora(image).calculate_beta2(25, 25, show=False)
    assert result["beta2"][0] == pytest.approx(0.5, abs=2e-3)


def test_ransac_is_reproducible_with_a_seed():
    generator = ImageGenerator(101, rng=np.random.default_rng(9))
    image = generator.generate_peak(25, 4, 200, 0.5)
    image = generator.add_noise(image, 0.2)
    anora = Anora(image)
    kwargs = {
        "method": "RANSAC",
        "RANSAC_trials": 40,
        "RANSAC_min_samples": 25,
        "random_seed": 123,
        "show": False,
    }
    first = anora.calculate_beta2(20, 30, **kwargs)
    second = anora.calculate_beta2(20, 30, **kwargs)
    assert first["beta2"] == pytest.approx(second["beta2"])
    assert np.array_equal(first["inliers"], second["inliers"])


def test_noise_zero_is_identity_and_seeded_noise_is_reproducible():
    image = np.full((21, 21), 10.0)
    first = ImageGenerator(21, rng=np.random.default_rng(123))
    second = ImageGenerator(21, rng=np.random.default_rng(123))
    assert np.array_equal(first.add_noise(image, 0), image)
    assert np.array_equal(first.add_noise(image, 0.5), second.add_noise(image, 0.5))
    with pytest.raises(ValueError, match="nonnegative"):
        first.add_noise(image, -1)


def test_elliptical_distortion_preserves_shape_center_and_identity():
    image = np.zeros((51, 61))
    image[25, 30] = 1
    identity = distort_image(image, horizontal_scale=1, vertical_scale=1)
    distorted = distort_image(image, horizontal_scale=1.05, vertical_scale=0.95)

    assert identity == pytest.approx(image, abs=1e-12)
    assert distorted.shape == image.shape
    assert np.unravel_index(np.argmax(distorted), distorted.shape) == (25, 30)
    with pytest.raises(ValueError, match="between 0.5 and 2.0"):
        distort_image(image, vertical_scale=0.49)


def test_radial_distortion_symmetry_mirrors_the_selected_axis():
    model = RadialDistortionModel(point_count=24)
    for axis_name in model.SYMMETRY_AXES:
        model.reset()
        model.set_axis(axis_name)
        model.set_scale(1, 1.12, symmetry=True)
        mirror = model.mirrored_index(1)
        assert mirror != 1
        assert model.scales[1] == pytest.approx(1.12)
        assert model.scales[mirror] == pytest.approx(1.12)


def test_radial_profile_supports_an_asymmetric_pear_distortion():
    generator = ImageGenerator(161)
    generator.add_peak(42, 200, 0, 2.5)
    projection = abel.Transform(
        generator.generate_image(),
        direction="forward",
        method="direct",
        verbose=False,
    ).transform
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pear_scales = 1 + 0.08 * np.cos(angles)
    distorted = distort_image_radial(projection, angles, pear_scales)

    center = projection.shape[0] // 2
    right_peak = np.argmax(distorted[center, center:])
    left_peak = np.argmax(distorted[center, :center + 1][::-1])
    assert right_peak > left_peak + 4

    corrected = circularize_image(
        distorted,
        method="argmax",
        radial_range=(28, 55),
        dt=0.1,
    )
    corrected_right = np.argmax(corrected[center, center:])
    corrected_left = np.argmax(corrected[center, :center + 1][::-1])
    assert abs(int(corrected_right) - int(corrected_left)) <= 1


def test_circularization_repairs_a_squashed_generated_projection():
    generator = ImageGenerator(161)
    generator.add_peak(42, 200, 0, 2.5)
    projection = abel.Transform(
        generator.generate_image(),
        direction="forward",
        method="direct",
        verbose=False,
    ).transform
    distorted = generator.distort_image(
        projection,
        horizontal_scale=1.0,
        vertical_scale=0.94,
    )
    corrected = circularize_image(
        distorted,
        method="argmax",
        radial_range=(28, 55),
        dt=0.1,
    )

    def angular_peak_spread(image):
        polar, radial, _ = abel.tools.polar.reproject_image_into_polar(
            image, dr=0.5, dt=0.1
        )
        radii = radial[:, 0]
        selected = (radii > 28) & (radii < 55)
        peaks = radii[selected][np.argmax(polar[selected], axis=0)]
        return float(np.std(peaks))

    distorted_spread = angular_peak_spread(distorted)
    corrected_spread = angular_peak_spread(corrected)
    assert distorted_spread > 0.5
    assert corrected_spread < distorted_spread * 0.5


def test_circularization_parameter_validation():
    image = np.ones((51, 51))
    with pytest.raises(ValueError, match="rmin < rmax"):
        circularize_image(image, radial_range=(20, 10))
    with pytest.raises(ValueError, match="cannot be negative"):
        circularize_image(image, tol=-0.01)


def test_log_file_uses_the_exact_requested_filename(tmp_path):
    dialog = SaveDialog.__new__(SaveDialog)
    dialog.parent_gui = SimpleNamespace(
        generator=SimpleNamespace(size=101, peaks=[[25.0, 200.0, 1.0, 4.0]]),
        noise_added=None,
    )
    filename = dialog.save_log_file(tmp_path, "example_parameters.txt")
    assert filename == "example_parameters.txt"
    assert (tmp_path / filename).is_file()
    assert not (tmp_path / "example_parameters_parameters.txt").exists()


def test_log_file_records_applied_distortion(tmp_path):
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    scales = 1 + 0.05 * np.cos(angles)
    dialog = SaveDialog.__new__(SaveDialog)
    dialog.parent_gui = SimpleNamespace(
        generator=SimpleNamespace(size=101, peaks=[]),
        noise_added=None,
        distorted_image=np.ones((101, 101)),
        distortion_profile=(angles, scales),
    )
    filename = dialog.save_log_file(tmp_path, "distorted_parameters.txt")
    contents = (tmp_path / filename).read_text()
    assert "Distortion Radial Profile (angle_deg, scale):" in contents
    assert "0.0, 1.050000" in contents
    assert "180.0, 0.950000" in contents


@pytest.mark.parametrize("subdivisions", range(DOT_MAX_SUBDIVISIONS + 1))
def test_dot_sphere_is_closed_and_outward_facing(subdivisions):
    vertices, faces = unit_icosphere(subdivisions)
    assert len(faces) == 20 * 4 ** subdivisions
    assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)
    assert np.all(edge_use_counts(faces) == 2)
    # A positive signed volume means every facet winds outward.
    triangles = vertices[faces]
    signed = np.sum(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    ) / 6
    assert 0 < signed <= 4 / 3 * np.pi


@pytest.mark.parametrize("beta2", [-1.0, 0.0, 1.0, 2.0])
def test_dot_cloud_reproduces_the_injected_anisotropy(beta2):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, beta2, 10)
    points = sample_distribution_points(
        generator.peaks, image_size=800, count=120_000, seed=7
    )
    radius = np.linalg.norm(points, axis=1)
    cosine = points[:, 1] / radius
    # For a distribution proportional to 1 + beta2 * P2, the mean of P2 over the
    # sphere is beta2 / 5, so the dots must carry the anisotropy back.
    recovered = 5 * np.mean(0.5 * (3 * cosine ** 2 - 1))
    assert recovered == pytest.approx(beta2, abs=0.03)


def test_dot_cloud_reproduces_the_radial_profile():
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    points = sample_distribution_points(
        generator.peaks, image_size=800, count=120_000, seed=7
    )
    radius = np.linalg.norm(points, axis=1)

    grid = np.linspace(140, 260, 4000)
    # The three-dimensional volume element carries a factor of R squared, which
    # pushes the mean radius slightly beyond the peak center.
    density = grid ** 2 * np.exp(-((grid - 200) ** 2) / 200)
    density /= _trapezoid(density, grid)
    expected_mean = _trapezoid(grid * density, grid)
    expected_std = np.sqrt(
        _trapezoid((grid - expected_mean) ** 2 * density, grid)
    )
    assert radius.mean() == pytest.approx(expected_mean, abs=0.2)
    assert radius.std() == pytest.approx(expected_std, abs=0.2)
    assert expected_mean > 200


def test_dot_count_per_peak_follows_the_three_dimensional_yield():
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    generator.add_peak(100, 100, 0.0, 10)
    points = sample_distribution_points(
        generator.peaks, image_size=800, count=120_000, seed=3
    )
    radius = np.linalg.norm(points, axis=1)
    outer = float(np.mean(radius > 150))
    # Peak intensity is defined as radius-independent yield, so twice the
    # intensity must draw twice the dots however far out the ring sits.
    assert outer == pytest.approx(2 / 3, abs=0.01)


def test_dot_positions_carry_no_lattice():
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    points = sample_distribution_points(
        generator.peaks, image_size=800, count=60_000, seed=11
    )
    # Points placed on any grid, jittered or not, leave the fractional part of
    # their coordinates uneven. Sampling the distribution directly does not.
    fractional = np.modf(np.abs(points))[0].ravel()
    counts, _ = np.histogram(fractional, bins=20, range=(0, 1))
    expected = counts.sum() / 20
    assert np.all(np.abs(counts - expected) < 0.1 * expected)


def test_dot_cloud_is_reproducible_with_a_seed():
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 1.0, 10)
    first = sample_distribution_points(
        generator.peaks, image_size=800, count=5_000, seed=5
    )
    again = sample_distribution_points(
        generator.peaks, image_size=800, count=5_000, seed=5
    )
    different = sample_distribution_points(
        generator.peaks, image_size=800, count=5_000, seed=6
    )
    assert np.array_equal(first, again)
    assert not np.array_equal(first, different)


def test_dot_cloud_mesh_places_one_closed_sphere_per_dot():
    points = np.array([[0.0, 0, 0], [10.0, 0, 0], [0.0, 10, 0]])
    vertices, faces = dot_cloud_mesh(points, diameter=2.0, subdivisions=0)
    assert len(faces) == 3 * 20
    assert count_mesh_surfaces(vertices, faces) == 3
    assert np.all(edge_use_counts(faces) == 2)
    # Each sphere is centered on its dot and has the requested diameter.
    for index, centre in enumerate(points):
        block = vertices[index * 12:(index + 1) * 12]
        assert np.allclose(block.mean(axis=0), centre, atol=1e-9)
        assert np.allclose(np.linalg.norm(block - centre, axis=1), 1.0)


def test_dot_export_writes_every_sphere(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 2.0, 10)
    target = tmp_path / "dots.stl"
    info = export_distribution_dots_stl(
        target, generator.peaks, image_size=800, count=2_000, seed=1
    )
    assert info["mode"] == "dots"
    assert info["dots"] == 2_000
    assert info["facets"] == 2_000 * 20
    assert target.stat().st_size == 84 + 50 * info["facets"]
    # A beta2 of 2 puts the dots along the symmetry axis, as for the shell.
    low, high = info["bounds"]
    assert high[1] - low[1] > high[0] - low[0]


@pytest.mark.parametrize(
    "settings",
    [
        {"count": DOT_MIN_COUNT - 1},
        {"count": DOT_MAX_COUNT + 1},
        {"count": 1000.5},
        {"diameter": 0.0},
        {"diameter": -1.0},
        {"subdivisions": -1},
        {"subdivisions": DOT_MAX_SUBDIVISIONS + 1},
    ],
)
def test_dot_export_rejects_invalid_settings(tmp_path, settings):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    with pytest.raises(ValueError):
        export_distribution_dots_stl(
            tmp_path / "dots.stl",
            generator.peaks,
            image_size=800,
            **{"count": 500, **settings},
        )


def test_binary_stl_round_trip_matches_the_dot_cloud(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 1.0, 10)
    target = tmp_path / "dots.stl"
    info = export_distribution_dots_stl(
        target, generator.peaks, image_size=800, count=800, seed=2
    )

    payload = target.read_bytes()
    # A binary file must not start with "solid" or readers parse it as ASCII.
    assert not payload.startswith(b"solid")
    declared = int(np.frombuffer(payload[80:84], dtype="<u4")[0])
    assert declared == info["facets"]
    records = np.frombuffer(payload[84:], dtype=STL_RECORD)
    assert len(records) == declared

    corners = records["vertices"].reshape(-1, 3)
    low, high = info["bounds"]
    assert np.allclose(corners.min(axis=0), low, atol=1e-3)
    assert np.allclose(corners.max(axis=0), high, atol=1e-3)
    # Every stored normal is a unit vector derived from its facet winding.
    assert np.allclose(np.linalg.norm(records["normal"], axis=1), 1.0, atol=1e-5)


def test_ascii_stl_declares_every_facet(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    target = tmp_path / "dots.stl"
    info = export_distribution_dots_stl(
        target,
        generator.peaks,
        image_size=800,
        count=300,
        seed=2,
        binary=False,
        name="model",
    )
    payload = target.read_text()
    assert payload.startswith("solid model")
    assert payload.rstrip().endswith("endsolid model")
    assert payload.count("facet normal") == info["facets"]
    assert payload.count("vertex ") == 3 * info["facets"]


def test_dot_density_tracks_the_distribution_it_samples():
    generator = ImageGenerator(800)
    generator.add_peak(200, 300, 1.5, 10)
    generator.add_peak(120, 150, -0.5, 8)
    peaks, amplitudes = _prepare_peaks(generator.peaks, 800)
    points = sample_distribution_points(
        generator.peaks, image_size=800, count=300_000, seed=13
    )

    # Count dots in thin shells and compare with the analytic field integrated
    # over each shell. Dots per unit volume must be the intensity itself.
    edges = np.linspace(80, 240, 33)
    counts, _ = np.histogram(np.linalg.norm(points, axis=1), bins=edges)

    def shell_integral(low, high, steps=64):
        radius = np.linspace(low, high, steps)
        # Probe along the magic angle, where cos(theta) is 1/sqrt(3) and P2 is
        # exactly zero. The angular factor is then 1 whatever beta2 is, which is
        # also its average over the sphere, so the field there is the radial
        # part alone and the shell integral is that times the shell volume.
        probe = np.stack(
            [radius * np.sqrt(2 / 3), radius / np.sqrt(3), np.zeros_like(radius)],
            axis=1,
        )
        field = evaluate_peak_field(probe, peaks, amplitudes) * radius ** 2
        return float(_trapezoid(field, radius))

    expected = np.array(
        [shell_integral(low, high) for low, high in zip(edges[:-1], edges[1:])]
    )

    busy = counts > 200
    total = counts[busy].sum()
    measured = counts[busy] / total
    predicted = expected[busy] / expected[busy].sum()
    # Counting dots is a Poisson process, so allow each shell five standard
    # deviations of its own count and nothing more.
    tolerance = 5 * np.sqrt(counts[busy]) / total
    assert np.all(np.abs(measured - predicted) < tolerance)


def test_glb_file_is_structurally_valid(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 1.0, 10)
    target = tmp_path / "dots.glb"
    info = export_distribution_dots_glb(
        target, generator.peaks, image_size=800, count=1_000,
        subdivisions=2, diameter=4.0, seed=4,
    )
    document, buffer = read_glb(target)

    assert document["asset"]["version"] == "2.0"
    # One root node groups the dots, so there is one node more than there are.
    assert len(document["nodes"]) == info["dots"] + 1
    assert document["buffers"][0]["byteLength"] == len(buffer)
    # Accessors must stay inside their views, and views inside the buffer.
    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(buffer)
    # glTF requires min and max on a position accessor.
    assert len(document["accessors"][0]["min"]) == 3
    assert len(document["accessors"][0]["max"]) == 3
    assert document["accessors"][2]["count"] == 3 * info["stored_triangles"]


def test_glb_stores_the_sphere_once_however_many_dots(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    for count in (500, 5_000):
        target = tmp_path / f"dots-{count}.glb"
        info = export_distribution_dots_glb(
            target, generator.peaks, image_size=800, count=count,
            subdivisions=2, seed=4,
        )
        document, _ = read_glb(target)
        # One mesh, referenced by every instance, is the point of instancing.
        assert len(document["meshes"]) == 1
        assert {node["mesh"] for node in document["nodes"][1:]} == {0}
        assert info["stored_triangles"] == 320
        assert info["facets"] == 320 * count


def test_glb_size_barely_depends_on_triangles_per_dot(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    sizes = {}
    for subdivisions in (0, 2):
        target = tmp_path / f"dots-{subdivisions}.glb"
        export_distribution_dots_glb(
            target, generator.peaks, image_size=800, count=5_000,
            subdivisions=subdivisions, seed=4,
        )
        sizes[subdivisions] = target.stat().st_size
    # Sixteen times the geometry must cost almost nothing, because only one
    # sphere is ever written. The same change multiplies an STL by sixteen.
    assert sizes[2] < 1.05 * sizes[0]
    assert sizes[2] < (84 + 50 * 5_000 * 320) / 50


def test_glb_places_every_dot_where_it_was_sampled(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 2.0, 10)
    target = tmp_path / "dots.glb"
    export_distribution_dots_glb(
        target, generator.peaks, image_size=800, count=3_000,
        subdivisions=0, diameter=4.0, seed=9,
    )
    document, buffer = read_glb(target)
    # glTF is written in metres, so convert the placements back to pixels.
    to_pixels = 1000 / 0.1
    placements = np.array(
        [node["translation"] for node in document["nodes"][1:]]
    ) * to_pixels
    expected = sample_distribution_points(
        generator.peaks, image_size=800, count=3_000, seed=9
    )
    # Placements are held to a hundredth of a pixel when written.
    assert np.allclose(placements, expected, atol=0.01)

    # The stored sphere carries the requested radius, not the dot positions.
    view = document["bufferViews"][0]
    vertices = np.frombuffer(
        buffer, dtype="<f4", count=document["accessors"][0]["count"] * 3,
        offset=view["byteOffset"],
    ).reshape(-1, 3) * to_pixels
    assert np.allclose(np.linalg.norm(vertices, axis=1), 2.0, atol=1e-3)


def test_glb_normals_point_outward():
    vertices, faces = unit_icosphere(1)
    normals = vertex_normals(vertices, faces)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # On a sphere centered at the origin the outward normal is the direction of
    # the vertex itself.
    assert np.all(np.sum(normals * vertices, axis=1) > 0.99)


@pytest.mark.parametrize(
    "settings",
    [
        {"count": DOT_MIN_COUNT - 1},
        {"count": DOT_MAX_COUNT + 1},
        {"diameter": 0.0},
        {"diameter": -2.0},
        {"subdivisions": DOT_MAX_SUBDIVISIONS + 1},
    ],
)
def test_glb_export_rejects_invalid_settings(tmp_path, settings):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    with pytest.raises(ValueError):
        export_distribution_dots_glb(
            tmp_path / "dots.glb",
            generator.peaks,
            image_size=800,
            **{"count": 500, **settings},
        )


def test_glb_groups_every_dot_under_one_object(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 1.0, 10)
    target = tmp_path / "dots.glb"
    info = export_distribution_dots_glb(
        target, generator.peaks, image_size=800, count=1_500,
        subdivisions=0, seed=8, name="newton_sphere",
    )
    document, _ = read_glb(target)

    # A viewer must open this as a single selectable object rather than as
    # thousands of separate top-level ones.
    assert document["scenes"][0]["nodes"] == [0]
    root = document["nodes"][0]
    assert root["name"] == "newton_sphere"
    assert "mesh" not in root
    # Every dot hangs off that root, and each keeps its own node identity.
    assert root["children"] == list(range(1, info["dots"] + 1))
    assert len(set(root["children"])) == info["dots"]
    for child in document["nodes"][1:]:
        assert child["mesh"] == 0
        assert "children" not in child
    # The instances take their name from the mesh, so they number cleanly and
    # do not collide with the group name.
    assert document["meshes"][0]["name"] == "dot"


def test_glb_grouping_costs_almost_nothing(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    target = tmp_path / "dots.glb"
    export_distribution_dots_glb(
        target, generator.peaks, image_size=800, count=5_000,
        subdivisions=2, seed=8,
    )
    # Grouping moves the index list from the scene into the root node rather
    # than adding one, so it must not meaningfully change the file size.
    assert target.stat().st_size < (1024 + 5_000 * 60 + 320 * 20)


def test_stl_is_written_in_millimetres(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    target = tmp_path / "dots.stl"
    info = export_distribution_dots_stl(
        target, generator.peaks, image_size=800, count=1_000, diameter=5.0,
        seed=1, millimetres_per_pixel=0.1,
    )
    assert info["units"] == "mm"
    records = np.frombuffer(target.read_bytes()[84:], dtype=STL_RECORD)
    corners = records["vertices"].reshape(-1, 3)
    span = corners.max(axis=0) - corners.min(axis=0)
    # A tenth of a millimetre per pixel puts a roughly 450 px cloud near 45 mm.
    assert np.all((span > 40) & (span < 50))
    assert np.allclose(span, np.array(info["bounds"][1]) - np.array(info["bounds"][0]),
                       atol=0.2)


def test_glb_is_written_in_metres_at_the_same_physical_size(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    target = tmp_path / "dots.glb"
    info = export_distribution_dots_glb(
        target, generator.peaks, image_size=800, count=1_000, diameter=5.0,
        seed=1, millimetres_per_pixel=0.1,
    )
    assert info["units"] == "m"
    document, _ = read_glb(target)
    placements = np.array([node["translation"] for node in document["nodes"][1:]])
    span_metres = placements.max(axis=0) - placements.min(axis=0)
    # glTF is specified in metres, so the numbers are a thousand times smaller
    # than the millimetre STL while describing the same object.
    assert np.all((span_metres > 0.03) & (span_metres < 0.05))
    reported = np.array(info["bounds"][1]) - np.array(info["bounds"][0])
    # The reported bounds include the dot radius at each end, which the raw
    # node translations do not.
    assert np.allclose(span_metres * 1000, reported, atol=6.0)


@pytest.mark.parametrize("writer", [export_distribution_dots_stl,
                                    export_distribution_dots_glb])
@pytest.mark.parametrize("scale", [0.0, -0.1, 101.0, float("nan")])
def test_export_rejects_an_impossible_scale(tmp_path, writer, scale):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    with pytest.raises(ValueError):
        writer(
            tmp_path / "dots.out", generator.peaks, image_size=800,
            count=500, millimetres_per_pixel=scale,
        )


def test_scale_changes_the_model_size_proportionally(tmp_path):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    spans = {}
    for scale in (0.1, 0.2):
        info = export_distribution_dots_stl(
            tmp_path / f"dots-{scale}.stl", generator.peaks, image_size=800,
            count=500, diameter=5.0, seed=1, millimetres_per_pixel=scale,
        )
        spans[scale] = np.array(info["bounds"][1]) - np.array(info["bounds"][0])
    assert np.allclose(spans[0.2], 2 * spans[0.1])
    assert MILLIMETRES_PER_PIXEL == 0.1


@pytest.mark.parametrize("millimetres_per_pixel", [0.01, 0.1, 1.0])
def test_glb_placements_are_not_quantised_by_rounding(tmp_path, millimetres_per_pixel):
    generator = ImageGenerator(800)
    generator.add_peak(200, 200, 0.0, 10)
    target = tmp_path / "dots.glb"
    count = 20_000
    export_distribution_dots_glb(
        target, generator.peaks, image_size=800, count=count, seed=1,
        millimetres_per_pixel=millimetres_per_pixel,
    )
    document, _ = read_glb(target)
    placements = np.array([node["translation"] for node in document["nodes"][1:]])

    # Writing in metres makes the numbers small, so a fixed number of decimals
    # would drop the dots onto a coarse grid and rebuild the lattice the cloud
    # exists to avoid. Precision has to track the units.
    to_pixels = 1000 / millimetres_per_pixel
    gaps = np.diff(np.unique(placements[:, 0] * to_pixels))
    assert gaps.min() == pytest.approx(0.01, abs=1e-6)
    # Distinct positions, not a handful of repeated grid values.
    assert len(np.unique(placements[:, 0])) > 0.5 * count


# The peaks recorded in the name of the reference model, TestPrint.glb.
REFERENCE_PEAKS = [
    (200, 200, 2.0, 10), (100, 200, -1.0, 10), (50, 100, 1.0, 5),
    (50, 50, 1.0, 5), (100, 100, -1.0, 5), (200, 200, 2.0, 10),
    (200, 200, 2.0, 5), (100, 100, -1.0, 5), (50, 50, 1.0, 5),
]


def test_default_sphere_matches_the_reference_model():
    # The reference cloud was measured at 0.1059 mm per pixel. The default
    # diameter must give its peaks that same scale back.
    scale = millimetres_per_pixel_for_diameter(
        REFERENCE_PEAKS, DEFAULT_SPHERE_DIAMETER_MM
    )
    assert distribution_reach(REFERENCE_PEAKS) == 230
    assert scale == pytest.approx(0.1059, rel=2e-3)


@pytest.mark.parametrize(
    "peaks",
    [REFERENCE_PEAKS, [(60, 100, 0.0, 4)], [(300, 100, 1.0, 20), (40, 50, 0.0, 5)]],
)
def test_sphere_diameter_holds_whatever_the_peaks(tmp_path, peaks):
    scale = millimetres_per_pixel_for_diameter(peaks, DEFAULT_SPHERE_DIAMETER_MM)
    target = tmp_path / "dots.glb"
    export_distribution_dots_glb(
        target, peaks, image_size=800, count=20_000, seed=3,
        millimetres_per_pixel=scale,
    )
    document, _ = read_glb(target)
    placements = np.array([node["translation"] for node in document["nodes"][1:]])
    radii_mm = np.linalg.norm(placements, axis=1) * 1000
    # Three widths past the outermost peak is the edge of the sphere, so only
    # the far tail of the outer shell may lie beyond it.
    assert np.mean(radii_mm > DEFAULT_SPHERE_DIAMETER_MM / 2) < 0.005
    assert radii_mm.max() > 0.9 * DEFAULT_SPHERE_DIAMETER_MM / 2


@pytest.mark.parametrize("diameter", [0.0, -10.0, float("nan"), float("inf")])
def test_sphere_diameter_must_be_positive(diameter):
    with pytest.raises(ValueError):
        millimetres_per_pixel_for_diameter(REFERENCE_PEAKS, diameter)


def test_sphere_diameter_needs_a_peak():
    with pytest.raises(ValueError):
        millimetres_per_pixel_for_diameter([], DEFAULT_SPHERE_DIAMETER_MM)
