"""Small HCIPy AO + DrWHO experiment using the existing SCExAO control matrix.

The saved control matrix assumes a 512 x 512 WFS image and 625 DM commands.
This module loads it with a memory map and saves image diagnostics only when requested.
"""

from dataclasses import dataclass

import hcipy as hp
import numpy as np


@dataclass
class Simulation:
    wavelength: float
    aperture: hp.Field
    clean_wavefront: hp.Wavefront
    atmosphere: object
    dm: hp.DeformableMirror
    pwfs: hp.PyramidWavefrontSensorOptics
    wfs_grid: hp.Grid
    science_propagator: hp.FraunhoferPropagator
    control_matrix: np.ndarray
    ncpa_opd: hp.Field


def make_simulation(control_matrix_path="control_matrix_test.npy", ncpa_rms=50e-9, seed=123):
    """Build the optics once; use the geometry of the saved control matrix."""
    wavelength = 500e-9
    diameter = 8.2
    pupil_diameter = 1.2 * diameter
    pupil_grid = hp.make_pupil_grid(512, pupil_diameter)
    wfs_grid = hp.make_pupil_grid(512, 2 * pupil_diameter)
    aperture = hp.evaluate_supersampled(hp.make_subaru_aperture(), pupil_grid, 6)
    clean_wavefront = hp.Wavefront(aperture.copy(), wavelength)

    cn2 = hp.Cn_squared_from_fried_parameter(0.20, wavelength=wavelength)
    layers = hp.make_mauna_kea_atmospheric_layers(
        pupil_grid, cn_squared=cn2, outer_scale=20.0
    )
    atmosphere = hp.MultiLayerAtmosphere(layers, scintillation=False)

    influence_functions = hp.make_gaussian_influence_functions(
        pupil_grid, 25, diameter / 25, 0.15
    )
    dm = hp.DeformableMirror(influence_functions)
    dm.actuators[:] = 0

    pwfs = hp.PyramidWavefrontSensorOptics(
        pupil_grid, wfs_grid, separation=pupil_diameter,
        pupil_diameter=diameter, wavelength_0=wavelength, q=3
    )
    focal_grid = hp.make_focal_grid(
        q=8, num_airy=16, spatial_resolution=wavelength / diameter,
        reference_wavelength=wavelength
    )
    science_propagator = hp.FraunhoferPropagator(pupil_grid, focal_grid)

    control_matrix = np.load(control_matrix_path, mmap_mode="r")
    expected_shape = (dm.num_actuators, wfs_grid.size)
    if control_matrix.shape != expected_shape:
        raise ValueError(
            f"Control matrix shape {control_matrix.shape} does not match {expected_shape}. "
            "Keep the original geometry or recalibrate the WFS."
        )

    # A fixed, science-only NCPA makes the first DrWHO experiment easier to read.
    basis = hp.make_zernike_basis(30, diameter, pupil_grid)
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=30)
    coefficients[:3] = 0
    ncpa_opd = basis.linear_combination(coefficients)
    mask = np.asarray(aperture) > 0
    ncpa_opd[mask] -= np.mean(ncpa_opd[mask])
    current_rms = np.sqrt(np.mean(np.asarray(ncpa_opd[mask]) ** 2))
    ncpa_opd[mask] *= ncpa_rms / current_rms
    ncpa_opd[~mask] = 0

    return Simulation(
        wavelength, aperture, clean_wavefront, atmosphere, dm, pwfs,
        wfs_grid, science_propagator, control_matrix, ncpa_opd
    )


def measure(sim, common_wavefront, include_common_psf=False):
    """Return a synchronized WFS frame and science images."""
    wfs_wavefront = sim.pwfs.forward(common_wavefront.copy())
    camera = hp.NoiselessDetector(sim.wfs_grid)
    camera.integrate(wfs_wavefront, 1.0)
    wfs = np.asarray(camera.read_out(), dtype=float).copy()
    flux = wfs.sum()
    if flux <= 0:
        raise ValueError("WFS frame has no flux")
    wfs /= flux

    science_input = common_wavefront.copy()
    science_input.electric_field *= np.exp(
        2j * np.pi * sim.ncpa_opd / sim.wavelength
    )
    common_psf = None
    if include_common_psf:
        common_psf = np.asarray(
            sim.science_propagator.forward(common_wavefront.copy()).intensity.shaped
        )
    psf_field = sim.science_propagator.forward(science_input).intensity
    psf = np.asarray(psf_field.shaped)
    # For this non-coronagraphic PSF, peak / total flux is a simple sharpness score.
    score = float(psf.max() / psf.sum())

    return wfs, score, psf, common_psf


def select_drwho_reference(wfs_frames, science_scores, best_fraction=0.1):
    """Average WFS frames paired with the best science frames in one window."""
    if not 0 < best_fraction <= 1:
        raise ValueError("best_fraction must be in (0, 1]")
    scores = np.asarray(science_scores)
    if len(scores) == 0 or len(wfs_frames) != len(scores):
        raise ValueError("Need equally many paired WFS frames and science scores")
    count = max(1, int(np.ceil(best_fraction * len(scores))))
    selected = np.argpartition(scores, -count)[-count:]
    reference = np.mean(np.asarray(wfs_frames)[selected], axis=0)
    return reference / reference.sum()


def run_drwho(
    sim, *, num_iterations=100, loop_rate_hz=2000, gain=0.4,
    leakage=0.05, window_size=50, best_fraction=0.1,
    reference_blend=1.0, update_reference=True, record_frames=False,
    save_every=5
):
    """Close the AO loop and update its WFS reference every DrWHO window.

    Set update_reference=False for a baseline using the same atmosphere and NCPA.
    The reference is updated *between* windows; WFS and science measurements
    are acquired from the same corrected pupil before each DM command update.
    """
    if num_iterations < 1 or window_size < 1 or loop_rate_hz <= 0 or save_every < 1:
        raise ValueError("Iterations, window size, loop rate, and save_every must be positive")
    if not 0 <= reference_blend <= 1:
        raise ValueError("reference_blend must be in [0, 1]")

    sim.dm.actuators[:] = 0
    sim.atmosphere.reset()
    reference, _, _, _ = measure(sim, sim.dm(sim.clean_wavefront.copy()))
    initial_reference = reference.copy()
    history = {"score": [], "reference_shift": [], "update_at": []}
    if record_frames:
        history["saved_at"] = []
        history["dm_surface_frames"] = []
        history["residual_opd_frames"] = []
        history["wfs_frames"] = []
        history["reference_frames"] = []
        history["common_psf_frames"] = []
        history["science_frames"] = []
    window_wfs, window_scores = [], []

    for index in range(num_iterations):
        sim.atmosphere.evolve_until(index / loop_rate_hz)
        incoming = sim.atmosphere.forward(sim.clean_wavefront.copy())
        corrected = sim.dm(incoming)
        save_this_frame = record_frames and (index + 1) % save_every == 0
        wfs, score, psf, common_psf = measure(
            sim, corrected, include_common_psf=save_this_frame
        )
        window_wfs.append(wfs)
        window_scores.append(score)
        history["score"].append(score)
        if save_this_frame:
            mask = np.asarray(sim.aperture) > 0
            residual_opd = np.asarray(
                sim.atmosphere.phase_for(sim.wavelength) * sim.wavelength / (2 * np.pi)
                + sim.dm.opd
            ).copy()
            residual_opd[mask] -= residual_opd[mask].mean()
            residual_opd[~mask] = np.nan
            dm_surface = np.asarray(sim.dm.surface).copy()
            dm_surface[~mask] = np.nan
            history["saved_at"].append(index + 1)
            history["dm_surface_frames"].append(
                (dm_surface.reshape(sim.aperture.grid.shape) * 1e9).astype(np.float32)
            )
            history["residual_opd_frames"].append(
                (residual_opd.reshape(sim.aperture.grid.shape) * 1e9).astype(np.float32)
            )
            history["wfs_frames"].append(
                wfs.reshape(sim.wfs_grid.shape).astype(np.float32)
            )
            history["reference_frames"].append(
                reference.reshape(sim.wfs_grid.shape).astype(np.float32)
            )
            history["common_psf_frames"].append(common_psf.astype(np.float32))
            history["science_frames"].append(psf.astype(np.float32))

        residual = wfs - reference
        delta = sim.control_matrix @ residual
        sim.dm.actuators = (1 - leakage) * sim.dm.actuators - gain * delta

        if len(window_scores) == window_size:
            if update_reference:
                selected = select_drwho_reference(
                    window_wfs, window_scores, best_fraction
                )
                reference = (1 - reference_blend) * reference + reference_blend * selected
                reference /= reference.sum()
            history["reference_shift"].append(float(np.linalg.norm(reference - initial_reference)))
            history["update_at"].append(index + 1)
            window_wfs.clear()
            window_scores.clear()

    return history, reference


def animate_frames(history, interval_ms=100):
    """Animate six synchronized diagnostics for every saved iteration."""
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    wfs_frames = history.get("wfs_frames")
    if not wfs_frames:
        raise ValueError("Run with record_frames=True to animate the frames")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("dm_surface_frames", "DM surface [nm]", "RdBu_r"),
        ("residual_opd_frames", "Residual OPD [nm]", "RdBu_r"),
        ("common_psf_frames", "Common path PSF", "inferno"),
        ("science_frames", "Science PSF + NCPA", "inferno"),
        ("wfs_frames", "Pyramid WFS frame", "viridis"),
        ("reference_frames", "WFS reference used", "viridis"),
    ]
    images = []
    for ax, (key, title, cmap) in zip(axes.flat, panels):
        limits = {}
        if key in ("dm_surface_frames", "residual_opd_frames"):
            span = max(float(np.nanmax(np.abs(frame))) for frame in history[key])
            limits = {"vmin": -max(span, 1e-6), "vmax": max(span, 1e-6)}
        elif key in ("common_psf_frames", "science_frames"):
            limits = {"vmin": -5, "vmax": 0}
        else:
            limits = {"vmin": 0, "vmax": 1}
        image = ax.imshow(history[key][0], origin="lower", cmap=cmap, **limits)
        ax.set_title(title)
        ax.set_axis_off()
        images.append(image)

    def display_psf(psf):
        return np.log10(np.maximum(psf / max(float(psf.max()), 1e-30), 1e-5))

    def update(index):
        for image, (key, _, _) in zip(images, panels):
            frame = history[key][index]
            if key in ("common_psf_frames", "science_frames"):
                frame = display_psf(frame)
            elif key in ("wfs_frames", "reference_frames"):
                frame = frame / max(float(frame.max()), 1e-30)
            image.set_data(frame)
        iteration = history["saved_at"][index]
        fig.suptitle(
            f"Iteration {iteration} · science score {history['score'][iteration - 1]:.5g}"
        )
        return images

    animation = FuncAnimation(
        fig, update, frames=len(wfs_frames), interval=interval_ms, blit=False
    )
    fig.tight_layout()
    plt.close(fig)
    return animation
