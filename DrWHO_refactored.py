# Functions for Python Implementatio of DrWHO
"""Small HCIPy AO + DrWHO experiment using the existing SCExAO control matrix.

The saved control matrix assumes a 512 x 512 WFS image and 625 DM commands.
This module loads it with a memory map and saves image diagnostics only when requested.
"""

from dataclasses import dataclass

import hcipy as hp
import numpy as np

#%%
@dataclass
class AOState:
    iteration: int

    # WFS / controller quantities
    wfs_image: np.ndarray
    wfs_residual: np.ndarray
    dm_surface: np.ndarray

    # Post-AO common-path residual
    ao_residual: np.ndarray

    # Science images
    science_ao: np.ndarray

    # NCPA
    ncpa_opd: hp.Field
    science_ncpa: np.ndarray

    # Image quality
    strehl: float
    science_metric: float

@dataclass
class Simulation:
    wavelength: float

    # Grids/Pupil
    pupil_grid: hp.Grid
    wfs_grid: hp.Grid
    focal_grid: hp.Grid
    aperture: hp.Field

    # AO hardware/optics 
    dm: hp.DeformableMirror
    pwfs: hp.PyramidWavefrontSensorOptics

    # Science Arm 
    science_propagator: hp.FraunhoferPropagator
    ncpa_opd: hp.Field

    # Reference PSF used for Strehl ratio computation
    ideal_psf: np.ndarray

    # AO reconstruction matrix
    control_matrix: np.ndarray


def measure(sim, common_wf):
    """
    Measure the symchronized WFS and science - plane images
    """

    # PyWFS image
    wfs_wavefront= sim.pwfs.forward(common_wf.copy())

    camera = hp.NoiselessDetector(sim.wfs_grid)
    camera.integrate(wfs_wavefront, 1.0)

    wfs = np.asarray(camera.read_out(), dtype=float).copy()

    flux = wfs.sum()

    if flux <=0:
        raise ValueError("WFS flux is non-positive.")
    wfs /= flux

    # Uncomment for noisy detection
    # camera = hp.NoisyDetect()

    # AO Corrected wavefront (Before NCPA)
    science_ao = np.asarray(
        sim.science_propagator
        .forward(common_wf.copy())
        .intensity
        .shaped,
        dtype=float
    )
    # Add science only NCPA
    science_input = common_wf.copy()

    science_input.electric_field *= np.exp(2j * np.pi *sim.ncpa_opd/sim.wavelength)
    science_ncpa = np.asarray(
        sim.science_propagator.forward(science_input).intensity.shaped,
        dtype=float
    )

    return wfs, science_ao, science_ncpa

def compute_strehl(science_image, ideal_psf):
    science = np.asarray(science_image, dtype=float)
    ideal = np.asarray(ideal_psf, dtype=float)

    science /= science.sum()
    ideal /= ideal.sum()

    return float(science.max() / ideal.max())


def build_ao_state(
    sim,
    iteration,
    corrected_wf,
    wfs_reference,
):
    wfs_image, science_ao, science_ncpa = measure(
        sim,
        corrected_wf
    )

    wfs_residual = wfs_image - wfs_reference

    strehl = compute_strehl(
        science_ncpa,
        sim.ideal_psf
    )

    science_metric = strehl

    dm_surface = np.asarray(
        sim.dm.surface
    ).copy()

    ao_residual = compute_ao_residual(
        corrected_wf,
        sim.aperture
    )

    return AOState(
        iteration=iteration,
        wfs_image=wfs_image,
        wfs_residual=wfs_residual,
        dm_surface=dm_surface,
        ao_residual=ao_residual,
        science_ao=science_ao,
        ncpa_opd=sim.ncpa_opd,
        science_ncpa=science_ncpa,
        strehl=strehl,
        science_metric=science_metric,
    )

def ao_step(
    sim,
    iteration,
    incoming_wavefront,
    wfs_reference,
    gain,
):
    """
    Run one extreme-AO iteration and return the resulting AOState.
    """

    # Apply current DM correction
    corrected_wf = sim.dm(incoming_wavefront)

    # Build synchronized AO state
    state = build_ao_state(
        sim=sim,
        iteration=iteration,
        corrected_wf=corrected_wf,
        wfs_reference=wfs_reference,
    )

    # Convert WFS residual into DM correction
    delta_command = sim.control_matrix @ state.wfs_residual

    # Update DM for the NEXT iteration
    sim.dm.actuators -= gain * delta_command

    return state

def make_simulation(
        control_matrix, 
        wavelength=500e-9,  # Default wavelength in meters
        telescope_diameter = 8.2,  # Default telescope diameter in meters for the Subaru Telescope
        n_pupil = 512, 
        n_actuators = 25,  # Default number of DM actuators across the pupil
        ncpa_rms = 50e-9,  # Default NCPA RMS in meters
        ncpa_modes = 30, 
        seed=123
    ):
    """ 
    Builds static components of the SCExAO + DRWHO simulation.
    """
    # Pupil 
    pupil_grid = hp.make_pupil_grid(
        n_pupil,
        diameter=telescope_diameter
    )

    aperture = hp.evaluate_supersampled(
        hp.make_subaru_aperture(), 
        pupil_grid,
        6
    )

    # DM
    actuator_spacing = telescope_diameter / (n_actuators)

    influence_functions = hp.make_gaussian_influence_functions(
        pupil_grid,
        25, 
        telescope_diameter/25, 
        0.15
    )

    dm = hp.DeformableMirror(influence_functions)

    dm.actuators[:] = 0.0

    # PyWFS 
    wfs_grid = hp.make_pupil_grid(
        n_pupil,
        2 * telescope_diameter
    )

    pwfs = hp.PyramidWavefrontSensorOptics(
        pupil_grid,
        wfs_grid,
        separation=telescope_diameter,
        pupil_diameter=telescope_diameter,
        wavelength_0=wavelength,
        q=3
    )

    # Science Focal Plane 
    focal_grid = hp.make_focal_grid(
        q=8,
        num_airy=16,
        spatial_resolution=wavelength / telescope_diameter,
        reference_wavelength=wavelength
    )

    science_propagator = hp.FraunhoferPropagator(
        pupil_grid,
        focal_grid
    )

    # Build the Ideal PSF 
    ideal_wavefront = hp.Wavefront(
        aperture.copy(),
        wavelength
    )

    ideal_psf = np.asarray(
        science_propagator
        .forward(ideal_wavefront)
        .intensity
        .shaped,
        dtype=float
    )

    # Build in the NCPA (static for now)
    basis = hp.make_zernike_basis(
        ncpa_modes,
        telescope_diameter,
        pupil_grid
    )

    rng = np.random.default_rng(seed)

    coefficients = rng.normal(
        size=ncpa_modes
    )

    # Remove piston / tip / tilt
    coefficients[:3] = 0.0

    ncpa_opd = basis.linear_combination(
        coefficients
    )

    mask = np.asarray(aperture) > 0

    # Remove piston
    ncpa_opd[mask] -= np.mean(
        ncpa_opd[mask]
    )

    # Normalize to requested RMS
    current_rms = np.sqrt(
        np.mean(
            np.asarray(ncpa_opd[mask])**2
        )
    )

    ncpa_opd[mask] *= (
        ncpa_rms / current_rms
    )

    ncpa_opd[~mask] = 0.0

    return Simulation(
        wavelength = wavelength,
        pupil_grid=pupil_grid,
        wfs_grid=wfs_grid,
        focal_grid=focal_grid,
        aperture=aperture,
        dm=dm, 
        pwfs=pwfs,
        science_propagator=science_propagator,
        ncpa_opd=ncpa_opd,
        ideal_psf=ideal_psf,
        control_matrix = control_matrix
    )

def compute_ao_residual(corrected_wf, aperture):
    """ 
    Computes the post-extreme-AO common-path residual wavefront error.

    Parameters
    ----------
    corrected_wf : hp.Wavefront
        The wavefront after AO correction.
    aperture : array_like
        The telescope aperture mask.

    Returns
    -------
    residual_opd : array_like
        The residual optical path difference after AO correction.
    """
    phase = np.asarray(
        corrected_wf.phase,
        dtype=float
    )

    residual_opd = (
        phase
        * corrected_wf.wavelength
        / (2 * np.pi)
    )

    residual_opd = residual_opd.copy()

    mask = np.asarray(aperture) > 0

    # Remove piston
    residual_opd[mask] -= np.mean(
        residual_opd[mask]
    )

    # Don't interpret values outside pupil
    residual_opd[~mask] = np.nan

    return residual_opd

def make_residual_wavefront(
    aperture,
    wavelength,
    residual_opd,
):
    """
    Create a wavefront representing the residual wavefront
    after upstream AO correction.

    Parameters
    ----------
    aperture : hp.Field
        Telescope pupil amplitude.

    wavelength : float
        Wavelength in meters.

    residual_opd : hp.Field or np.ndarray
        Residual optical path difference in meters.

    Returns
    -------
    hp.Wavefront
        Residual wavefront.
    """

    residual_wf = hp.Wavefront(
        aperture.copy(),
        wavelength
    )

    phase = (
        2 * np.pi
        * residual_opd
        / wavelength
    )

    residual_wf.electric_field *= np.exp(
        1j * phase
    )

    return residual_wf

def make_random_residual(
    pupil_grid,
    aperture,
    rms=50e-9,
    seed=None,
):
    rng = np.random.default_rng(seed)

    residual = hp.Field(
        rng.normal(size=pupil_grid.size),
        pupil_grid
    )

    mask = np.asarray(aperture) > 0

    residual[mask] -= residual[mask].mean()

    current_rms = np.sqrt(
        np.mean(np.asarray(residual[mask])**2)
    )

    residual[mask] *= rms / current_rms
    residual[~mask] = 0

    return residual