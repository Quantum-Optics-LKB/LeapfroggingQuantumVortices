# %% Imports et blabla
from scipy.fft import fft2, ifft2
import contrast as contrast
from PIL import Image
from scipy import ndimage
from skimage import restoration
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import multiprocessing
from cycler import cycler
from matplotlib.animation import FuncAnimation
import polarTransform as pT
from matplotlib.colors import LinearSegmentedColormap

# import cupy as cp
from scipy.optimize import curve_fit
import tqdm
import time
import cv2
import os
import regex as re
import velocity as velocity
import pickle
import faulthandler
import scipy.optimize as opt
from matplotlib.colors import LogNorm
import cmasher as cmr
# import polarTransform

# from findpeaks import findpeaks

faulthandler.enable()

# matplotlib.use("Qt5Agg")

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
# try to load previous fftw wisdom
try:
    with open("fft.wisdom", "rb") as file:
        wisdom = pickle.load(file)
        pyfftw.import_wisdom(wisdom)
except FileNotFoundError:
    print("No FFT wisdom found, starting over ...")
# for dark theme
# plt.style.use("dark_background")
# plt.rcParams["figure.facecolor"] = "#00000080"
# plt.rcParams["axes.facecolor"] = "#00000080"
# plt.rcParams["savefig.facecolor"] = "#00000080"
# plt.rcParams['savefig.transparent'] = True
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Liberation Sans']
# for plots
tab_colors = [
    "tab:blue",
    "tab:orange",
    "forestgreen",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "teal",
]
fills = [
    "lightsteelblue",
    "navajowhite",
    "darkseagreen",
    "lightcoral",
    "violet",
    "indianred",
    "lavenderblush",
    "lightgray",
    "darkkhaki",
    "darkturquoise",
]
edges = tab_colors
custom_cycler = (
    (cycler(color=tab_colors))
    + (cycler(markeredgecolor=edges))
    + (cycler(markerfacecolor=fills))
)
plt.rc("axes", prop_cycle=custom_cycler)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.size": 12
# })
# sshfs taladjidi@patriot.lkb.upmc.fr:/partages/EQ15B/LEON-15B /home/tangui/LEON
# path_leon = "/run/user/1000/gvfs/sftp:host=patriot.lkb.upmc.fr,user=taladjidi/partages/EQ15B/LEON-15B"
# path_leon = "/run/user/1000/gvfs/sftp:host=88.160.142.14,port=16384,user=aladjidi/home/aladjidi/Disk0/LEON"

# %% Paths and parameters

path_cache = "C:/Users/.../data_cache"
k0 = 2 * np.pi / 780e-9
L = 20e-2
d_slm = 8e-6
d_real = 3.45e-6 * 2
d_def = 1.1e-6
Nx, Ny = 2464, 2056

Nx_def, Ny_def = 2048, 2048

# %% Functions


def load_cached_array(
    filepath: str, filename: str, cache_folder_path: str, max_cache_time: int = 36000
) -> np.ndarray:
    """
    Load a numpy array with a caching mechanism to avoid repeated network loading.

    Args:
        filepath (str): Path to the original .npy file (network location).
        cache_path (str): Path to the local cache folder.
        max_cache_time (int): Maximum cache duration in seconds (default: 10 hours).

    Returns:
        np.ndarray: The loaded numpy array.
    """
    cache_path = f"{cache_folder_path}/{filename}"
    if os.path.exists(cache_path):
        cache_age = time.time() - os.path.getmtime(cache_path)
        original_age = (
            time.time() - os.path.getmtime(filepath)
            if os.path.exists(filepath)
            else float("inf")
        )

        if cache_age >= max_cache_time:
            print("Cache expired, deleting...")
            os.remove(cache_path)

        elif os.path.getmtime(cache_path) >= os.path.getmtime(filepath):
            print("Loading from cache...")
            return np.load(cache_path, allow_pickle=True)

    print("Loading from source and updating cache...")
    data = np.load(f"{filepath}/{filename}", allow_pickle=True)
    np.save(cache_path, data)
    return data


def mask_vortices(phase, vortices, mask_radius=4):
    """
    Masquer les vortex en les remplaçant par la valeur médiane locale.
    """
    masked_phase = phase.copy()
    for vortex in vortices:
        x, y, _ = vortex  # x, y, charge du vortex (non utilisé ici)

        # Définir une zone autour du vortex pour le masquer
        x_start, x_end = max(0, x - mask_radius), min(phase.shape[0], x + mask_radius)
        y_start, y_end = max(0, y - mask_radius), min(phase.shape[1], y + mask_radius)

        # Remplacer les valeurs par la médiane locale
        local_patch = phase[x_start:x_end, y_start:y_end]
        median_value = np.median(local_patch)
        masked_phase[x_start:x_end, y_start:y_end] = median_value

    return masked_phase


def reinstate_vortices(phase, vortices):
    """
    Réinsérer les vortex détectés avec leur charge initiale.
    """
    restored_phase = phase.copy()
    for vortex in vortices:
        x, y, charge = vortex
        # Réinsérer la singularité de phase
        restored_phase[x, y] += charge * np.pi  # Ajuste selon ton besoin
    return restored_phase


def Vortex2D(mn, y_val, x_val, xi) -> np.ndarray:
    y, x = mn
    # y, x = np.meshgrid(m, n)
    # x = x - n / 2.0 + x_val
    # y = y - m / 2.0 + y_val
    r = np.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
    theta = np.arctan2(y, x)
    Psi = r / np.sqrt(r**2 + (2 * xi) ** 2)
    Amp = np.abs(Psi)
    return Amp.ravel()


def Vortex(x, y, x_val, y_val, ell, xi) -> np.ndarray:
    """Generate a vortex profile.
    Args:
        m (int): Height of the pattern
        n (int): Width of the pattern
        x_val (int): Position of the soliton
        y_val (int): Position of the soliton
        ell (int): Charge of the vortex
        xi (float): xi parameter
        pitch (int, optional): Pixel pitch of the grating in px
        theta (int, optional): Angle of the grating in degrees. Defaults to 90.
    """
    x = x - x_val
    y = y - y_val
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    Psi = r / np.sqrt(r**2 + (xi / 0.83) ** 2) * np.exp(1j * ell * theta)
    Amp = np.abs(Psi)
    Phase = np.angle(Psi)

    return Amp, Phase


def JRS_padé(x, y):
    """
    Compute the density of the rarefaction pulse at U = 0.5.

    Parameters:
    x (float or np.ndarray): x-coordinate(s).
    y (float or np.ndarray): y-coordinate(s).

    Returns:
    float or np.ndarray: Density rho(x, y).
    """
    # Numerator coefficients for u and v
    a00 = -0.825937
    a10 = -0.114393
    a01 = -0.0271467

    b00 = -0.737901
    b10 = -0.108587
    b01 = -0.0542934

    # Denominator coefficients (common to both u and v)
    c10 = 0.37355
    c01 = 0.14235
    c20 = 0.03495
    c11 = 0.03495
    c02 = 0.008737

    # Compute u(x, y)
    u_num = a00 + a10 * x**2 + a01 * y**2
    u_den = 1 + c10 * x**2 + c01 * y**2 + c20 * x**4 + c11 * x**2 * y**2 + c02 * y**4
    u = 1 + u_num / u_den

    # Compute v(x, y)
    v_num = x * (b00 + b10 * x**2 + b01 * y**2)
    v = v_num / u_den  # Note: same denominator as u

    # Compute density rho(x, y)
    amp = u**2 + v**2
    psi = u + 1j * v
    amp = np.abs(psi)
    phase = np.angle(psi)
    return amp, phase


def Soliton1D(x, Mach, xi):
    Psi = (
        np.sqrt(1 - Mach**2) * np.tanh((x / (np.sqrt(2) * xi)) * np.sqrt(1 - Mach**2))
        + 1j * Mach
    )
    Amp = np.abs(Psi)
    Phase = np.angle(Psi) + np.pi
    return Amp, Phase


def power_law(x, a, b):
    """Calcul la loi de puissance avec les constantes a et b"""
    return a * np.power(x, b)


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def compute_phase_time(scan: str, plot: bool = False):
    """Compute the phase.

    Computes the phase from the interferograms for a time scan
    dataset.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    field_npy = np.load(f"{scan}/field.npy")
    N_times = field_npy.shape[0]
    N_avg = field_npy.shape[1]
    laser_powers = np.linspace(0, 1, N_times)
    field = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    field_ref = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    field_vortex = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    # ref_arm = np.array(Image.open(
    #     f"{path_real}/ref.TIF"), dtype=np.uint16).astype(np.float32)
    # ref_arm = ndimage.zoom(ref_arm, .5)
    taus = np.zeros(N_times)
    taus_err = np.zeros(N_times)
    # prepare plans
    a = pyfftw.empty_aligned((Ny, Nx), dtype=np.float32)
    c = pyfftw.empty_aligned((Ny // 2, Nx // 2), dtype=np.complex64)
    plan_fft = pyfftw.builders.rfft2(a)
    plan_ifft = pyfftw.builders.ifft2(c)
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plans = (plan_fft, plan_ifft)
    # del a, b
    pbar = tqdm.tqdm(total=N_times * N_avg, desc="Computing phase", position=1)
    for i in range(N_times):
        tau = np.zeros(N_avg)
        for j in range(N_avg):
            im_ref = np.array(
                Image.open(f"{scan}/dn_ref_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            im = np.array(
                Image.open(f"{scan}/dn_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            im_vortex = np.array(
                Image.open(f"{scan}/dn_vortex_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            field_ref[i, j, :, :] = contrast.im_osc_fast_t(
                im_ref,
                plans=plans,
                center=(im_ref.shape[-2] // 2, im_ref.shape[-1] // 4),
                radius=im_ref.shape[-2] // 8,
            )
            field[i, j, :, :] = contrast.im_osc_fast_t(
                im,
                center=(im_ref.shape[-2] // 2, im_ref.shape[-1] // 4),
                radius=im_ref.shape[-2] // 8,
            )
            field_vortex[i, j, :, :] = contrast.im_osc_fast_t(
                im_vortex,
                plans=plans,
                center=(im_ref.shape[-2] // 2, im_ref.shape[-1] // 4),
                radius=im_ref.shape[-2] // 8,
            )
            phi_ref = contrast.angle_fast(field_ref[i, j, :, :])
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            threshold = 2e-2
            mask = rho_ref < threshold * np.max(rho_ref)
            phi_ref_masked = np.ma.array(phi_ref, mask=mask)
            phi_ref_unwrapped = restoration.unwrap_phase(
                phi_ref_masked, wrap_around=(True, True)
            )
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(np.abs(field_ref[i, j, :, :]) ** 2)
            # ax[1].imshow(phi_ref_unwrapped)
            # plt.show()
            tau[j] = np.abs(np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped))
            pbar.update(1)
        taus[i] = np.mean(tau)
        taus_err[i] = np.std(tau)
    pbar.close()
    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{scan}/field_ref.npy", field_ref)
    np.save(f"{scan}/field.npy", field)
    np.save(f"{scan}/field_vortex.npy", field_vortex)
    np.save(f"{scan}/taus_reproc.npy", taus)
    np.save(f"{scan}/taus_err_reproc.npy", taus_err)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(f"Saved {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[0, -1, :, :]) ** 2)
    ax[1].imshow(np.angle(field[0, -1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{scan}/field.svg", dpi=300)
    taus0 = np.load(f"{scan}/taus.npy")
    taus0_err = np.load(f"{scan}/taus_err.npy")
    fig1, ax1 = plt.subplots()
    ax1.errorbar(
        laser_powers,
        taus0,
        yerr=taus0_err,
        marker="o",
        label="During acquisition",
    )
    ax1.errorbar(laser_powers, taus, yerr=taus_err, marker="o", label="Reprocessed")
    ax1.legend()
    ax1.set_xlabel("Power in a.u.")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def compute_phase_full_time(scan: str, plot: bool = False):
    """Compute the phase.

    Computes the phase from the interferograms for a time scan
    dataset.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    # powers = np.load(f"{path_real}/{scan}/powers.npy")
    # N_times = powers.size
    input_power = np.load(f"{path_real}/{scan}/input_power.npy")
    N_times = input_power.size
    N_avg = np.load(f"{path_real}/{scan}/N_avg.npy")
    laser_powers = np.load(f"{path_real}/{scan}/laser_settings.npy")
    field = np.zeros((N_times, N_avg, Ny, Nx), dtype=np.complex64)
    field_ref = np.zeros((N_times, N_avg, Ny, Nx), dtype=np.complex64)
    field_vortex = np.zeros((N_times, N_avg, Ny, Nx), dtype=np.complex64)
    # ref_arm = np.array(Image.open(
    #     f"{path_real}/ref.TIF"), dtype=np.uint16).astype(np.float32)
    # ref_arm = ndimage.zoom(ref_arm, .5)
    taus = np.zeros(N_times)
    taus_err = np.zeros(N_times)
    # prepare plans
    a = pyfftw.empty_aligned((Ny, Nx), dtype=np.float32)
    c = pyfftw.empty_aligned((Ny, Nx), dtype=np.complex64)
    plan_fft = pyfftw.builders.rfft2(a)
    plan_ifft = pyfftw.builders.ifft2(c)
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plans = (plan_fft, plan_ifft)
    # del a, b
    pbar = tqdm.tqdm(total=N_times * N_avg, desc="Computing phase", position=1)
    for i in range(N_times):
        tau = np.zeros(N_avg)
        for j in range(N_avg):
            im_ref = np.array(
                Image.open(f"{path_dn}/{scan}/dn_ref_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            im = np.array(
                Image.open(f"{path_dn}/{scan}/dn_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            im_vortex = np.array(
                Image.open(f"{path_dn}/{scan}/dn_vortex_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            field_ref[i, j, :, :] = contrast.im_osc(im_ref, plot=True)
            field[i, j, :, :] = contrast.im_osc(im)
            # plt.imshow(np.abs(field[i, j, :, :]) ** 2)
            # plt.show()
            field_vortex[i, j, :, :] = contrast.im_osc(im_vortex)
            phi_ref = np.angle(field_ref[i, j, :, :])
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            threshold = 2e-2
            mask = rho_ref < threshold * np.nanmax(rho_ref)
            phi_ref_masked = np.ma.array(phi_ref, mask=mask)
            phi_ref_unwrapped = restoration.unwrap_phase(
                phi_ref_masked, wrap_around=(True, True)
            )
            tau[j] = np.abs(np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped))
            pbar.update(1)
        taus[i] = np.mean(tau)
        taus_err[i] = np.std(tau)
    pbar.close()
    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{path_dn}/{scan}/field_ref_full.npy", field_ref)
    np.save(f"{path_dn}/{scan}/field_full.npy", field)
    np.save(f"{path_dn}/{scan}/field_vortex_full.npy", field_vortex)
    np.save(f"{path_dn}/{scan}/taus_reproc.npy", taus)
    np.save(f"{path_dn}/{scan}/taus_err_reproc.npy", taus_err)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(f"Saved {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[0, -1, :, :]) ** 2)
    ax[1].imshow(np.angle(field[0, -1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{path_dn}/{scan}/field.svg", dpi=300)
    taus0 = np.load(f"{path_dn}/{scan}/taus.npy")
    taus0_err = np.load(f"{path_dn}/{scan}/taus_err.npy")
    fig1, ax1 = plt.subplots()
    ax1.errorbar(
        laser_powers,
        taus0,
        yerr=taus0_err,
        marker="o",
        label="During acquisition",
    )
    ax1.errorbar(laser_powers, taus, yerr=taus_err, marker="o", label="Reprocessed")
    ax1.legend()
    ax1.set_xlabel("Power in a.u.")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{path_dn}/{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def compute_phase_dist(scan: str, plot: bool = False):
    """Computes the phase from the interferograms for a distance scan
    dataset

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    # powers = np.load(f"{path_real}/{scan}/powers.npy")
    # N_times = powers.size
    input_power = np.load(f"{path_real}/{scan}/input_power.npy")
    N_points = input_power.size
    intervortex_dists = np.load(f"{path_real}/{scan}/intervortex_dists.npy")
    field = np.zeros((N_points, Ny // 2, Nx // 2), dtype=np.complex64)
    field_ref = np.zeros((N_points, Ny // 2, Nx // 2), dtype=np.complex64)
    field_vortex = np.zeros((N_points, Ny // 2, Nx // 2), dtype=np.complex64)
    taus = np.zeros(N_points)
    for i in tqdm.tqdm(range(N_points), desc="Computing phase", position=1):
        im_ref = np.array(
            Image.open(f"{path_dn}/{scan}/dn_ref_{i}.tiff"), dtype=np.uint16
        )
        im = np.array(Image.open(f"{path_dn}/{scan}/dn_{i}.tiff"), dtype=np.uint16)
        im_vortex = np.array(
            Image.open(f"{path_dn}/{scan}/dn_vortex.tiff"), dtype=np.uint16
        )
        field_ref[i, :, :] = contrast.im_osc_fast_t(im_ref)
        field[i, :, :] = contrast.im_osc_fast_t(im)
        field_vortex = contrast.im_osc_fast_t(im_vortex)
        phi_ref = contrast.angle_fast(field_ref[i, :, :])
        rho_ref = field_ref[i, :, :] * np.conj(field_ref[i, :, :])
        rho_ref = rho_ref.real
        threshold = 1e-2
        mask = rho_ref < threshold * np.max(rho_ref)
        phi_ref_masked = np.ma.array(phi_ref, mask=mask)
        phi_ref_unwrapped = restoration.unwrap_phase(
            phi_ref_masked, wrap_around=(True, True)
        )
        taus[i] = np.abs(np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped))

    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{path_dn}/{scan}/field_ref.npy", field_ref)
    np.save(f"{path_dn}/{scan}/field.npy", field)
    np.save(f"{path_dn}/{scan}/field_vortex.npy", field_vortex)
    np.save(f"{path_dn}/{scan}/taus.npy", taus)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(f"Saved {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[-1, :, :]))
    ax[1].imshow(np.angle(field[-1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{path_dn}/{scan}/field.svg", dpi=300)
    fig1, ax1 = plt.subplots()
    ax1.plot(intervortex_dists, taus, marker="o")
    ax1.set_xlabel("Initial dist in px")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{path_dn}/{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def compute_phase_time_single(
    scan: str, name_dn="dn_mvortex", name_field="field_mvortex", plot: bool = False
):
    """Compute the phase.

    Computes the phase from the interferograms for a time scan
    dataset.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    field_npy = np.load(f"{scan}/{name_field}.npy")
    N_times = field_npy.shape[0]
    N_avg = field_npy.shape[1]
    field_c = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    # prepare plans
    a = pyfftw.empty_aligned((Ny, Nx), dtype=np.float32)
    c = pyfftw.empty_aligned((Ny // 2, Nx // 2), dtype=np.complex64)
    plan_fft = pyfftw.builders.rfft2(a)
    plan_ifft = pyfftw.builders.ifft2(c)
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plans = (plan_fft, plan_ifft)
    pbar = tqdm.tqdm(total=N_times * N_avg, desc="Computing phase", position=1)
    for i in range(N_times):
        for j in range(N_avg):
            im_mvortex = np.array(
                Image.open(f"{scan}/{name_dn}_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            field_c[i, j, :, :] = contrast.im_osc_fast_t(
                im_mvortex,
                plans=plans,
                center=(im_mvortex.shape[-2] // 2, im_mvortex.shape[-1] // 4),
                radius=im_mvortex.shape[-2] // 8,
            )

            pbar.update(1)
    pbar.close()
    np.save(f"{scan}/field_mvortex.npy", field_c)


def compute_n2_Isat(scan: str, plot: bool = False):
    print("Loading data ...")

    def delta_n_avg(I, I_sat, n2, b):
        return n2 * I / (1 + I / I_sat) ** 2 + b

    waist = 1.7e-3
    L = 0.2
    k0 = 2 * np.pi / 780e-9
    taus = np.load(f"{scan}/taus.npy")
    # taus = taus[: taus.size // 2]
    # plt.figure()
    # plt.plot(taus)
    # plt.show()
    delta_n = taus / (k0 * L)
    # laser_powers = np.load(f"{scan}/laser_settings.npy")
    laser_powers = np.linspace(0.05, 2.56, len(delta_n))
    laser_intensity = 2 * laser_powers / (np.pi * (waist) ** 2)
    # laser_intensity = laser_intensity[5:]
    # delta_n = delta_n[5:]
    popt, pcov = curve_fit(delta_n_avg, laser_intensity, delta_n, p0=[6e5, 1e-10, 1])
    plt.figure()
    plt.plot(laser_intensity, delta_n, marker="o")
    plt.plot(
        laser_intensity,
        delta_n_avg(laser_intensity, *popt),
        label=f"Isat = {popt[0]:.2e} W/m$^2$ | n2 = {popt[1]:.2e} m$^2$/W",
    )
    plt.xlabel("Laser intensity (W/m$^2$)")
    plt.ylabel(r"$\Delta n$")
    plt.title("T=1%")
    plt.legend()
    plt.grid()
    plt.savefig(f"{scan}/n2_Isat.svg", dpi=300)
    plt.show()


def compute_final_xi(scan: str, plot: bool = False):
    """Find final xi value"""
    print("Loading data ...")
    t0 = time.perf_counter()
    fields_ref = np.load(f"{scan}/field_ref.npy")
    fields = np.load(f"{scan}/field.npy")
    fields_vortex = np.load(f"{scan}/field_vortex.npy")
    taus = np.load(f"{scan}/taus.npy")
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    window_m = 10
    vort_size = []
    vort_size_err = []
    xi_avg = []
    for i in tqdm.tqdm(range(fields.shape[0]), desc="Plotting", position=1):
        vort_size_avg = []
        for j in range(fields.shape[1]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]
            field_vortex = fields_vortex[i, j, :, :]
            field = ndimage.gaussian_filter(field, 2)
            field_ref = ndimage.gaussian_filter(field_ref, 2)
            field_vortex = ndimage.gaussian_filter(field_vortex, 2)
            # Compute the phase and density
            contrast.exp_angle_fast(field, field_ref)
            contrast.exp_angle_fast(field_vortex, field_ref)
            phi_vort_flat = contrast.angle_fast(field_vortex)
            rho = field.real * field.real + field.imag * field.imag
            # rho = ndimage.gaussian_filter(rho, 3)
            rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            rho_vort = (
                field_vortex.real * field_vortex.real
                + field_vortex.imag * field_vortex.imag
            )
            rho = np.sqrt(rho)
            rho = rho / np.nanmax(rho)
            rho_ref = np.sqrt(rho_ref)
            rho_ref = rho_ref / np.nanmax(rho_ref)
            rho_vort = np.sqrt(rho_vort)
            rho_vort = rho_vort / np.nanmax(rho_vort)

            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi)
            window = int(radius_mask_ref) / 1.5
            x, y = np.meshgrid(
                np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2])
            )
            mask_fluid = np.abs(
                (x - mask_ref.shape[-1] // 2) ** 2 + (y - mask_ref.shape[-2] // 2) ** 2
                < radius_mask_ref**2
            )
            mask_fluid = np.logical_not(mask_fluid)
            rho_masked = rho
            rho_masked[mask_fluid] = np.nan
            # plt.figure()
            # plt.imshow(rho_masked, cmap='gray')
            # plt.show()

            ####### Compute xi with vortex radius
            vortices = np.array([])
            vort = velocity.vortex_detection(phi_vort_flat, plot=False, r=50)
            vort[:, 0] -= phi_vort_flat.shape[-1] // 2
            vort[:, 1] -= phi_vort_flat.shape[-2] // 2
            # vort_in_window_x = np.logical_and(vort[:, 0] > -window, vort[:, 0] < window)
            # vort_in_window_y = np.logical_and(vort[:, 1] > -window, vort[:, 1] < window)
            # vort_select = np.logical_and(vort_in_window_x, vort_in_window_y)
            vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > window**2)
            vortices = vort[vort_select, :]
            try:
                center_x = field.shape[1] // 2 + vortices[0][0]
                center_y = field.shape[0] // 2 + vortices[0][1]
            except:
                vort_rad = np.nan
                continue
            # Find the vortex position, and width
            rho_zoom = rho_vort[
                int(center_y - window_m) : int(center_y + window_m),
                int(center_x - window_m) : int(center_x + window_m),
            ]
            x = np.arange(rho_zoom.shape[1])
            y = np.arange(rho_zoom.shape[0])
            X, Y = np.meshgrid(x, y)
            inital_guess = (window_m, window_m, 10)
            try:
                popt, pcov = opt.curve_fit(
                    Vortex2D, (Y, X), rho_zoom.ravel(), p0=inital_guess
                )
            except RuntimeError:
                popt = np.array([window_m, window_m, np.nan])
            vortex_reshape = Vortex2D((Y, X), *popt).reshape(window_m * 2, window_m * 2)
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(rho_zoom, cmap="inferno")
            # ax[1].imshow(vortex_reshape, cmap="inferno")
            # plt.show()
            vort_rad = popt[2] * d_real
            if vort_rad > 250e-6:
                vort_rad = np.nan
            vort_size_avg += [vort_rad]
        vort_size_avg = np.array(vort_size_avg)
        vort_size += [np.nanmean(vort_size_avg)]
        vort_size_err += [np.nanstd(vort_size_avg)]
        xi_avg += [np.sqrt(0.2 / (taus[i] * 8e6))]
    vort_size = np.array(vort_size)
    vort_size_err = np.array(vort_size_err)
    xi_avg = np.array(xi_avg)
    T = 0.06
    deltan = taus / (k0 * L)
    deltan_T = deltan * ((T * np.log(T)) / (T - 1))
    xi_avg_T = 1 / (k0 * np.sqrt(deltan_T))
    np.save(f"{scan}/xis_final.npy", vort_size)
    np.save(f"{scan}/xis_final_abs.npy", xi_avg_T)
    plt.figure()
    plt.errorbar(
        taus,
        vort_size * 1e6,
        yerr=vort_size_err * 1e6,
        marker="o",
        label="Vortex radius",
    )
    # plt.plot(taus, vort_size*1e6, marker='o', label='Vortex radius')
    plt.plot(taus, xi_avg * 1e6, marker="s", label="Average xi")
    plt.plot(taus, xi_avg_T * 1e6, marker="s", label="Average xi with absorption")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\xi$ $(\mu m)$")
    plt.grid()
    plt.legend()
    plt.savefig(f"{scan}/xi_vortex.svg", dpi=300)
    if plot:
        plt.show()


def plot_fields(scan: str, window: int = 50, plot: bool = False):
    """Plot the fields.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the fields. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    taus = np.load(f"{scan}/taus.npy")
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    field = fields[:, -1, :, :]
    field_ref = fields_ref[:, -1, :, :]
    rate = sz / t
    # if field_ref.shape[-1] == Nx // 2:
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    delta_n = taus / (k0 * L)
    xis = np.load(f"{scan}/xis.npy")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    im0 = ax[0].imshow(
        np.ones(field.shape[1:]),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    im1 = ax[1].imshow(
        np.ones(field.shape[1:]),
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
    )
    for a in ax:
        a.set_xlabel(r"$x/\xi$")
        a.set_ylabel(r"$y/\xi$")
    ax[0].set_title("Density")
    ax[1].set_title("Normalized phase")
    # ax[0].locator_params(axis="both", nbins=5)
    # ax[1].locator_params(axis="both", nbins=5)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    fig.colorbar(im0, ax=ax[0], label=r"$\sqrt{\rho}$ in a.u.", shrink=0.6)
    cbar = fig.colorbar(im1, ax=ax[1], label=r"$\phi/\phi_0$ in rad", shrink=0.6)
    cbar.set_ticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[
            r"$\pi$",
            r"$-\frac{\pi}{2}$",
            r"$0$",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
        ],
    )
    fig.canvas.draw()
    mat = np.array(fig.canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # create folder if it does not exist
    if not (os.path.exists(f"{scan}")):
        os.mkdir(f"{scan}")
    video_writer = cv2.VideoWriter(
        f"{scan}/field.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    video_writer1 = cv2.VideoWriter(
        f"{scan}/field_rescaled.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    for i in tqdm.tqdm(range(field.shape[0]), desc="Plotting fields", position=1):
        fig.suptitle(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i] * 1e6:.0f} $\mu m$")
        ext_real = [
            -field.shape[-1] * d_real / (2 * xis[i]),
            field.shape[-1] * d_real / (2 * xis[i]),
            -field.shape[-2] * d_real / (2 * xis[i]),
            field.shape[-2] * d_real / (2 * xis[i]),
        ]
        contrast.exp_angle_fast(field[i, :, :], field_ref[i, :, :])
        phi_flat = contrast.angle_fast(field[i, :, :])
        rho = (
            field[i, :, :].real * field[i, :, :].real
            + field[i, :, :].imag * field[i, :, :].imag
        )
        rho = np.sqrt(rho)
        rho = rho / np.max(rho)
        im0.set_data(rho)
        im0.set_clim(np.min(rho), 1.0)
        im1.set_data(phi_flat)
        im0.set_extent(ext_real)
        im1.set_extent(ext_real)
        for a in ax:
            a.set_xlim((ext_real[0], ext_real[1]))
            a.set_ylim((ext_real[2], ext_real[3]))
        fig.canvas.draw()
        if plot:
            plt.show()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer.write(mat)
        fig.savefig(f"{scan}/fields_{i}.svg", dpi=300)
        window = window
        for a in ax:
            a.set_xlim((-window, window))
            a.set_ylim((-window, window))
        fig.canvas.draw()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer1.write(mat)
        fig.savefig(f"{scan}/fields_rescaled_{i}.svg", dpi=300)
        if plot:
            plt.show()
    print()
    plt.close("all")
    video_writer.release()
    video_writer1.release()


def plot_energy(scan: str, plot: bool = False):
    """Plot the energies.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the fields. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    taus = np.load(f"{scan}/taus.npy")
    u_comp = np.load(f"{scan}/u_comp.npy")
    u_inc = np.load(f"{scan}/u_inc.npy")
    t = time.perf_counter() - t0
    sz = u_comp.nbytes + u_inc.nbytes
    rate = sz / t
    # if u_comp.shape[-1] == Nx // 2:
    d_real = 3.45e-6 * 2
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    delta_n = taus / (k0 * L)
    xis = 1 / (k0 * np.sqrt(delta_n))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    im0 = ax[0].imshow(
        np.ones(u_comp.shape[-2:]),
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    im1 = ax[1].imshow(
        np.ones(u_inc.shape[-2:]),
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    for a in ax:
        a.set_xlabel(r"$x/\xi$")
        a.set_ylabel(r"$y/\xi$")
    ax[0].set_title(r"$|u_{comp}|^2$")
    ax[1].set_title(r"$|u_{inc}|^2$")
    # ax[0].locator_params(axis="both", nbins=5)
    # ax[1].locator_params(axis="both", nbins=5)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    fig.colorbar(im0, ax=ax[0], shrink=0.6)
    fig.colorbar(im1, ax=ax[1], shrink=0.6)
    fig.canvas.draw()
    mat = np.array(fig.canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # create folder if it does not exist
    if not (os.path.exists(f"{scan}")):
        os.mkdir(f"{scan}")
    video_writer = cv2.VideoWriter(
        f"{scan}/energy.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    video_writer1 = cv2.VideoWriter(
        f"{scan}/energy_rescaled.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    # comp_max = np.zeros(u_comp.shape[0])
    # inc_max = np.zeros(u_inc.shape[0])
    # for i in range(u_inc.shape[0]):
    #     comp_max[i] = np.max((np.abs(u_comp[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0))
    #     inc_max[i] = np.max((np.abs(u_inc[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0))
    # comp_max = np.max(comp_max)
    # inc_max = np.max(inc_max)
    for i in tqdm.tqdm(range(u_inc.shape[0]), desc="Plotting energies", position=1):
        # print('image ', i)
        fig.suptitle(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i] * 1e6:.0f} $\mu m$")
        ext_real = [
            -u_comp.shape[-1] * d_real / (2 * xis[i]),
            u_comp.shape[-1] * d_real / (2 * xis[i]),
            -u_comp.shape[-2] * d_real / (2 * xis[i]),
            u_comp.shape[-2] * d_real / (2 * xis[i]),
        ]
        comp = (np.abs(u_comp[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0)
        comp /= np.nanmax(comp)
        inc = (np.abs(u_inc[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0)
        inc /= np.nanmax(inc)
        im0.set_data(comp)
        # im0.set_clim(np.nanmin(comp), np.nanmax(comp))
        im1.set_data(inc)
        # im1.set_clim(np.nanmin(inc), np.nanmax(inc))
        im0.set_extent(ext_real)
        im1.set_extent(ext_real)
        for a in ax:
            a.set_xlim((ext_real[0], ext_real[1]))
            a.set_ylim((ext_real[2], ext_real[3]))
        fig.canvas.draw()
        if plot:
            plt.show()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer.write(mat)
        fig.savefig(f"{scan}/energies_{i}.svg", dpi=300)
        window = 100
        for a in ax:
            a.set_xlim((-window, window))
            a.set_ylim((-window, window))
        fig.canvas.draw()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer1.write(mat)
        fig.savefig(f"{scan}/energies_rescaled_{i}.svg", dpi=300)
        if plot:
            plt.show()
    plt.close("all")
    video_writer.release()
    video_writer1.release()


def energy_time(scan: str, plot: bool = False):
    """Compute the energy.

    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    field_ref = np.load(f"{scan}/field_ref.npy")

    N_times = field_ref.shape[0]
    N_avg = field_ref.shape[1]
    field = np.load(f"{scan}/field.npy")
    field_vortex = np.load(f"{scan}/field_vortex.npy")
    taus = np.load(f"{scan}/taus.npy")
    taus_err = np.load(f"{scan}/taus_err.npy")
    t = time.perf_counter() - t0

    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    filtering_radius = 4
    # field = ndimage.gaussian_filter(field, filtering_radius)
    # field_vortex = ndimage.gaussian_filter(field_vortex, filtering_radius)
    # compute energies
    e_comp = np.zeros(field.shape[:2])
    e_inc = np.zeros(field.shape[:2])
    e_int = np.zeros(field.shape[:2])
    u_inc = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    u_comp = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    e_comp_vortex = np.zeros(field.shape[:2])
    e_inc_vortex = np.zeros(field.shape[:2])
    e_int_vortex = np.zeros(field.shape[:2])
    u_inc_vortex = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    u_comp_vortex = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
        # for i in range(30,31):
        for j in range(N_avg):
            contrast.exp_angle_fast(field[i, j, :, :], field_ref[i, j, :, :])
            contrast.exp_angle_fast_scalar(
                field[i, j, :, :],
                field[i, j, field.shape[-2] // 2, field.shape[-1] // 2],
            )
            contrast.exp_angle_fast(field_vortex[i, j, :, :], field_ref[i, j, :, :])
            contrast.exp_angle_fast_scalar(
                field_vortex[i, j, :, :],
                field_vortex[
                    i,
                    j,
                    field_vortex.shape[-2] // 2,
                    field_vortex.shape[-1] // 2,
                ],
            )
            rho = (
                field[i, j, :, :].real * field[i, j, :, :].real
                + field[i, j, :, :].imag * field[i, j, :, :].imag
            )
            rho_vortex = (
                field_vortex[i, j, :, :].real * field_vortex[i, j, :, :].real
                + field_vortex[i, j, :, :].imag * field_vortex[i, j, :, :].imag
            )
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi) / 1.3
            x, y = np.meshgrid(
                np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2])
            )
            mask_fluid = (x - mask_ref.shape[-1] // 2) ** 2 + (
                y - mask_ref.shape[-2] // 2
            ) ** 2 < radius_mask_ref**2
            mask_fluid = np.logical_not(mask_fluid)

            field[i, j, :, :] = ndimage.gaussian_filter(
                field[i, j, :, :], filtering_radius
            )
            (
                _,
                u_inc[i, j, :, :, :],
                u_comp[i, j, :, :, :],
            ) = velocity.helmholtz_decomp(field[i, j, :, :], dx=d_real, plot=False)
            # u_inc[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_inc[i, j, :, :, :], filtering_radius
            # )
            # u_comp[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_comp[i, j, :, :, :], filtering_radius
            # )
            for k in range(u_comp_vortex.shape[2]):
                u_comp[i, j, k, :, :][mask_fluid] = 0
                u_inc[i, j, k, :, :][mask_fluid] = 0
            e_comp[i, j] = np.sum(u_comp[i, j, :, :, :] ** 2)
            e_inc[i, j] = np.sum(u_inc[i, j, :, :, :] ** 2)
            e_int[i, j] = np.sum(rho * rho)

            field_vortex[i, j, :, :] = ndimage.gaussian_filter(
                field_vortex[i, j, :, :], filtering_radius
            )
            (
                _,
                u_inc_vortex[i, j, :, :, :],
                u_comp_vortex[i, j, :, :, :],
            ) = velocity.helmholtz_decomp(
                field_vortex[i, j, :, :], dx=d_real, plot=False
            )
            # u_inc_vortex[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_inc_vortex[i, j, :, :, :], filtering_radius
            # )
            # u_comp_vortex[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_comp_vortex[i, j, :, :, :], filtering_radius
            # # )
            for jj in range(u_comp_vortex.shape[2]):
                u_comp_vortex[i, j, k, :, :][mask_fluid] = 0
                u_inc_vortex[i, j, k, :, :][mask_fluid] = 0
            e_comp_vortex[i, j] = np.sum(np.abs(u_comp_vortex[i, j, :, :, :]) ** 2)
            e_inc_vortex[i, j] = np.sum(np.abs(u_inc_vortex[i, j, :, :, :]) ** 2)
            e_int_vortex[i, j] = np.sum(rho_vortex * rho_vortex)
    print("\nSaving data ...")
    t0 = time.perf_counter()
    # np.save(f"{scan}/e_comp.npy", e_comp)
    # np.save(f"{scan}/e_inc.npy", e_inc)
    # np.save(f"{scan}/e_int.npy", e_int)
    np.save(f"{scan}/u_inc.npy", u_inc)
    np.save(f"{scan}/u_comp.npy", u_comp)
    # np.save(f"{scan}/e_comp_vortex.npy", e_comp_vortex)
    # np.save(f"{scan}/e_inc_vortex.npy", e_inc_vortex)
    # np.save(f"{scan}/e_int_vortex.npy", e_int_vortex)
    # np.save(f"{scan}/u_inc_vortex.npy", u_inc_vortex)
    # np.save(f"{scan}/u_comp_vortex.npy", u_comp_vortex)
    t = time.perf_counter() - t0
    sz = (
        e_comp.nbytes
        + e_inc.nbytes
        + u_inc.nbytes
        + u_comp.nbytes
        + e_comp_vortex.nbytes
        + e_inc_vortex.nbytes
        + u_inc_vortex.nbytes
        + u_comp_vortex.nbytes
    )
    rate = sz / t
    print(f"Saved {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title(r"$E_{two}$")
    ax[0].errorbar(
        taus,
        np.mean(e_comp, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_comp, axis=-1),
        label="Compressible",
        marker="o",
    )
    ax[0].errorbar(
        taus,
        np.mean(e_inc, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_inc, axis=-1),
        label="Incompressible",
        marker="^",
    )
    # ax[0].errorbar(
    #     taus,
    #     np.mean(e_int, axis=-1),
    #     xerr=taus_err,
    #     yerr=np.std(e_int, axis=-1),
    #     label="Interaction",
    #     marker="^",
    # )
    ax[1].set_title(r"$E_{one}$")
    ax[1].errorbar(
        taus,
        np.mean(e_comp_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_comp_vortex, axis=-1),
        label="Compressible",
        marker="o",
    )
    ax[1].errorbar(
        taus,
        np.mean(e_inc_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_inc_vortex, axis=-1),
        label="Incompressible",
        marker="^",
    )
    # ax[1].errorbar(
    #     taus,
    #     np.mean(e_int_vortex, axis=-1),
    #     xerr=taus_err,
    #     yerr=np.std(e_int_vortex, axis=-1),
    #     label="Interaction",
    #     marker="^",
    # )
    ax[2].set_title(r"$E_{vortex}/E_{fluid}$")
    ax[2].errorbar(
        taus,
        np.mean(e_comp, axis=-1) / np.mean(e_comp_vortex, axis=-1),
        yerr=np.std(e_comp / e_comp_vortex, axis=-1),
        xerr=taus_err,
        label="Compressible",
        marker="o",
    )
    # ax[2].errorbar(
    #     taus,
    #     np.mean(e_inc, axis=-1) / np.mean(e_inc_vortex, axis=-1),
    #     yerr=np.std(e_inc / e_inc_vortex, axis=-1),
    #     xerr=taus_err,
    #     label="Incompressible",
    #     marker="^",
    # )
    # ax[2].errorbar(
    #     taus,
    #     np.mean(e_int, axis=-1) / np.mean(e_int_vortex, axis=-1),
    #     yerr=np.std(e_int / e_int_vortex, axis=-1),
    #     xerr=taus_err,
    #     label="Interaction",
    #     marker="s",
    # )
    for a in ax:
        a.set_xlabel(r"$\tau$ in rad")
        a.set_ylabel(r"$E$ in a.u.")
        a.legend()
    ax[2].axhline(1, color="grey", linestyle="--")
    ax[2].set_ylim(bottom=-0.05)
    fig.savefig(f"{scan}/energies.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def velocity_fields(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 7.5
    x0 = 0
    y0 = 0
    rad = 3
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    v_x = []

    # for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
    # which = np.arange(18, 20, 1)
    # which = np.arange(0, 20, 2)
    which = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
    print("\n", taus[25:40], "\n")
    fields_vis = np.zeros(
        (len(which), fields.shape[-2], fields.shape[-1]), dtype=np.complex64
    )
    u_tot_vis = np.zeros(
        (len(which), 2, fields.shape[-2], fields.shape[-1]), dtype=np.float32
    )
    for i in which:
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)

            extent = np.array(
                [
                    -field.shape[-1] * d_real / (2 * xi),
                    field.shape[-1] * d_real / (2 * xi),
                    -field.shape[-2] * d_real / (2 * xi),
                    field.shape[-2] * d_real / (2 * xi),
                ]
            )

            (
                u_tot,
                _,
                _,
            ) = velocity.helmholtz_decomp_phase(field, dx=d_real, plot=False)
            # u_tot = velocity.velocity(np.angle(field))

            field_zoom = field[
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_tot_zoom_0 = u_tot[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_tot_zoom_1 = u_tot[
                1,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            flow_tot = np.hypot(u_tot_zoom_0, u_tot_zoom_1)
            flow_tot = flow_tot / np.nanmax(flow_tot)
            YY, XX = np.indices(flow_tot.shape)

            # plt.figure()
            # plt.imshow(u_tot_zoom_0)
            # plt.colorbar()
            # plt.show()
            v_x_i = u_tot_zoom_0[u_tot_zoom_0.shape[0] // 2 - 6, :]
            v_x += [v_x_i]

            amp_zoom = np.abs(field_zoom)
            amp_zoom = amp_zoom / np.nanmax(amp_zoom)
            phase_zoom = contrast.angle_fast(field_zoom)

            # vortices = velocity.vortex_detection(phase_zoom, plot=True, r=7)
            sell = velocity.sell_detection(np.angle(field), plot=False, r=6)
            sell_array = np.zeros(field.shape[:2])
            for k in range(sell.shape[0]):
                sell_array[int(sell[k, 1]), int(sell[k, 0])] = sell[k, 2]
            sell_array[sell_array == 0] = np.nan
            sell_array_zoom = sell_array[
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            fig, ax = plt.subplots(1, 3, figsize=[12, 5])
            fig.suptitle(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xi * 1e6:.0f} $\mu m$")
            im0 = ax[0].imshow(
                amp_zoom,
                cmap="gray",
                interpolation="none",
            )
            ax[0].set_title("Amplitude")
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            fig.colorbar(im0, ax=ax[0], label=r"$\sqrt{\rho}$")
            im1 = ax[1].imshow(
                phase_zoom,
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="none",
            )
            ax[1].set_title("Phase")
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("y")
            fig.colorbar(im1, ax=ax[1], label=r"$\phi$ in rad")
            im2 = ax[2].imshow(flow_tot, cmap="viridis", vmin=0, vmax=1)
            ax[2].streamplot(
                XX,
                YY,
                u_tot_zoom_0,
                u_tot_zoom_1,
                density=2,
                color="white",
                linewidth=0.5,
            )
            ax[2].set_title(r"$u^{tot}$")
            ax[2].set_xlabel("x")
            ax[2].set_ylabel("y")
            fig.colorbar(im2, ax=ax[2], label=r"$|u^{tot}|$")
            x_indices, y_indices = np.meshgrid(
                np.arange(sell_array_zoom.shape[1]), np.arange(sell_array_zoom.shape[0])
            )
            # Identifier les indices des valeurs spécifiques
            saddle_mask = sell_array_zoom.ravel() == 1  # Points où c = 1
            node_mask = sell_array_zoom.ravel() == -1  # Points où c = -1

            # Scatter pour les "saddle" (c = 1)
            ax[2].scatter(
                x_indices.ravel()[saddle_mask],
                y_indices.ravel()[saddle_mask],
                color="tab:green",
                marker="D",
                label="Saddle",
                zorder=3,
            )

            # Scatter pour les "node" (c = -1)
            ax[2].scatter(
                x_indices.ravel()[node_mask],
                y_indices.ravel()[node_mask],
                color="tab:orange",
                marker="D",
                label="Node",
                zorder=3,
            )

            # Ajouter une légende
            ax[2].legend(loc="upper left", fontsize=10)

            scale_factor = xi / d_real
            for a in ax:
                a.set_xlim(0, sell_array_zoom.shape[1])
                a.set_ylim(0, sell_array_zoom.shape[0])
                x_ticks = np.linspace(0, sell_array_zoom.shape[1], num=3)
                y_ticks = np.linspace(0, sell_array_zoom.shape[0], num=5)
                a.set_xticks(x_ticks)
                a.set_yticks(y_ticks)
                a.set_xticklabels(
                    [f"{tick / scale_factor:.0f}" for tick in a.get_xticks()]
                )
                a.set_yticklabels(
                    [f"{tick / scale_factor:.0f}" for tick in a.get_yticks()]
                )
                a.set_xlabel(r"$x/\xi$")
                a.set_ylabel(r"$y/\xi$")

            plt.savefig(f"{scan}/velocity_{i}.png", dpi=300)
            if plot:
                plt.show()

            fields_vis[i - which[0], :, :] = field
            u_tot_vis[i - which[0], :, :] = u_tot
        # np.save(f"{scan}/fields_vis.npy", fields_vis)
        # np.save(f"{scan}/u_tot_vis.npy", u_tot_vis)


def vortex_tracking(scan, windows_size, plot):
    t0 = time.perf_counter()
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    # fields = np.flip(fields, axis=1)
    # fields_ref = np.flip(fields_ref, axis=1)
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")

    # Stocker les positions pour chaque vortex
    num_vortices_max = 4
    vortex_trajectories = [[] for _ in range(num_vortices_max)]
    previous_vortices = None  # Positions des vortex de l'étape précédente
    rad = 2
    all_vortices = []

    rhos_zoom = []
    good_index = []
    for i in tqdm.tqdm(range(fields.shape[0]), position=1):
        for j in range(fields.shape[1]):
            field = fields[i, j, :, :]
            rho_raw = np.abs(field)
            if j == 0:
                rhos_zoom += [rho_raw]
            field_ref = fields_ref[i, j, :, :]

            # Compute the phase and density
            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            field_ref = ndimage.gaussian_filter(field_ref, rad)
            phi_vort_flat = contrast.angle_fast(field)
            rho = field.real * field.real + field.imag * field.imag
            rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            rho = np.sqrt(rho)
            rho = rho / np.nanmax(rho)
            rho_ref = np.sqrt(rho_ref)
            rho_ref = rho_ref / np.nanmax(rho_ref)

            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi) / 1.5
            window = int(radius_mask_ref)
            x, y = np.meshgrid(
                np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2])
            )
            mask_fluid = np.abs(
                (x - mask_ref.shape[-1] // 2) ** 2 + (y - mask_ref.shape[-2] // 2) ** 2
                < radius_mask_ref**2
            )
            mask_fluid = np.logical_not(mask_fluid)
            rho_masked = rho
            rho_masked[mask_fluid] = np.nan

            vort = velocity.vortex_detection(phi_vort_flat, plot=False, r=2)
            vort[:, 0] -= phi_vort_flat.shape[-1] // 2
            vort[:, 1] -= phi_vort_flat.shape[-2] // 2
            vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > window**2)
            vortices = vort[vort_select, :]

            w = windows_size * xis[i] / d_real
            w = int(w)
            roi = (
                slice(rho.shape[-2] // 2 - w, rho.shape[-2] // 2 + w),
                slice(rho.shape[-1] // 2 - w, rho.shape[-1] // 2 + w),
            )
            vortices[:, 0] *= d_real / xis[i]
            vortices[:, 1] *= d_real / xis[i]
            if j == 0 and i != fields.shape[0] - 1:
                all_vortices += [vortices]
            if i == fields.shape[0] - 1 and j == 1:
                all_vortices += [vortices]

            if len(vortices) != num_vortices_max:
                print(
                    f"'\n', Warning: Expected {num_vortices_max} vortices, but found {len(vortices)} at index {i}, {j} \n",
                )
                continue
            good_index += [i]

            if previous_vortices is not None:
                distances = np.linalg.norm(
                    vortices[:, None, :] - previous_vortices[None, :, :], axis=-1
                )
                from scipy.optimize import linear_sum_assignment

                row_ind, col_ind = linear_sum_assignment(distances)
                vortices = vortices[col_ind]  # Réordonner les vortex actuels
            for v_idx in range(num_vortices_max):
                vortex_trajectories[v_idx].append(vortices[v_idx])
            previous_vortices = vortices.copy()

            break

    rhos_zoom_a = np.array(rhos_zoom)
    good_index = np.array(good_index)
    vortex_trajectories = [np.array(traj) for traj in vortex_trajectories]
    other_vortices = np.array(all_vortices)
    vortex = ["-", "-", "+", "+"]
    color = ["tab:blue", "lightsteelblue", "darksalmon", "tab:red"]
    print("Plotting ...")
    for i in range(len(rhos_zoom_a)):
        plt.figure()
        ext_real = [
            -rhos_zoom_a.shape[-1] * d_real / (2 * xis[i]),
            rhos_zoom_a.shape[-1] * d_real / (2 * xis[i]),
            rhos_zoom_a.shape[-2] * d_real / (2 * xis[i]),
            -rhos_zoom_a.shape[-2] * d_real / (2 * xis[i]),
        ]
        plt.title(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i] * 1e6:.0f} $\mu m$")
        plt.imshow(rhos_zoom_a[i], cmap="gray", extent=ext_real)
        plt.xlim(-windows_size, windows_size)
        plt.ylim(-windows_size, windows_size)
        plt.grid()
        plt.xlabel(r"$x/\xi$")
        plt.ylabel(r"$y/\xi$")
        if i in good_index:
            valid_trajectories = [traj for traj in vortex_trajectories if len(traj) > i]
            for v_idx, trajectory in enumerate(valid_trajectories):
                trajectory = np.array(trajectory)
                plt.plot(
                    trajectory[: i + 1, 0],
                    trajectory[: i + 1, 1],
                    "-",
                    color=color[v_idx],
                )

                if len(trajectory) > i:
                    plt.scatter(
                        trajectory[i, 0],
                        trajectory[i, 1],
                        facecolor="none",
                        edgecolors=color[v_idx],
                        label=f"{vortex[v_idx]}",
                    )
        else:
            for k in range(len(other_vortices[i])):
                if other_vortices[i][k, 2] == 1:
                    plt.scatter(
                        other_vortices[i][k, 0],
                        other_vortices[i][k, 1],
                        facecolor="none",
                        edgecolors="lightsteelblue",
                        label="-",
                    )
                if other_vortices[i][k, 2] == -1:
                    plt.scatter(
                        other_vortices[i][k, 0],
                        other_vortices[i][k, 1],
                        facecolor="none",
                        edgecolors="darksalmon",
                        label="+",
                    )
        plt.legend()
        plt.savefig(f"{scan}/traj_{i}.png", dpi=300)
        if plot:
            plt.show()

    return vortex_trajectories


def make_mp4_and_gif(scan, fps=2):
    import imageio as imageio

    img_array = []
    images = []

    # Récupération et tri des fichiers PNG
    lst = os.listdir(scan)
    lst_sorted = sorted(
        [filename for filename in lst if filename.endswith(".png")],
        key=lambda x: int(x.split(".")[0].split("_")[1]),
    )

    print("Images triées :", lst_sorted)

    size = None

    idx = 0
    for filename in lst_sorted:
        img_path = os.path.join(scan, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Impossible de lire l'image {filename}, ignorée.")
            continue

        height, width, layers = img.shape
        size = (
            (width, height) if size is None else size
        )  # Définit la taille sur la première image

        if (idx == 0) or (idx == len(lst_sorted) - 1):
            img_array.append(img)
            img_array.append(img)
            images.append(Image.open(img_path))
            images.append(Image.open(img_path))
        img_array.append(img)
        images.append(Image.open(img_path))
        idx += 1

    # Création de la vidéo MP4
    if img_array:
        mp4_path = os.path.join(scan, "vortex_tracking.mp4")
        out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

        for i in range(len(img_array)):
            out.write(img_array[i])

        out.release()
        print(f"Vidéo MP4 créée : {mp4_path}")
    else:
        print("Aucune image valide pour créer une vidéo.")

    # Création du GIF
    if images:
        gif_path = os.path.join(scan, "vortex_tracking.gif")
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),  # durée en ms par image
            loop=0,  # boucle infinie
        )
        print(f"GIF créé : {gif_path}")
    else:
        print("Aucune image valide pour créer un GIF.")


def one_jrs(scan, plot):
    print("Loading data ...")
    xi = 40e-6
    im = Image.open(f"{scan}/fluid_p.tiff")
    im_ref = Image.open(f"{scan}/ref_p.tiff")
    fluid = np.array(im)
    ref = np.array(im_ref)
    h, w = (206, 206)
    # h, w = (512, 512)
    # h, w = (1024, 1024)
    filtering_radius = 4
    exponent = 4

    field_ref = contrast.im_osc_fast_t(
        fluid / ref,
        center=(fluid.shape[-2] // 2, fluid.shape[-1] // 4),
        radius=fluid.shape[-2] // 8,
    )

    x0 = 0
    y0 = 0
    roi = 15
    roi_xi = int(roi * (xi / d_real))

    for i in np.array([4]):
        dn = Image.open(f"{scan}/pair_{i}.tiff")
        im_field = np.array(dn)
        field = contrast.im_osc_fast_t(
            im_field / ref,
            center=(im_field.shape[-2] // 2, im_field.shape[-1] // 4),
            radius=im_field.shape[-2] // 8,
        )
        contrast.exp_angle_fast(field, field_ref)

        # import cv2

        # field = cv2.bilateralFilter(
        #     np.abs(field).astype(np.float32), d=5, sigmaColor=150, sigmaSpace=75
        # ) * np.exp(
        #     1j
        #     * cv2.bilateralFilter(
        #         np.angle(field).astype(np.float32), d=5, sigmaColor=150, sigmaSpace=75
        #     )
        # )

        field_copy = field.copy()
        field = ndimage.gaussian_filter(field, filtering_radius)

        extent = np.array(
            [
                -field.shape[-1] * d_real / (2 * xi),
                field.shape[-1] * d_real / (2 * xi),
                -field.shape[-2] * d_real / (2 * xi),
                field.shape[-2] * d_real / (2 * xi),
            ]
        )

        (
            u_tot,
            _,
            _,
        ) = velocity.helmholtz_decomp(field, dx=d_real, plot=False)
        # u_tot = velocity.velocity(np.angle(field))

        field_zoom = field[
            field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
            field.shape[-1] // 2 + x0 - roi_xi // 2 : field.shape[-1] // 2
            + x0
            + roi_xi // 2,
        ]
        field_copy_zoom = field_copy[
            field_copy.shape[-2] // 2 - y0 - roi_xi : field_copy.shape[-2] // 2
            - y0
            + roi_xi,
            field_copy.shape[-1] // 2 + x0 - roi_xi // 2 : field_copy.shape[-1] // 2
            + x0
            + roi_xi // 2,
        ]
        u_tot_zoom_0 = u_tot[
            0,
            field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
            field.shape[-1] // 2 + x0 - roi_xi // 2 : field.shape[-1] // 2
            + x0
            + roi_xi // 2,
        ]
        u_tot_zoom_1 = u_tot[
            1,
            field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
            field.shape[-1] // 2 + x0 - roi_xi // 2 : field.shape[-1] // 2
            + x0
            + roi_xi // 2,
        ]

        flow_tot = np.hypot(u_tot_zoom_0, u_tot_zoom_1)
        flow_tot = flow_tot / np.nanmax(flow_tot)
        YY, XX = np.indices(flow_tot.shape)

        amp_zoom = np.abs(field_copy_zoom)
        amp_zoom = amp_zoom / np.nanmax(amp_zoom)
        phase_zoom = contrast.angle_fast(field_copy_zoom)

        # vortices = velocity.vortex_detection(phase_zoom, plot=True, r=7)
        sell = velocity.sell_detection(np.angle(field), plot=False, r=10)
        sell_array = np.zeros(field.shape[:2])
        for k in range(sell.shape[0]):
            sell_array[int(sell[k, 1]), int(sell[k, 0])] = sell[k, 2]
        sell_array[sell_array == 0] = np.nan
        sell_array_zoom = sell_array[
            field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
            field.shape[-1] // 2 + x0 - roi_xi // 2 : field.shape[-1] // 2
            + x0
            + roi_xi // 2,
        ]

    fig, ax = plt.subplots(1, 3, figsize=[12, 5])

    im0 = ax[0].imshow(
        amp_zoom,
        cmap="gray",
        interpolation="none",
    )
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=ax[0], orientation="horizontal", pad=0.15)
    cbar0.set_label(r"$\sqrt{\rho}$", labelpad=10)

    im1 = ax[1].imshow(
        phase_zoom,
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
    )
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=ax[1], orientation="horizontal", pad=0.15)
    cbar1.set_label(r"$\phi$ in rad", labelpad=10)

    im2 = ax[2].imshow(flow_tot, cmap="viridis", vmin=0, vmax=1)
    ax[2].streamplot(
        XX,
        YY,
        u_tot_zoom_0,
        u_tot_zoom_1,
        density=2,
        color="white",
        linewidth=0.5,
        zorder=2,  # S'assurer que le streamplot est en arrière-plan
    )
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=ax[2], orientation="horizontal", pad=0.15)
    cbar2.set_label(r"$|u^{tot}|$", labelpad=10)

    # Ajout des marqueurs en premier plan
    x_indices, y_indices = np.meshgrid(
        np.arange(sell_array_zoom.shape[1]), np.arange(sell_array_zoom.shape[0])
    )
    saddle_mask = sell_array_zoom.ravel() == 1  # Points où c = 1
    node_mask = sell_array_zoom.ravel() == -1  # Points où c = -1

    ax[2].scatter(
        x_indices.ravel()[saddle_mask],
        y_indices.ravel()[saddle_mask],
        color="tab:green",
        marker="D",
        label="Saddle",
        zorder=3,  # Mettre en premier plan
    )

    ax[2].scatter(
        x_indices.ravel()[node_mask],
        y_indices.ravel()[node_mask],
        color="tab:orange",
        marker="D",
        label="Node",
        zorder=3,  # Mettre en premier plan
    )

    # Ajouter une légende
    ax[2].legend(loc="upper left", fontsize=10)

    scale_factor = xi / d_real
    for a in ax:
        a.set_xlim(0, sell_array_zoom.shape[1])
        a.set_ylim(0, sell_array_zoom.shape[0])
        x_ticks = np.linspace(0, sell_array_zoom.shape[1], num=3)
        y_ticks = np.linspace(0, sell_array_zoom.shape[0], num=5)
        a.set_xticks(x_ticks)
        a.set_yticks(y_ticks)
        a.set_xticklabels([f"{tick / scale_factor:.0f}" for tick in a.get_xticks()])
        a.set_yticklabels([f"{tick / scale_factor:.0f}" for tick in a.get_yticks()])
        a.set_xlabel(r"$x/\xi$")
        a.set_ylabel(r"$y/\xi$")

    plt.savefig(f"{scan}/velocity_{i}.png", dpi=300)
    if plot:
        plt.show()


def vortex_jrs_analytic():
    Q1 = 1
    Q2 = 1
    Q3 = 1
    delta_r1 = 6.0
    delta_r2 = 15
    delta_r3 = 6.0
    x1 = 0
    x2 = 5
    x3 = 5
    Nbr = 7
    N = 100 * Nbr
    roi = 2 * Nbr
    line = 0
    # Generate grid (without shifting X and Y)
    x = np.linspace(-roi, roi, N)
    y = np.linspace(-roi, roi, N)
    X, Y = np.meshgrid(x, y)

    ampv1, phasev1 = Vortex(X, Y, x1, delta_r1 / 2, Q1, 1)
    ampv2, phasev2 = Vortex(X, Y, x1, -delta_r1 / 2, -Q1, 1)
    ampv3, phasev3 = Vortex(X - x2, Y, 0, delta_r2 / 2, Q2, 1)
    ampv4, phasev4 = Vortex(X - x2, Y, 0, -delta_r2 / 2, -Q2, 1)
    ampv5, phasev5 = Vortex(X - x3, Y, 0, delta_r3 / 2, -Q3, 1)
    ampv6, phasev6 = Vortex(X - x3, Y, 0, -delta_r3 / 2, Q3, 1)

    ampj, phasej = JRS_padé(X - 5, Y + 3)

    amptot = ampv1 + ampv2 + ampv3 + ampv5 + ampj
    phasetot = phasev1 + phasev2 + phasev3 + phasev5 + phasej
    psitot = amptot * np.exp(1j * (phasetot))
    amptot = np.abs(psitot)
    phasetot = np.angle(psitot)
    velotot = velocity.velocity(phasetot, dx=roi * 2 / N)

    vortices = velocity.vortex_detection(phasetot, plot=False, r=7)
    sell = velocity.sell_detection(phasetot, plot=False, r=7)

    scale_x = (x.max() - x.min()) / X.shape[1]
    scale_y = (y.max() - y.min()) / Y.shape[0]

    vortices[:, 0] = x.min() + vortices[:, 0] * scale_x
    vortices[:, 1] = y.min() + vortices[:, 1] * scale_y

    sell[:, 0] = x.min() + sell[:, 0] * scale_x
    sell[:, 1] = y.min() + sell[:, 1] * scale_y

    plt.figure()
    plt.imshow(
        amptot,
        extent=[x.min(), x.max(), y.max(), y.min()],
        cmap="gray",
        interpolation="none",
    )
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(
        phasetot,
        extent=[x.min(), x.max(), y.max(), y.min()],
        cmap="twilight_shifted",
        interpolation="none",
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.colorbar()
    plt.axhline(line)
    plt.streamplot(X, Y, velotot[0], velotot[1], color="white", linewidth=1, density=1)

    for i in range(sell.shape[0]):
        if sell[i, 2] == 1:
            plt.scatter(
                sell[i, 0],
                sell[i, 1],
                color="tab:orange",
                marker="D",
                label="sell",
                zorder=3,
            )
        if sell[i, 2] == -1:
            plt.scatter(
                sell[i, 0],
                sell[i, 1],
                color="tab:green",
                marker="D",
                label="node",
                zorder=3,
            )
    for j in range(vortices.shape[0]):
        if vortices[j, 2] == 1:
            plt.scatter(
                vortices[j, 0],
                vortices[j, 1],
                color="blue",
                marker="o",
                label="vortex",
                zorder=3,
            )
        if vortices[j, 2] == -1:
            plt.scatter(
                vortices[j, 0],
                vortices[j, 1],
                color="red",
                marker="o",
                label="antivortex",
                zorder=3,
            )

    # plt.legend(
    #     loc="upper left",
    # )
    plt.show()


def single_shot(plot):
    print("Loading data ...")
    xi = 40e-6
    scan = f"treatment/data_cache"
    im = Image.open(f"{scan}/dn.tiff")
    im_ref = Image.open(f"{scan}/dn_ref.tiff")
    fluid_ref = np.array(im_ref)
    h, w = (206, 206)
    # h, w = (512, 512)
    # h, w = (1024, 1024)
    filtering_radius = 2
    exponent = 4

    field_ref = contrast.im_osc_fast_t(
        fluid_ref,
        center=(fluid_ref.shape[-2] // 2, fluid_ref.shape[-1] // 4),
        radius=fluid_ref.shape[-2] // 8,
    )

    x0 = 0
    y0 = 0
    roi = 15
    roi_xi = int(roi * (xi / d_real))

    fluid = np.array(im)
    field = contrast.im_osc_fast_t(
        fluid,
        center=(fluid.shape[-2] // 2, fluid.shape[-1] // 4),
        radius=fluid.shape[-2] // 8,
    )
    contrast.exp_angle_fast(field, field_ref)

    field_copy = field.copy()
    field = ndimage.gaussian_filter(field, filtering_radius)

    extent = np.array(
        [
            -field.shape[-1] * d_real / (2 * xi),
            field.shape[-1] * d_real / (2 * xi),
            -field.shape[-2] * d_real / (2 * xi),
            field.shape[-2] * d_real / (2 * xi),
        ]
    )

    (
        u_tot,
        _,
        _,
    ) = velocity.helmholtz_decomp_phase(field, dx=d_real, plot=False)
    # u_tot = velocity.velocity(np.angle(field))

    field_zoom = field[
        field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
        field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
    ]
    field_copy_zoom = field_copy[
        field_copy.shape[-2] // 2 - y0 - roi_xi : field_copy.shape[-2] // 2
        - y0
        + roi_xi,
        field_copy.shape[-1] // 2 + x0 - roi_xi : field_copy.shape[-1] // 2
        + x0
        + roi_xi,
    ]
    u_tot_zoom_0 = u_tot[
        0,
        field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
        field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
    ]
    u_tot_zoom_1 = u_tot[
        1,
        field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
        field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
    ]

    flow_tot = np.hypot(u_tot_zoom_0, u_tot_zoom_1)
    flow_tot = flow_tot / np.nanmax(flow_tot)
    YY, XX = np.indices(flow_tot.shape)

    amp_zoom = np.abs(field_copy_zoom)
    amp_zoom = amp_zoom / np.nanmax(amp_zoom)
    phase_zoom = contrast.angle_fast(field_copy_zoom)

    # vortices = velocity.vortex_detection(phase_zoom, plot=True, r=7)
    sell = velocity.sell_detection(np.angle(field), plot=False, r=10)
    sell_array = np.zeros(field.shape[:2])
    for k in range(sell.shape[0]):
        sell_array[int(sell[k, 1]), int(sell[k, 0])] = sell[k, 2]
    sell_array[sell_array == 0] = np.nan
    sell_array_zoom = sell_array[
        field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
        field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
    ]

    fig, ax = plt.subplots(1, 3, figsize=[12, 5])

    im0 = ax[0].imshow(
        amp_zoom,
        cmap="gray",
        interpolation="none",
    )
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=ax[0], orientation="horizontal", pad=0.15)
    cbar0.set_label(r"$\sqrt{\rho}$", labelpad=10)

    im1 = ax[1].imshow(
        phase_zoom,
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
    )
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=ax[1], orientation="horizontal", pad=0.15)
    cbar1.set_label(r"$\phi$ in rad", labelpad=10)

    im2 = ax[2].imshow(flow_tot, cmap="viridis", vmin=0, vmax=1)
    ax[2].streamplot(
        XX,
        YY,
        u_tot_zoom_0,
        u_tot_zoom_1,
        density=2,
        color="white",
        linewidth=0.5,
        zorder=2,  # S'assurer que le streamplot est en arrière-plan
    )
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=ax[2], orientation="horizontal", pad=0.15)
    cbar2.set_label(r"$|u^{tot}|$", labelpad=10)

    # Ajout des marqueurs en premier plan
    x_indices, y_indices = np.meshgrid(
        np.arange(sell_array_zoom.shape[1]), np.arange(sell_array_zoom.shape[0])
    )
    saddle_mask = sell_array_zoom.ravel() == 1  # Points où c = 1
    node_mask = sell_array_zoom.ravel() == -1  # Points où c = -1

    ax[2].scatter(
        x_indices.ravel()[saddle_mask],
        y_indices.ravel()[saddle_mask],
        color="tab:green",
        marker="D",
        label="Saddle",
        zorder=3,  # Mettre en premier plan
    )

    ax[2].scatter(
        x_indices.ravel()[node_mask],
        y_indices.ravel()[node_mask],
        color="tab:orange",
        marker="D",
        label="Node",
        zorder=3,  # Mettre en premier plan
    )

    # Ajouter une légende
    ax[2].legend(loc="upper left", fontsize=10)

    # scale_factor = xi / d_real
    # for a in ax:
    #     a.set_xlim(0, sell_array_zoom.shape[1])
    #     a.set_ylim(0, sell_array_zoom.shape[0])
    #     x_ticks = np.linspace(0, sell_array_zoom.shape[1], num=3)
    #     y_ticks = np.linspace(0, sell_array_zoom.shape[0], num=5)
    #     a.set_xticks(x_ticks)
    #     a.set_yticks(y_ticks)
    #     a.set_xticklabels([f"{tick / scale_factor:.0f}" for tick in a.get_xticks()])
    #     a.set_yticklabels([f"{tick / scale_factor:.0f}" for tick in a.get_yticks()])
    #     a.set_xlabel(r"$x/\xi$")
    #     a.set_ylabel(r"$y/\xi$")

    if plot:
        plt.show()


def velocity_x(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 7.5
    x0 = 0
    y0 = 10
    rad = 3
    line = -10
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    v_x_t = []
    v_x_i = []
    v_x_c = []
    phase_x_t = []
    rho_x_t = []

    # which = np.arange(16, 40, 2)
    which = np.array([16, 23, 28, 39])
    # which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(which):
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            amp_tot = np.abs(field)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field)

            extent = np.array(
                [
                    -field.shape[-1] * d_real / (2 * xi),
                    field.shape[-1] * d_real / (2 * xi),
                    -field.shape[-2] * d_real / (2 * xi),
                    field.shape[-2] * d_real / (2 * xi),
                ]
            )

            velo = velocity.velocity(np.angle(field), dx=d_real / xi)
            # plt.figure()
            # plt.imshow(np.hypot(velo[0], velo[1]), cmap="gray")
            # plt.axhline(velo[0].shape[0] // 2 + line)
            # plt.colorbar()
            # plt.show()

            # plt.figure()
            # plt.plot(velo[0][velo[0].shape[0] // 2 + line, :], label="Total")
            # plt.show()

            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi, plot=False
            )

            field_zoom = field[
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_tot_zoom_0 = u_tot[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_inc_zoom_0 = u_inc[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_comp_zoom_0 = u_comp[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            # flow_comp = np.hypot(u_inc_zoom_0, u_inc_zoom_1)
            # flow_comp = flow_comp / np.nanmax(flow_comp)
            # YY, XX = np.indices(flow_comp.shape)
            # plt.figure()
            # plt.imshow(
            #     flow_comp,
            #     cmap="viridis",
            #     vmin=0,
            #     vmax=1,
            # )
            # plt.streamplot(
            #     XX,
            #     YY,
            #     u_comp_zoom_0,
            #     u_comp_zoom_1,
            #     color="white",
            #     linewidth=0.5,
            #     density=2,
            # )
            # plt.show()

            v_x_t_j = u_tot_zoom_0[
                u_tot_zoom_0.shape[0] // 2 - line : u_tot_zoom_0.shape[0] // 2 + line, :
            ]
            v_x_i_j = u_inc_zoom_0[
                u_tot_zoom_0.shape[0] // 2 - line : u_tot_zoom_0.shape[0] // 2 + line, :
            ]
            v_x_c_j = u_comp_zoom_0[
                u_tot_zoom_0.shape[0] // 2 - line : u_tot_zoom_0.shape[0] // 2 + line, :
            ]

            v_x_t += [np.mean(v_x_t_j, axis=0)]
            v_x_i += [np.mean(v_x_i_j, axis=0)]
            v_x_c += [np.mean(v_x_c_j, axis=0)]

            amp_zoom = np.abs(field_zoom)
            amp_zoom = amp_zoom / np.max(amp_zoom)
            phase_zoom = contrast.angle_fast(field_zoom)
            # vortices = velocity.vortex_detection(phase_zoom, plot=True, r=7)

            rho_xx = amp_zoom[
                amp_zoom.shape[0] // 2 - line : amp_zoom.shape[0] // 2 + line, :
            ]
            rho_x_t += [np.mean(rho_xx, axis=0)]
            phase_x = phase_zoom[phase_zoom.shape[0] // 2 + line, :]
            phase_x_t += [phase_x]

            # plt.figure()
            # plt.plot(np.unwrap(phase_x))
            # plt.title(f"Image: {i}")
            # plt.show()

            plt.figure()
            plt.title(f"Image: {i}")
            plt.imshow(
                amp_zoom,
                cmap="gray",
                interpolation="none",
            )
            plt.colorbar()
            plt.axhline(amp_zoom.shape[0] // 2 + line)
            plt.axhline(amp_zoom.shape[0] // 2 - line)
            plt.show()

            fig, ax = plt.subplots(1, 2)
            im0 = ax[0].imshow(
                amp_zoom,
                cmap="gray",
                interpolation="none",
            )
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            cbar0 = fig.colorbar(im0, ax=ax[0], orientation="horizontal", pad=0.15)
            cbar0.set_label(r"$\sqrt{\rho}$", labelpad=10)
            im1 = ax[1].imshow(
                phase_zoom,
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="none",
            )
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("y")
            cbar1 = fig.colorbar(im1, ax=ax[1], orientation="horizontal", pad=0.15)
            cbar1.set_label(r"$\phi$ in rad", labelpad=10)
            plt.suptitle(f"t={taus[i]}")
            plt.tight_layout()
            plt.show()

    v_x_t = np.array(v_x_t)
    v_x_i = np.array(v_x_i)
    v_x_c = np.array(v_x_c)
    rho_x_t = np.array(rho_x_t)
    phase_x_t = np.array(phase_x_t)

    which2 = which
    which3 = np.array([5, 15])
    c_max = 0.9
    cmap_base = plt.get_cmap("viridis")
    colors = cmap_base(np.linspace(0, c_max, len(which3)))  # Seulement 4 couleurs

    fig, ax = plt.subplots(1, 3, figsize=[12, 5])

    for j, i in enumerate(np.flip(which2)):
        if i not in which3:
            continue
        color = colors[np.where(which3 == i)[0][0]]  # Index local dans which3
        x_i = np.linspace(-roi, roi, len(v_x_t[i]))

        ax[1].plot(
            x_i,
            v_x_t[i],
            label=rf"$\tau$={taus[i]:.0f}",
            color=color,
            linewidth=3,
        )
        ax[2].plot(
            x_i,
            phase_x_t[i],
            color=color,
            linewidth=3,
        )
        ax[0].plot(
            x_i,
            rho_x_t[i],
            color=color,
            linewidth=3,
        )

    ax[0].set_xlabel(r"$x/\xi$")
    ax[0].set_ylabel(r"$\sqrt{\rho_x}$")
    ax[0].set_title("Amplitude")
    ax[1].set_xlabel(r"$x/\xi$")
    ax[1].set_ylabel(r"$u_x^{tot} / c_s$")
    ax[1].set_title("Total velocity")
    ax[2].set_xlabel(r"$x/\xi$")
    ax[2].set_ylabel(r"$\phi$ in rad")
    ax[2].set_title("Phase")
    ax[0].legend()
    for a in ax:
        a.set_xlim(-roi, roi)
        a.set_ylim(-0.75, 1.5)
        a.axhline(0, color="black", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def point_vortex():
    # v_bgd = np.load(f"{scan}/v_x_t.npy", allow_pickle=True)
    # x_bgd = np.load(f"{scan}/x_roi.npy", allow_pickle=True)

    xi = 13e-6
    d_real = 3.45e-6 * 2
    size = xi / d_real
    t_max = 30
    dt = 0.2
    n_steps = int(t_max / dt)

    x_roi = np.arange(0, 20.25, 0.25)
    x_roi = x_roi / np.max(x_roi)
    v_fit = x_roi / (1 + x_roi)
    v_fit /= np.max(v_fit)
    # v_fit *= 0.0
    v_fit *= 0.4
    # v_fit *= 0.6
    # v_fit *= 1.9
    # v_fit *= xi
    x_roi = np.arange(0, 20.25, 0.25)

    Q = np.array([1, 1, -1, -1])
    d = 1.5 * size
    D = 8 * size
    positions = np.array(
        [
            [0.0, D / 2 + d / 2],
            [0.0, D / 2 - d / 2],
            [0.0, -D / 2 - d / 2],
            [0.0, -D / 2 + d / 2],
        ]
    )

    history = []

    def outward_map(pos, step):
        r = np.linalg.norm(pos)
        r = round(r * 2) / 2
        return v_fit[np.where(x_roi == r)] * xi

    def velocity_field(pos, positions, charges, step):
        """Vitesse = champ de phase des vortex + flux radial"""
        v_vortex = np.zeros(2)
        for j, rj in enumerate(positions):
            if np.allclose(pos, rj):
                continue
            diff = pos - rj
            r2 = np.dot(diff, diff)
            if r2 == 0:
                continue
            vj = (charges[j] / r2) * np.array([-diff[1], diff[0]])
            v_vortex += vj

        v_total = (size) * v_vortex

        # norm_pos = np.linalg.norm(pos)
        # if norm_pos != 0:
        #     radial_dir = pos / norm_pos
        #     v_total += outward_map(pos, step) * radial_dir

        return v_total

    for step in tqdm.tqdm(range(n_steps)):
        new_positions = np.copy(positions)
        for i in range(len(positions)):
            v = velocity_field(positions[i], positions, Q, step)
            new_positions[i] += v * dt
        history.append(np.copy(positions))
        positions = new_positions

    # Affichage
    history = np.array(history) / xi
    vortex = ["+", "+", "-", "-"]
    color = ["lightsteelblue", "tab:blue", "darksalmon", "tab:red"]

    plt.figure(figsize=(5, 3))
    for i in range(len(positions)):
        plt.plot(history[:, i, 0], history[:, i, 1], "-", color=color[i])
        plt.scatter(
            history[:, i, 0],
            history[:, i, 1],
            facecolor="none",
            edgecolors=color[i],
            label=f"{vortex[i]}",
        )
        plt.scatter(
            history[0, i, 0],
            history[0, i, 1],
            facecolor=color[i],
            edgecolors=color[i],
            zorder=3,
        )
    # plt.ylim(0, 12)
    # plt.xlim(-2.5, 17.5)
    plt.title("Trajectoires des vortex (unités ξ)")
    plt.xlabel("x/ξ")
    plt.ylabel("y/ξ")
    plt.grid(True)
    plt.legend()
    plt.show()

    # def plot_velocity_along_x(history, Q, x_window=(-2, 20), n_points=200):
    #     import matplotlib.cm as cm
    #     from matplotlib.colors import Normalize

    #     hbar = 1.0
    #     m = 1.0
    #     x_vals = np.linspace(x_window[0], x_window[1], n_points)
    #     y_fixed = 0.0
    #     velocities = []

    #     def total_velocity_at_point(x, vortex_positions, charges):
    #         v = np.zeros(2)
    #         point = np.array([x, y_fixed])
    #         for j, rj in enumerate(vortex_positions):
    #             diff = point - rj
    #             r2 = np.dot(diff, diff)
    #             if r2 == 0:
    #                 continue
    #             vj = (charges[j] / r2) * np.array([-(diff[1]), diff[0]])
    #             v += vj
    #         return (hbar / m) * v

    #     for snapshot in history:
    #         vx = np.array([total_velocity_at_point(x, snapshot, Q)[0] for x in x_vals])
    #         velocities.append(vx)

    #     velocities = np.array(velocities)

    #     # Tracé
    #     fig, ax = plt.subplots()
    #     norm = Normalize(vmin=0, vmax=len(velocities))
    #     cmap = cm.get_cmap("inferno")

    #     for t, vx in enumerate(velocities):
    #         ax.plot(x_vals, vx, color=cmap(norm(t)), alpha=0.8)

    #     ax.set_title("Composante vx sur y = 0 au cours du temps")
    #     ax.set_xlabel(r"$x/\xi$")
    #     ax.set_ylabel(r"$v_x^{tot}\rm{\ (a.u.)}$")
    #     ax.grid(True)
    #     plt.show()

    # plot_velocity_along_x(history=history, Q=Q, x_window=x_window)

    # fig, ax = plt.subplots()
    # lines = [ax.plot([], [], "o")[0] for _ in range(len(positions))]
    # colors = ["r", "r", "b", "b"]

    # def init():
    #     ax.set_xlim(x_window[0], x_window[1])
    #     ax.set_ylim(-20, 20)
    #     return lines

    # def update(frame):
    #     for i, line in enumerate(lines):
    #         line.set_data(history[frame, i, 0], history[frame, i, 1])
    #         line.set_color(colors[i])
    #     return lines

    # ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True)
    # plt.title("Point vortex model")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.grid(True)
    # ani.save("vortex_animation.mp4", writer="ffmpeg", fps=20)
    # plt.show()


def point_vortex_BEC():
    # Unités naturelles
    dt = 0.1  # Unité de z_nl
    t_max = 57  # Temps total en unités de z_nl (ξ final)

    n_steps = int(t_max / dt)
    print("\n", dt, t_max, n_steps, "\n")

    # Profil de vitesse radiale en unités de c_s
    x_roi = np.arange(0, 20.25, 0.25)  # Rayon en unités de ξ
    x_roi /= np.max(x_roi)
    v_fit = x_roi / (1 + x_roi)
    v_fit /= np.max(v_fit)
    v_fit *= 0.35  # Outward flow (c_s)
    x_roi = np.arange(0, 20.25, 0.25)  # Rechargé pour la correspondance avec r arrondi

    # plt.figure()
    # plt.plot(x_roi, v_fit, label="Flux radial")
    # plt.ylim(-0.05, 0.5)
    # plt.show()

    # Configuration initiale des vortex (unités ξ)
    Q = np.array([1, 1, -1, -1])  # Charges topologiques
    d = 1.5  # Distance entre paires (unités ξ)
    D = 5  # Séparation entre paires (unités ξ)

    positions = np.array(
        [
            [0.0, D / 2 + d / 2],
            [0.0, D / 2 - d / 2],
            [0.0, -D / 2 - d / 2],
            [0.0, -D / 2 + d / 2],
        ]
    )  # Toutes les coordonnées sont en unités de ξ

    history = []

    def outward_map(pos, step):
        """Renvoie le flux radial (en unités de c_s) à une distance r (en ξ)."""
        r = np.linalg.norm(pos)
        r = round(r * 2) / 2
        try:
            return v_fit[np.where(x_roi == r)][0]
        except IndexError:
            return 0.0

    def velocity_field(pos, positions, charges, step):
        """Champ de vitesse total : vortex + flux radial (toutes unités en ξ, c_s)."""
        v_vortex = np.zeros(2)
        for j, rj in enumerate(positions):
            if np.allclose(pos, rj):
                continue
            diff = pos - rj
            r2 = np.dot(diff, diff)
            if r2 == 0:
                continue
            vj = (charges[j] / r2) * np.array([-diff[1], diff[0]])
            v_vortex += vj

        v_total = v_vortex

        # Ajout du flux radial optionnel (commenté ici)
        norm_pos = np.linalg.norm(pos)
        if norm_pos != 0:
            radial_dir = pos / norm_pos
            v_total += outward_map(pos, step) * radial_dir

        return v_total

    for step in tqdm.tqdm(range(n_steps)):
        new_positions = np.copy(positions)
        for i in range(len(positions)):
            v = velocity_field(positions[i], positions, Q, step)  # En unités de c_s
            new_positions[i] += v * dt  # Position en ξ
        history.append(np.copy(positions))
        positions = new_positions

    # Affichage (positions déjà en ξ)
    history = np.array(history)
    vortex = ["+", "+", "-", "-"]
    color = ["mediumaquamarine", "tab:blue", "darksalmon", "tab:red"]

    plt.figure(figsize=(5, 3))
    for i in range(len(positions)):
        plt.plot(history[:, i, 0], history[:, i, 1], "-", color=color[i])
        # plt.scatter(
        #     history[:, i, 0],
        #     history[:, i, 1],
        #     facecolor="none",
        #     edgecolors=color[i],
        #     label=f"{vortex[i]}",
        # )
        # plt.scatter(
        #     history[0, i, 0],
        #     history[0, i, 1],
        #     facecolor=color[i],
        #     edgecolors=color[i],
        #     zorder=3,
        # )
    plt.title("Trajectoires des vortex (unités ξ)")
    plt.xlabel("x/ξ")
    plt.ylabel("y/ξ")
    plt.xlim(-2.5, 15)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.legend()
    # plt.tight_layout()
    plt.show()


def velocity_decomp(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 7.5
    x0 = 30
    # x0 = 60
    y0 = 4
    rad = 3
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    v_x = []

    # for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
    # which = np.arange(18, 20, 1)
    # which = np.arange(0, 20, 2)
    # which = np.array([16, 23, 30, 39])
    # which = np.arange(23, 40, 1)
    which = np.array([16, 23, 28, 39])
    for i in which:
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field_raw = field.copy()
            field = ndimage.gaussian_filter(field, rad)

            extent = np.array(
                [
                    -field.shape[-1] * d_real / (2 * xi),
                    field.shape[-1] * d_real / (2 * xi),
                    -field.shape[-2] * d_real / (2 * xi),
                    field.shape[-2] * d_real / (2 * xi),
                ]
            )

            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(field, dx=d_real, plot=False)
            # u_tot = velocity.velocity(np.angle(field))

            field_zoom = field[
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            field_raw_zoom = field_raw[
                field_raw.shape[-2] // 2 - y0 - roi_xi : field_raw.shape[-2] // 2
                - y0
                + roi_xi,
                field_raw.shape[-1] // 2 + x0 - roi_xi : field_raw.shape[-1] // 2
                + x0
                + roi_xi,
            ]
            u_tot_zoom_0 = u_tot[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_tot_zoom_1 = u_tot[
                1,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_inc_zoom_0 = u_inc[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_inc_zoom_1 = u_inc[
                1,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_comp_zoom_0 = u_comp[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_comp_zoom_1 = u_comp[
                1,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            flow_tot = np.hypot(u_tot_zoom_0, u_tot_zoom_1)
            flow_tot = flow_tot / np.nanmax(flow_tot)
            flow_inc = np.hypot(u_inc_zoom_0, u_inc_zoom_1)
            flow_inc = flow_inc / np.nanmax(flow_inc)
            flow_comp = np.hypot(u_comp_zoom_0, u_comp_zoom_1)
            flow_comp = flow_comp / np.nanmax(flow_comp)
            YY, XX = np.indices(flow_tot.shape)

            amp_raw_zoom = np.abs(field_raw_zoom)
            amp_raw_zoom = amp_raw_zoom / np.nanmax(amp_raw_zoom)
            phase_zoom = contrast.angle_fast(field_zoom)
            phase_raw_zoom = contrast.angle_fast(field_raw_zoom)

            vortices = velocity.vortex_detection(np.angle(field), plot=False, r=10)
            vortices_array = np.zeros(field.shape[:2])
            for k in range(vortices.shape[0]):
                vortices_array[int(vortices[k, 1]), int(vortices[k, 0])] = vortices[
                    k, 2
                ]
            vortices_array[vortices_array == 0] = np.nan
            vortices_array_zoom = vortices_array[
                vortices_array.shape[-2] // 2 - y0 - roi_xi : vortices_array.shape[-2]
                // 2
                - y0
                + roi_xi,
                vortices_array.shape[-1] // 2 + x0 - roi_xi : vortices_array.shape[-1]
                // 2
                + x0
                + roi_xi,
            ]

            sell = velocity.sell_detection(np.angle(field), plot=False, r=2)
            sell_array = np.zeros(field.shape[:2])
            for k in range(sell.shape[0]):
                sell_array[int(sell[k, 1]), int(sell[k, 0])] = sell[k, 2]
            sell_array[sell_array == 0] = np.nan
            sell_array_zoom = sell_array[
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            amp_raw_zoom_x = amp_raw_zoom[amp_raw_zoom.shape[0] // 2, :]
            plt.figure()
            plt.imshow(amp_raw_zoom, cmap="gray", interpolation="none")
            plt.show()
            plt.figure()
            plt.plot(amp_raw_zoom_x)
            plt.ylim(0, 1)
            plt.xlim(0, amp_raw_zoom.shape[1])
            plt.show()

            fig, ax = plt.subplots(figsize=[6, 5])  # Un seul axe
            fig.suptitle(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xi * 1e6:.0f} $\mu m$")

            # Affichage de la phase avec fond colormappé
            im0 = ax.imshow(
                phase_raw_zoom,
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="none",
            )

            # Ajout du champ de vecteurs
            ax.streamplot(
                XX,
                YY,
                u_tot_zoom_0,
                u_tot_zoom_1,
                density=1,
                color="white",
                linewidth=1,
            )

            ax.set_title(r"$u^{tot}$")

            # Grille d’indices
            x_indices, y_indices = np.meshgrid(
                np.arange(sell_array_zoom.shape[1]), np.arange(sell_array_zoom.shape[0])
            )

            # Points selle et nœud
            saddle_mask = sell_array_zoom.ravel() == 1
            node_mask = sell_array_zoom.ravel() == -1

            ax.scatter(
                x_indices.ravel()[saddle_mask],
                y_indices.ravel()[saddle_mask],
                color="tab:orange",
                marker="D",
                label="Saddle",
                zorder=3,
            )
            ax.scatter(
                x_indices.ravel()[node_mask],
                y_indices.ravel()[node_mask],
                color="black",
                marker="D",
                label="Node",
                zorder=3,
            )

            # Vortex+
            x_indices, y_indices = np.meshgrid(
                np.arange(vortices_array_zoom.shape[1]),
                np.arange(vortices_array_zoom.shape[0]),
            )
            vortices_mask_plus = vortices_array_zoom.ravel() == 1
            vortices_mask_minus = vortices_array_zoom.ravel() == -1

            ax.scatter(
                x_indices.ravel()[vortices_mask_plus],
                y_indices.ravel()[vortices_mask_plus],
                color="blue",
                marker="o",
                label="Vortex+",
                zorder=3,
            )
            ax.scatter(
                x_indices.ravel()[vortices_mask_minus],
                y_indices.ravel()[vortices_mask_minus],
                color="red",
                marker="o",
                label="Vortex-",
                zorder=3,
            )

            # Légende
            ax.legend(loc="upper right", fontsize=10)

            # Mise à l’échelle en unités ξ
            scale_factor = xi / d_real
            ax.set_xlim(0, sell_array_zoom.shape[1])
            ax.set_ylim(0, sell_array_zoom.shape[0])

            x_ticks = np.linspace(0, sell_array_zoom.shape[1], num=3)
            y_ticks = np.linspace(0, sell_array_zoom.shape[0], num=5)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([f"{tick / scale_factor:.0f}" for tick in x_ticks])
            ax.set_yticklabels([f"{tick / scale_factor:.0f}" for tick in y_ticks])

            ax.set_xlabel(r"$x/\xi$")
            ax.set_ylabel(r"$y/\xi$")

            # Sauvegarde et affichage
            plt.savefig(f"{scan}/velocity_{i}.png", dpi=300)
            if plot:
                plt.show()


def rho_x(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    delta_Rs = np.flip(np.linspace(25, 50, 20))
    xis = np.load(f"{scan}/xis.npy")
    roi = 40
    x0 = 0
    y0 = -10
    lines = 10
    rad = 5
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    phase_x_t = []
    rho_x_t = []
    velo_x_t = []
    velo_x_i = []
    t_i = []

    # for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
    # which = np.arange(2, 13, 1)
    # which = np.arange(10, 30, 2)
    # which = np.array([2, 5, 8, 12])
    which = np.array([18])
    # which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(np.flip(which)):
        xi = xis[i]
        t_i += [taus[i]]
        delta_R = int(delta_Rs[i])
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([1]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            field_ref = ndimage.gaussian_filter(field_ref, rad)

            extent = np.array(
                [
                    -field.shape[-1] * d_real / (2 * xi),
                    field.shape[-1] * d_real / (2 * xi),
                    -field.shape[-2] * d_real / (2 * xi),
                    field.shape[-2] * d_real / (2 * xi),
                ]
            )

            field_zoom = field[
                field.shape[-2] // 2 - roi_xi : field.shape[-2] // 2 + roi_xi,
                field.shape[-1] // 2 - roi_xi : field.shape[-1] // 2 + roi_xi,
            ]
            field_ref_zoom = field_ref[
                field.shape[-2] // 2 - roi_xi : field.shape[-2] // 2 + roi_xi,
                field.shape[-1] // 2 - roi_xi : field.shape[-1] // 2 + roi_xi,
            ]

            amp_zoom = np.abs(field_zoom)
            amp_zoom = amp_zoom / np.max(amp_zoom)
            amp_zoom_ref = np.abs(field_ref_zoom)
            amp_zoom_ref = amp_zoom_ref / np.max(amp_zoom_ref)
            # amp_zoom = amp_zoom - amp_zoom_ref
            phase_zoom = contrast.angle_fast(field_zoom)
            rho_x = amp_zoom[
                amp_zoom.shape[0] // 2 + y0 - lines : amp_zoom.shape[0] // 2
                + y0
                + lines,
                :,
            ]
            rho_x = np.mean(rho_x, axis=0)
            phase_x = phase_zoom[
                phase_zoom.shape[0] // 2 + y0 - lines : phase_zoom.shape[0] // 2
                + y0
                + lines,
                :,
            ]
            phase_x = np.mean(phase_x, axis=0)
            phase_x = phase_zoom[
                phase_zoom.shape[0] // 2 + y0,
                :,
            ]
            rho_x_t += [rho_x]
            phase_x_t += [phase_x]
            velocities = velocity.helmholtz_decomp(
                rho_x * np.exp(1j * phase_zoom), dx=d_real / xi
            )
            velo_tot = velocities[0]
            velo_x_tot = velo_tot[0][phase_zoom.shape[0] // 2 + y0, :]
            velo_x_t += [velo_x_tot]

            velo_inc = velocities[1]
            velo_x_inc = velo_inc[0][phase_zoom.shape[0] // 2 + y0, :]
            velo_x_i += [velo_x_inc]

            fig, ax = plt.subplots(1, 2)
            im0 = ax[0].imshow(
                amp_zoom,
                cmap="gray",
                interpolation="none",
            )
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            cbar0 = fig.colorbar(im0, ax=ax[0], orientation="horizontal", pad=0.15)
            cbar0.set_label(r"$\sqrt{\rho}$", labelpad=10)
            im1 = ax[1].imshow(
                phase_zoom,
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="none",
            )
            ax[0].axhline(
                phase_zoom.shape[0] // 2 + y0 - lines,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            ax[0].axhline(
                phase_zoom.shape[0] // 2 + y0 + lines,
                color="black",
                linestyle="--",
                linewidth=1,
            )

            ax[1].set_xlabel("x")
            ax[1].set_ylabel("y")
            cbar1 = fig.colorbar(im1, ax=ax[1], orientation="horizontal", pad=0.15)
            cbar1.set_label(r"$\phi$ in rad", labelpad=10)
            plt.suptitle(rf"$\Delta R_0 / \xi=${delta_R}")
            plt.tight_layout()
            plt.show()

            # u_tot = velocity.velocity(phase_zoom, dx=d_real / xi)
            # u_tot = np.hypot(u_tot[0], u_tot[1])
            # c_s = np.abs(field_zoom) ** 2
            # c_s /= np.max(c_s)
            # Rip = u_tot / 2 + np.sqrt(c_s)
            # Rim = u_tot / 2 - np.sqrt(c_s)

            # fig, ax = plt.subplots(1, 2)
            # im0 = ax[0].imshow(
            #     Rip,
            #     cmap="gray",
            #     interpolation="none",
            # )
            # im1 = ax[1].imshow(
            #     Rim,
            #     cmap="gray",
            #     interpolation="none",
            # )
            # plt.show()

    fig, ax = plt.subplots(1, 4, figsize=[12, 5])
    rho_x_t = np.array(rho_x_t)
    phase_x_t = np.array(phase_x_t)
    velo_x_t = np.array(velo_x_t)
    velo_x_i = np.array(velo_x_i)
    t_i = np.array(t_i)
    Rim = velo_x_t / 2 - rho_x_t
    Rip = velo_x_t / 2 + rho_x_t
    colors = plt.cm.inferno(np.flip(np.linspace(0, 0.9, len(which))))
    for i in range(len(which)):
        x_i = np.linspace(-roi, roi, len(rho_x_t[i]))
        ax[0].plot(
            x_i,
            rho_x_t[i],
            # label=rf"$\Delta R_0/\xi=${delta_Rs[which[i]]:.0f}",
            label=rf"$\tau=${t_i[i]:.0f}",
            color=colors[i],
        )
        ax[1].plot(
            x_i,
            velo_x_t[i],
            color=colors[i],
        )
        ax[2].plot(
            x_i,
            velo_x_i[i],
            color=colors[i],
        )
        ax[3].plot(
            x_i,
            Rip[i],
            label=r"$\lambda_{+}$",
            color=colors[i],
        )
        ax[3].plot(
            x_i,
            Rim[i],
            label=r"$\lambda_{-}$",
            color=colors[i],
        )
    ax[0].set_ylabel(r"$\sqrt{\rho_x}$")
    ax[1].set_ylabel(r"$u_x^{tot} / c_s$")
    ax[2].set_ylabel(r"$u_x^{inc} / c_s$")
    ax[3].set_ylabel(r"$\lambda_{-}$")
    ax[0].set_title("Amplitude")
    ax[1].set_title("Total velocity")
    ax[2].set_title("Inc velocity")
    ax[3].set_title("Riemann invariant")
    ax[0].legend()
    for j in range(len(ax)):
        ax[j].set_xlim(-roi, roi)
        ax[j].set_ylim(-1, 1.5)
        ax[j].axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax[j].set_xlabel(r"$x/\xi$")
    ax[0].set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()


def vortex_tracking_corrected(scan, windows_size, plot):
    t0 = time.perf_counter()
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    fields = np.flip(fields, axis=1)
    fields_ref = np.flip(fields_ref, axis=1)
    taus = np.load(f"{scan}/taus.npy")
    # dtaus = np.abs(np.diff(taus))
    xis = np.load(f"{scan}/xis.npy")
    # xis = np.sqrt(L / (k0 * taus))
    dtaus = np.diff(L / (k0 * xis**2))
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")

    # Stocker les positions pour chaque vortex
    num_vortices_max = 4
    vortex_trajectories = [[] for _ in range(num_vortices_max)]

    init_pos = np.array(
        [
            [0, 2.5 - 1.5 / 2],
            [0.0, 2.5 + 1.5 / 2],
            [0.0, -2.5 - 1.5 / 2],
            [0.0, -2.5 + 1.5 / 2],
        ]
    )

    previous_vortices = None  # Positions des vortex de l'étape précédente
    rad = 2
    all_vortices = []

    print("\n", "fields.shape: ", fields.shape, "\n")

    which = np.array([0, 1, 3, 9, 10, 11, 13, 19, 20, 21, 23, 29, 30, 31, 33, 39])

    # rhos_zoom = []
    good_index = []
    # velo_bgd = []
    for i in tqdm.tqdm(range(fields.shape[0] - 3), position=1):
        # for i in np.arange(10):
        # for i in tqdm.tqdm(range(10)):
        vortices_append = []
        for j in range(fields.shape[1]):
            # for j in np.array([0]):
            field = fields[i, j, :, :]
            rho_raw = np.abs(field)
            field_ref = fields_ref[i, j, :, :]
            field_ref_copy = field_ref.copy()

            # Compute the phase and density
            contrast.exp_angle_fast(field, field_ref)
            phase_raw = contrast.angle_fast(field)
            field = ndimage.gaussian_filter(field, rad)
            phi_vort_flat = contrast.angle_fast(field)
            rho = field.real * field.real + field.imag * field.imag
            rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            rho = np.sqrt(rho)
            rho = rho / np.nanmax(rho)
            rho_ref = np.sqrt(rho_ref)
            rho_ref = rho_ref / np.nanmax(rho_ref)

            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi) / 1.5
            window = int(radius_mask_ref)
            x, y = np.meshgrid(
                np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2])
            )
            mask_fluid = np.abs(
                (x - mask_ref.shape[-1] // 2) ** 2 + (y - mask_ref.shape[-2] // 2) ** 2
                < radius_mask_ref**2
            )
            mask_fluid = np.logical_not(mask_fluid)
            rho_masked = rho
            rho_masked[mask_fluid] = np.nan

            vort = velocity.vortex_detection(phi_vort_flat, plot=False, r=2)
            vort[:, 0] -= phi_vort_flat.shape[-1] // 2
            vort[:, 1] -= phi_vort_flat.shape[-2] // 2
            vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > window**2)
            vortices = vort[vort_select, :]

            xi = xis[i] * (1 / rho_ref)
            # plt.figure()
            # plt.imshow(rho_ref**2)
            # plt.colorbar()
            # plt.scatter(
            #     vortices[:, 0] + phi_vort_flat.shape[-1] // 2,
            #     vortices[:, 1] + phi_vort_flat.shape[-2] // 2,
            #     color="red",
            #     marker="o",
            # )
            # plt.show()
            xi = xi[
                np.mean(vortices[:, 1]).astype(int) + phi_vort_flat.shape[-1] // 2,
                np.mean(vortices[:, 0]).astype(int) + phi_vort_flat.shape[-2] // 2,
            ]

            # xi = xis[i]

            w = windows_size * xi / d_real
            w = int(w)
            # if i in which:
            #     if j == 0:
            #         x0 = int(2.5 * xis[i] / d_real)
            #         y0 = int(12 * xis[i] / d_real)
            #         x1 = int(17.5 * xis[i] / d_real)
            #         y1 = int(12 * xis[i] / d_real)
            #         roi2 = (
            #             slice(rho.shape[-2] // 2 - y0, rho.shape[-2] // 2 + y1),
            #             slice(rho.shape[-1] // 2 - x0, rho.shape[-1] // 2 + x1),
            #         )
            #         rho_raw = rho_raw / np.nanmax(rho_raw)
            #         rho_raw = rho_raw[roi2]
            #         phase_raw = phase_raw[roi2]
            #         extent = np.array(
            #             [
            #                 -rho_raw.shape[-1] * d_real / (2 * xis[i]),
            #                 rho_raw.shape[-1] * d_real / (2 * xis[i]),
            #                 -rho_raw.shape[-2] * d_real / (2 * xis[i]),
            #                 rho_raw.shape[-2] * d_real / (2 * xis[i]),
            #             ]
            #         )
            #         fig, ax = plt.subplots(1, 2, figsize=[12, 5])
            #         im0 = ax[0].imshow(
            #             rho_raw,
            #             cmap="gray",
            #             interpolation="none",
            #             extent=extent,
            #         )
            #         ax[0].set_xlabel("x")
            #         ax[0].set_ylabel("y")
            #         cbar0 = fig.colorbar(
            #             im0, ax=ax[0], orientation="horizontal", pad=0.15
            #         )
            #         cbar0.set_label(r"$\sqrt{\rho}$", labelpad=10)
            #         im1 = ax[1].imshow(
            #             phase_raw,
            #             cmap="twilight_shifted",
            #             vmin=-np.pi,
            #             vmax=np.pi,
            #             interpolation="none",
            #             extent=extent,
            #         )
            #         ax[1].set_xlabel("x")
            #         ax[1].set_ylabel("y")
            #         cbar1 = fig.colorbar(
            #             im1, ax=ax[1], orientation="horizontal", pad=0.15
            #         )
            #         cbar1.set_label(r"$\phi$ in rad", labelpad=10)
            #         plt.suptitle(
            #             rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i] * 1e6:.0f} $\mu m$"
            #         )
            #         plt.tight_layout()
            #         plt.savefig(f"{scan}/rho_raw_{i}_{j}.svg", dpi=300)
            #         # plt.show()

            vortices[:, 0] *= d_real / xi
            vortices[:, 1] *= d_real / xi
            if j == 0 and i != fields.shape[0] - 1:
                all_vortices += [vortices]
            if i == fields.shape[0] - 1 and j == 1:
                all_vortices += [vortices]

            # if j == 0:
            #     ## Compute the velocity field and compensate the outward flow
            #     field_ref_copy = ndimage.gaussian_filter(field_ref_copy, rad)
            #     amp_ref = np.abs(field_ref_copy)
            #     amp_ref = amp_ref / np.nanmax(amp_ref)
            #     phase_ref = contrast.angle_fast(field_ref_copy)
            #     _, _, u_comp_ref = velocity.helmholtz_decomp(
            #         amp_ref * np.exp(1j * phase_ref), dx=d_real / xis[i], plot=False
            #     )
            #     u_comp_ref_hypot = np.hypot(u_comp_ref[0], u_comp_ref[1])
            #     u_comp_ref_x = u_comp_ref[0]
            #     u_comp_ref_x = u_comp_ref_x[roi]
            #     vortt = vortices
            #     vortt[:, 0] += phi_vort_flat.shape[-1] // 2
            #     vortt[:, 1] += phi_vort_flat.shape[-2] // 2
            #     velo_bgd += [u_comp_ref_x[u_comp_ref_x.shape[0] // 2, :]]

            if len(vortices) != num_vortices_max:
                print(
                    f"'\n', Warning: Expected {num_vortices_max} vortices, but found {len(vortices)} at index {i}, {j} \n",
                )
                continue
            good_index += [i]

            if previous_vortices is not None:
                distances = np.linalg.norm(
                    vortices[:, None, :] - previous_vortices[None, :, :], axis=-1
                )
                from scipy.optimize import linear_sum_assignment

                row_ind, col_ind = linear_sum_assignment(distances)
                if i in [5]:
                    # print("\n", col_ind, vortices[col_ind], "\n")
                    col_ind_0 = col_ind[0].copy()
                    col_ind_1 = col_ind[1].copy()
                    col_ind_2 = [col_ind_1, col_ind_0, col_ind[2], col_ind[3]]
                    vortices = vortices[col_ind_2]
                else:
                    vortices = vortices[col_ind]
            vortices_append += [vortices]
        vortices_append = np.array(vortices_append)
        vortices = np.nanmean(vortices_append, axis=0)
        for v_idx in range(num_vortices_max):
            vortex_trajectories[v_idx].append(vortices[v_idx])
        previous_vortices = vortices.copy()

    # plt.figure()
    # velo_bgd = np.array(velo_bgd)
    # for i in range(velo_bgd.shape[0]):
    #     w = windows_size * xis[i] / d_real
    #     w = int(w)
    #     r = np.linspace(-w, w, 2 * w)
    #     plt.plot(r, velo_bgd[i], label=f"{taus[i]:.0f}")
    # plt.title("Velocity background")
    # plt.xlabel(r"r/$\xi$")
    # plt.ylabel(r"$u^{bgd}_x/c_s$")
    # plt.legend()
    # plt.show()

    # rhos_zoom_a = np.array(rhos_zoom)
    good_index = np.array(good_index)
    vortex_trajectories = [np.array(traj) for traj in vortex_trajectories]
    other_vortices = np.array(all_vortices)
    vortex = ["+", "+", "-", "-"]
    color = ["mediumaquamarine", "tab:blue", "tab:red", "darksalmon"]

    print("Plotting ...")
    plt.figure(figsize=(5, 3))
    plt.title(rf"Vortices trajectories")
    plt.grid()
    plt.xlabel(r"$x/\xi$")
    plt.ylabel(r"$y/\xi$")
    valid_trajectories = [traj for traj in vortex_trajectories]
    np.save(f"{scan}/vortex_trajectories.npy", valid_trajectories)
    correction = True
    for v_idx, trajectory in enumerate(valid_trajectories):
        trajectory = np.array(trajectory)
        if correction:
            for j in range(trajectory.shape[0]):
                # if j == 10 or j == 20 or j == 30 or j == len(trajectory) - 1:
                #     continue
                # trajectory[j, 0] = trajectory[j, 0] / (d_real / xis[j])
                # trajectory[j, 1] = trajectory[j, 1] / (d_real / xis[j])
                # trajectory[j, 0] += velo_out.shape[-1] // 2
                # trajectory[j, 1] += velo_out.shape[-2] // 2
                # x_futur = (
                #     velo_out[j, 0, 0, int(trajectory[j, 1]), int(trajectory[j, 0])]
                #     * dtaus[j]
                # )
                # y_futur = (
                #     velo_out[j, 0, 1, int(trajectory[j, 1]), int(trajectory[j, 0])]
                #     * dtaus[j]
                # )
                # trajectory[j, 0] -= velo_out.shape[-1] // 2
                # trajectory[j, 1] -= velo_out.shape[-2] // 2
                # trajectory[j, 0] = trajectory[j, 0] * (d_real / xis[j])
                # trajectory[j, 1] = trajectory[j, 1] * (d_real / xis[j])
                # trajectory[j + 1, 0] -= x_futur
                # trajectory[j + 1, 1] -= y_futur
                if j > 29:
                    trajectory[j, 0] -= trajectory[30, 0] - trajectory[30 - 1, 0]
                    trajectory[j, 1] -= trajectory[30, 1] - trajectory[30 - 1, 1]

        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "-",
            color=color[v_idx],
        )
        plt.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            facecolor="none",
            edgecolors=color[v_idx],
            label=f"{vortex[v_idx]}",
        )
        plt.scatter(
            init_pos[v_idx, 0],
            init_pos[v_idx, 1],
            facecolor=color[v_idx],
            edgecolors=color[v_idx],
            zorder=3,
        )
    plt.xlim(-2.5, 15)
    plt.ylim(0, 10)
    plt.title(rf"Vortex trajectories with correction={correction}")
    if plot:
        plt.show()

    return vortex_trajectories


def field_velocity_x(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 30
    x0 = 20
    y0 = 10
    rad = 3
    line = 0
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    v_x_t = []
    v_x_i = []
    v_x_c = []
    phase_x_t = []
    rho_x_t = []

    # which = np.arange(16, 40, 2)
    # which = np.array([5, 15])
    which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(which):
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            amp_tot = np.abs(field)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field)

            extent = np.array(
                [
                    -field.shape[-1] * d_real / (2 * xi),
                    field.shape[-1] * d_real / (2 * xi),
                    -field.shape[-2] * d_real / (2 * xi),
                    field.shape[-2] * d_real / (2 * xi),
                ]
            )

            # velo = velocity.velocity(np.angle(field), dx=d_real / xi)
            # plt.figure()
            # plt.imshow(np.hypot(velo[0], velo[1]), cmap="gray")
            # plt.axhline(velo[0].shape[0] // 2 + line)
            # plt.colorbar()
            # plt.show()

            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi, plot=False
            )

            field_zoom = field[
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_tot_zoom_0 = u_tot[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_inc_zoom_0 = u_inc[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_comp_zoom_0 = u_comp[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            # flow_comp = np.hypot(u_inc_zoom_0, u_inc_zoom_1)
            # flow_comp = flow_comp / np.nanmax(flow_comp)
            # YY, XX = np.indices(flow_comp.shape)
            # plt.figure()
            # plt.imshow(
            #     flow_comp,
            #     cmap="viridis",
            #     vmin=0,
            #     vmax=1,
            # )
            # plt.streamplot(
            #     XX,
            #     YY,
            #     u_comp_zoom_0,
            #     u_comp_zoom_1,
            #     color="white",
            #     linewidth=0.5,
            #     density=2,
            # )
            # plt.show()

            v_x_t_j = u_tot_zoom_0[u_tot_zoom_0.shape[0] // 2 + line, :]
            v_x_i_j = u_inc_zoom_0[u_tot_zoom_0.shape[0] // 2 + line, :]
            v_x_c_j = u_comp_zoom_0[u_tot_zoom_0.shape[0] // 2 + line, :]

            v_x_t += [v_x_t_j]
            v_x_i += [v_x_i_j]
            v_x_c += [v_x_c_j]

            amp_zoom = np.abs(field_zoom)
            amp_zoom = amp_zoom / np.max(amp_zoom)
            phase_zoom = contrast.angle_fast(field_zoom)

            rho_x_t += [amp_zoom[amp_zoom.shape[0] // 2 + line, :]]
            phase_x_t += [phase_zoom[phase_zoom.shape[0] // 2 + line, :]]

            h, w = amp_zoom.shape
            line_index = h // 2 + line
            roi_x = roi
            roi_y = roi
            x_extent = np.linspace(-roi_x, roi_x, w)
            y_extent = np.linspace(-roi_y, roi_y, h)
            profile = amp_zoom[line_index, :]
            profile_scaled = profile / np.max(profile) * (roi_y)
            y_profile_line = y_extent[line_index]
            y_profile = y_profile_line + profile_scaled
            plt.figure(figsize=[5, 5])
            plt.imshow(
                amp_zoom,
                cmap="gray",
                interpolation="none",
                extent=[-roi_x, roi_x, -roi_y, roi_y],
                origin="lower",
            )
            plt.axhline(y=y_profile_line, color="black", linestyle="--", linewidth=1)
            plt.plot(x_extent, y_profile - roi, color="tab:blue")
            plt.tight_layout()
            plt.show()

    v_x_t = np.array(v_x_t)
    v_x_i = np.array(v_x_i)
    v_x_c = np.array(v_x_c)
    rho_x_t = np.array(rho_x_t)
    phase_x_t = np.array(phase_x_t)

    which2 = which
    which3 = np.array([5, 15])
    c_max = 0.85
    cmap_base = plt.get_cmap("viridis")
    colors = cmap_base(np.linspace(0, c_max, len(which3)))

    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    for j, i in enumerate(which2):
        if i not in which3:
            continue
        color = colors[np.where(which3 == i)[0][0]]
        x_i = np.linspace(-roi, roi, len(v_x_t[i]))

        ax[0].plot(
            x_i,
            v_x_t[i],
            label=rf"$\tau$={taus[i]:.0f}",
            color=color,
            linewidth=3,
        )
        ax[1].plot(
            x_i,
            v_x_i[i],
            color=color,
            linewidth=3,
        )
        rim = v_x_t[i] / 2 - rho_x_t[i]
        ax[2].plot(
            x_i,
            rim,
            color=color,
            linewidth=3,
        )

    ax[0].set_ylabel(r"$u_x^{tot} / c_s$")
    ax[1].set_ylabel(r"$u_x^{inc} / c_s$")
    ax[2].set_ylabel(r"$\lambda_{-}$")
    ax[0].set_title("Total velocity")
    ax[1].set_title("Inc velocity")
    ax[2].set_title("Riemann invariant")
    ax[0].set_xlabel(r"$x/\xi$")
    ax[1].set_xlabel(r"$x/\xi$")
    ax[2].set_xlabel(r"$x/\xi$")
    ax[0].legend()
    ax[1].legend()
    # for a in ax:
    # a.set_aspect("equal")
    #     a.set_xlim(-5, 12)
    #     a.set_ylim(-0.73, 0.73)
    plt.tight_layout()
    plt.show()


def field_velocity_x_fit(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    xis = np.load(f"{scan}/xis.npy")
    roi = 12
    x0 = 15
    y0 = 4
    rad = 4
    line = 0

    v_pvms = []
    x_is = []

    which = np.array([16, 23, 30, 39])
    for i in tqdm.tqdm(which):
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            amp_tot = np.abs(field)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field)

            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi, plot=False
            )

            u_tot_zoom_0 = u_tot[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_inc_zoom_0 = u_inc[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            v_x_i_j = u_inc_zoom_0[u_tot_zoom_0.shape[0] // 2 + line, :]

            if i == 16 or i == 23:

                def v_pvm(x, al1, al2, x1, x2, delta_r1, delta_r2):
                    """Vortex model for velocity profile"""
                    vort1 = al1 * (
                        2
                        * 1
                        / ((x - x1) ** 2 + (delta_r1 / 2) ** 2)
                        * np.abs(delta_r1 / 2)
                    )
                    vort2 = al2 * (
                        2
                        * 1
                        / ((x - x2) ** 2 + (delta_r2 / 2) ** 2)
                        * np.abs(delta_r2 / 2)
                    )
                    sum = vort1 + vort2
                    return sum

                x_i = np.linspace(-roi, roi, len(v_x_i_j))
                x_is += [x_i]
                alg1 = 0.9
                alg2 = 0.1
                x1g = -3
                x2g = 2
                d1g = 5
                d2g = 4
                v_x_pdv = v_x_i_j
                init = [alg1, alg2, x1g, x2g, d1g, d2g]
                popt, pcov = curve_fit(v_pvm, x_i, v_x_pdv, p0=init)
                v_pvms += [v_pvm(x_i, *popt)]
                plt.figure(figsize=[5, 5])
                plt.plot(
                    x_i,
                    v_x_i_j,
                    linewidth=3,
                )
                plt.plot(
                    x_i,
                    v_pvm(x_i, *popt),
                    "--",
                    label=r"$al1$ = "
                    f"{popt[0]:.1f}\n"
                    r"$al2$ = "
                    f"{popt[1]:.1f}\n"
                    r"$x_1$ = "
                    f"{popt[2]:.1f}\n"
                    r"$x_2$ = "
                    f"{popt[3]:.1f}\n"
                    r"$\Delta_{r_1}$ = "
                    f"{popt[4]:.1f}\n"
                    r"$\Delta_{r_2}$ = "
                    f"{popt[5]:.1f}\n",
                )
                plt.legend()
                plt.show()

            if i == 30:

                def v_pvm(x, al1, al2, x1, x2, delta_r1, delta_r2):
                    """Vortex model for velocity profile"""
                    vort1 = al1 * (
                        2
                        * 1
                        / ((x - x1) ** 2 + (delta_r1 / 2) ** 2)
                        * np.abs(delta_r1 / 2)
                    )
                    vort2 = al2 * (
                        2
                        * 1
                        / ((x - x2) ** 2 + (delta_r2 / 2) ** 2)
                        * np.abs(delta_r2 / 2)
                    )
                    sum = vort1 + vort2
                    return sum

                x_i = np.linspace(-roi, roi, len(v_x_i_j))
                x_is += [x_i]
                alg1 = 0.7
                alg2 = 0.0
                x1g = 0
                x2g = 5
                d1g = 4
                d2g = 10
                v_x_pdv = v_x_i_j
                init = [alg1, alg2, x1g, x2g, d1g, d2g]
                popt, pcov = curve_fit(v_pvm, x_i, v_x_pdv, p0=init)
                v_pvms += [v_pvm(x_i, *popt)]
                plt.figure(figsize=[5, 5])
                plt.plot(
                    x_i,
                    v_x_i_j,
                    linewidth=3,
                )
                plt.plot(
                    x_i,
                    v_pvm(x_i, *popt),
                    "--",
                    label=r"$al1$ = "
                    f"{popt[0]:.1f}\n"
                    r"$al2$ = "
                    f"{popt[1]:.1f}\n"
                    r"$x_1$ = "
                    f"{popt[2]:.1f}\n"
                    r"$x_2$ = "
                    f"{popt[3]:.1f}\n"
                    r"$\Delta_{r_1}$ = "
                    f"{popt[4]:.1f}\n"
                    r"$\Delta_{r_2}$ = "
                    f"{popt[5]:.1f}\n",
                )
                plt.legend()
                plt.show()

            if i == 39:

                def v_pvm(x, al1, al2, x1, x2, delta_r1, delta_r2, delta_r3):
                    """Vortex model for velocity profile"""
                    vort1 = al1 * (
                        2
                        * 1
                        / ((x - x1) ** 2 + (delta_r1 / 2) ** 2)
                        * np.abs(delta_r1 / 2)
                    )
                    vort2 = al2 * (
                        2
                        * 1
                        / ((x - x2) ** 2 + (delta_r2 / 2) ** 2)
                        * np.abs(delta_r2 / 2)
                    )
                    vort3 = al2 * (
                        2
                        * (1)
                        / ((x - x2) ** 2 + (delta_r3 / 2) ** 2)
                        * np.abs(delta_r3 / 2)
                    )
                    sum = vort1 + vort2 + vort3
                    return sum

                x_i = np.linspace(-roi, roi, len(v_x_i_j))
                x_is += [x_i]
                alg1 = 0.6
                alg2 = 0.2
                x1g = 0
                x2g = 7
                d1g = 4
                d2g = 12
                d3g = 5
                v_x_pdv = v_x_i_j
                init = [alg1, alg2, x1g, x2g, d1g, d2g, d3g]
                popt, pcov = curve_fit(v_pvm, x_i, v_x_pdv, p0=init)
                v_pvms += [v_pvm(x_i, *popt)]
                plt.figure(figsize=[5, 5])
                plt.title(f"i = {i}")
                plt.plot(
                    x_i,
                    v_x_i_j,
                    linewidth=3,
                )
                plt.plot(
                    x_i,
                    v_pvm(x_i, *popt),
                    "--",
                    label=r"$al1$ = "
                    f"{popt[0]:.1f}\n"
                    r"$al2$ = "
                    f"{popt[1]:.1f}\n"
                    r"$x_1$ = "
                    f"{popt[2]:.1f}\n"
                    r"$x_2$ = "
                    f"{popt[3]:.1f}\n"
                    r"$\Delta_{r_1}$ = "
                    f"{popt[4]:.1f}\n"
                    r"$\Delta_{r_2}$ = "
                    f"{popt[5]:.1f}\n"
                    r"$\Delta_{r_3}$ = "
                    f"{popt[6]:.1f}\n",
                )
                plt.legend()
                plt.show()

    v_pvms = np.array(v_pvms)
    x_is = np.array(x_is)

    which2 = np.arange(0, 40, 1)
    which3 = np.array([16, 23, 30, 39])
    c_max = 0.85
    cmap_base = plt.get_cmap("viridis")
    colors = cmap_base(np.linspace(0, c_max, len(which3)))  # Seulement 4 couleurs

    o = 0
    plt.figure(figsize=[3, 3])
    for i in range(len(which2)):
        if i not in which3:
            continue
        color = colors[np.where(which3 == i)[0][0]]  # Index local dans which3
        v_pvm_i = v_pvms[o]
        x_i = x_is[o]

        plt.plot(
            x_i,
            v_pvm_i,
            color=color,
            linewidth=3,
        )
        o += 1
    plt.xlim(-5, 12)
    plt.ylim(0, 0.73)
    plt.tight_layout()
    plt.show()


def velocity_bgd(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 20
    x0 = -10
    y0 = 0
    rad = 2
    # line = 2
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    v_x_t = []
    v_t = []
    xs = []

    # which = np.arange(16, 40, 2)
    which = np.arange(0, 20, 1)
    # which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(which):
        xi = xis[i]
        xi_cs = np.sqrt(L / (k0 * taus[i]))
        roi_xi = int(roi * (xi / d_real))
        if i == 0:
            y0 = 0
            x0 = -20
        else:
            y0 = 0
            x0 = -10
        for j in np.array([0]):
            field_ref = fields_ref[i, j, :, :]

            field_ref = ndimage.gaussian_filter(field_ref, rad)
            amp_tot = np.abs(field_ref)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field_ref)

            (
                u_tot,
                _,
                _,
            ) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi_cs, plot=False
            )

            u_tot_norm = np.hypot(u_tot[0], u_tot[1])
            u_tot_norm = u_tot_norm[
                u_tot_norm.shape[-2] // 2 - y0 - roi_xi : u_tot_norm.shape[-2] // 2
                - y0
                + roi_xi,
                u_tot_norm.shape[-1] // 2 + x0 - roi_xi : u_tot_norm.shape[-1] // 2
                + x0
                + roi_xi,
            ]

            amp_tot = amp_tot[
                amp_tot.shape[-2] // 2 - y0 - roi_xi : amp_tot.shape[-2] // 2
                - y0
                + roi_xi,
                amp_tot.shape[-1] // 2 + x0 - roi_xi : amp_tot.shape[-1] // 2
                + x0
                + roi_xi,
            ]

            fig, ax = plt.subplots(1, 2, figsize=[10, 5])
            ax[0].imshow(
                amp_tot,
                cmap="gray",
                interpolation="none",
                extent=[
                    -roi_xi * d_real / (1 * xi),
                    roi_xi * d_real / (1 * xi),
                    -roi_xi * d_real / (1 * xi),
                    roi_xi * d_real / (1 * xi),
                ],
            )
            ax[0].set_title("Amplitude")
            ax[0].set_xlabel(r"$x/\xi$")
            ax[0].set_ylabel(r"$y/\xi$")
            ax[1].imshow(
                u_tot_norm,
                cmap="gray",
                interpolation="none",
                extent=[
                    -roi_xi * d_real / (1 * xi),
                    roi_xi * d_real / (1 * xi),
                    -roi_xi * d_real / (1 * xi),
                    roi_xi * d_real / (1 * xi),
                ],
            )
            ax[1].set_title("Velocity")
            ax[1].set_xlabel(r"$x/\xi$")
            ax[1].set_ylabel(r"$y/\xi$")
            plt.tight_layout()
            plt.show()

            u_tot_polar = pT.convertToPolarImage(
                u_tot_norm[:, :],
                center=[u_tot_norm.shape[0] // 2, u_tot_norm.shape[1] // 2],
                finalRadius=u_tot_norm.shape[0] // 2,
                radiusSize=u_tot_norm.shape[0] // 2,
                angleSize=u_tot_norm.shape[1],
            )[0].T

            u_tot_polar_mean = np.mean(u_tot_polar, axis=1)
            v_x_t += [u_tot_polar_mean]
            x_roi = np.linspace(0, roi, u_tot_polar_mean.shape[0])
            xs += [x_roi]

    v_x_t = np.array(v_x_t)
    xs = np.array(xs)
    np.save(f"{scan}/v_x_t.npy", v_x_t)
    np.save(f"{scan}/x_roi.npy", xs)
    plt.figure()
    for i in range(v_x_t.shape[0]):
        plt.plot(xs[i], v_x_t[i], label=rf"$\tau=${taus[i]:.0f}")
    plt.title("Velocity background")
    plt.xlabel(r"r/$\xi$")
    plt.ylabel(r"$u^{bgd}_x/c_s$")
    plt.legend()
    plt.show()


def velocity_bgd_test(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    rad = 2
    N_times = fields_ref.shape[0]

    v_x_t = []
    xs = []

    which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(which):
        for j in np.array([0]):
            field_ref = fields_ref[i, j, :, :]

            field_ref = ndimage.gaussian_filter(field_ref, rad)
            amp_tot = np.abs(field_ref)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field_ref)

            amp_tot = amp_tot[
                :,
                amp_tot.shape[-1] // 2 - amp_tot.shape[-2] // 2 : amp_tot.shape[-1] // 2
                + amp_tot.shape[-2] // 2,
            ]
            phase_tot = phase_tot[
                :,
                phase_tot.shape[-1] // 2 - phase_tot.shape[-2] // 2 : phase_tot.shape[
                    -1
                ]
                // 2
                + phase_tot.shape[-2] // 2,
            ]

            # find images barrycenter for np.roll
            frame_sum_x = np.sum(amp_tot, axis=0)
            frame_sum_y = np.sum(amp_tot, axis=1)
            frame_sum_x /= np.sum(frame_sum_x)
            frame_sum_y /= np.sum(frame_sum_y)
            x = np.arange(amp_tot.shape[1])
            y = np.arange(amp_tot.shape[0])
            x_bary = np.sum(x * frame_sum_x)
            y_bary = np.sum(y * frame_sum_y)

            # plt.figure()
            # plt.imshow(phase_tot)
            # plt.scatter(
            #     x_bary,
            #     y_bary,
            #     color="red",
            # )
            # plt.show()

            amp_tot = np.roll(
                amp_tot,
                shift=[
                    amp_tot.shape[0] // 2 - int(y_bary),
                    amp_tot.shape[1] // 2 - int(x_bary),
                ],
                axis=(0, 1),
            )
            phase_tot = np.roll(
                phase_tot,
                shift=[
                    amp_tot.shape[0] // 2 - int(y_bary),
                    amp_tot.shape[1] // 2 - int(x_bary),
                ],
                axis=(0, 1),
            )

            (
                u_tot,
                _,
                _,
            ) = velocity.helmholtz_decomp_phase(
                amp_tot * np.exp(1j * phase_tot), dx=d_real, plot=False
            )

            u_tot_norm = np.hypot(u_tot[0], u_tot[1]) / k0
            roi_xi = amp_tot.shape[0] // 2 * (d_real * 1e3)

            u_tot_polar = pT.convertToPolarImage(
                u_tot_norm[:, :],
                center=[u_tot_norm.shape[0] // 2, u_tot_norm.shape[1] // 2],
                finalRadius=u_tot_norm.shape[0] // 2,
                radiusSize=u_tot_norm.shape[0] // 2,
                angleSize=u_tot_norm.shape[1],
            )[0].T

            u_tot_polar_mean = np.mean(u_tot_polar, axis=1)
            v_x_t += [u_tot_polar_mean]
            x_roi = np.linspace(0, roi_xi, u_tot_polar_mean.shape[0])
            xs += [x_roi]

    v_x_t = np.array(v_x_t)
    xs = np.array(xs)
    np.save(f"{scan}/v_x.npy", v_x_t)
    np.save(f"{scan}/r.npy", xs)
    plt.figure()
    for i in range(v_x_t.shape[0]):
        plt.plot(xs[i], v_x_t[i], label=rf"$\tau=${taus[i]:.0f}")
    plt.title("Velocity background")
    plt.xlabel(r"$r$ (mm)")
    plt.ylabel(r"$(1/k_0) * \nabla \phi_r$")
    plt.legend()
    plt.show()


def plot_traj(scan):
    traj = np.load(f"{scan}/vortex_trajectories_p.npy")

    d = 1.5
    D = 5

    init_pos = np.array(
        [
            [0.0, D / 2 + d / 2],
            [0.0, D / 2 - d / 2],
            [0.0, -D / 2 - d / 2],
            [0.0, -D / 2 + d / 2],
        ]
    )

    vortex = ["+", "+", "-", "-"]
    color = ["tab:blue", "mediumaquamarine", "tab:red", "darksalmon"]

    print("Plotting ...")
    plt.figure(figsize=(5, 3))
    plt.title(rf"Vortices trajectories")
    plt.grid()
    plt.xlabel(r"$x/\xi$")
    plt.ylabel(r"$y/\xi$")
    valid_trajectories = [traj for traj in traj]

    correction = True
    for v_idx, trajectory in enumerate(valid_trajectories):
        trajectory = np.array(trajectory)
        if correction:
            for j in range(trajectory.shape[0]):
                # if j == 10 or j == 20 or j == 30 or j == len(trajectory) - 1:
                #     continue
                # trajectory[j, 0] = trajectory[j, 0] / (d_real / xis[j])
                # trajectory[j, 1] = trajectory[j, 1] / (d_real / xis[j])
                # trajectory[j, 0] += velo_out.shape[-1] // 2
                # trajectory[j, 1] += velo_out.shape[-2] // 2
                # x_futur = (
                #     velo_out[j, 0, 0, int(trajectory[j, 1]), int(trajectory[j, 0])]
                #     * dtaus[j]
                # )
                # y_futur = (
                #     velo_out[j, 0, 1, int(trajectory[j, 1]), int(trajectory[j, 0])]
                #     * dtaus[j]
                # )
                # trajectory[j, 0] -= velo_out.shape[-1] // 2
                # trajectory[j, 1] -= velo_out.shape[-2] // 2
                # trajectory[j, 0] = trajectory[j, 0] * (d_real / xis[j])
                # trajectory[j, 1] = trajectory[j, 1] * (d_real / xis[j])
                # trajectory[j + 1, 0] -= x_futur
                # trajectory[j + 1, 1] -= y_futur
                if j > 29:
                    trajectory[j, 0] -= trajectory[30, 0] - trajectory[30 - 1, 0]
                    trajectory[j, 1] -= trajectory[30, 1] - trajectory[30 - 1, 1]

        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "-",
            color=color[v_idx],
        )
        plt.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            facecolor="none",
            edgecolors=color[v_idx],
            label=f"{vortex[v_idx]}",
        )
        plt.scatter(
            init_pos[v_idx, 0],
            init_pos[v_idx, 1],
            facecolor=color[v_idx],
            edgecolors=color[v_idx],
            zorder=3,
        )
    plt.xlim(-2.5, 15)
    plt.ylim(0, 10)
    plt.title(rf"Vortex trajectories with correction={correction}")
    plt.show()


def vortex_fit(scan):
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field_vortex.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 5
    x0 = 0
    y0 = 10
    rad = 0
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    which = np.array([3, 9])
    which = np.arange(23, 40, 1)
    for i in which:
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            # extent = np.array(
            #     [
            #         -field.shape[-1] * d_real * 1e3 / (2),
            #         field.shape[-1] * d_real * 1e3 / (2),
            #         -field.shape[-2] * d_real * 1e3 / (2),
            #         field.shape[-2] * d_real * 1e3 / (2),
            #     ]
            # )
            # phase = np.angle(field_ref)
            # phase_unwrap = restoration.unwrap_phase(phase, wrap_around=(True, True))
            # phi_az = phase_unwrap[
            #     phase_unwrap.shape[0] // 2,
            #     :,
            # ]
            # tau_ref = np.abs(np.nanmax(phi_az) - np.nanmin(phi_az))
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # im = ax[0].imshow(
            #     phase, cmap="twilight_shifted", interpolation="none", extent=extent
            # )
            # ax[0].set_title("Phase wrapped")
            # plt.colorbar(im, ax=ax[0])
            # im = ax[1].imshow(
            #     phase_unwrap,
            #     cmap="inferno",
            #     interpolation="none",
            #     extent=extent,
            # )
            # ax[1].set_title("Phase unwrapped")
            # ax[1].axhline(y=0, color="white", linestyle="--")
            # plt.colorbar(im, ax=ax[1])
            # ax[2].plot(
            #     np.linspace(
            #         -field.shape[-1] * d_real * 1e3 / (2),
            #         field.shape[-1] * d_real * 1e3 / (2),
            #         phi_az.shape[0],
            #     ),
            #     phi_az,
            # )
            # ax[2].set_title(f"Phase profile, |Δφ| = {tau_ref:.2f} rad")
            # for a in ax:
            #     a.set_xlabel(r"$x$ (mm)")
            #     if a != ax[2]:
            #         a.set_ylabel(r"$y$ (mm)")
            #     if a == ax[2]:
            #         a.set_ylabel(r"$\phi$ (rad)")

            # plt.show()

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            window_m = 40
            field_zoom = field[
                field.shape[-2] // 2 - y0 - window_m // 2 : field.shape[-2] // 2
                - y0
                + window_m // 2,
                field.shape[-1] // 2 + x0 - window_m // 2 : field.shape[-1] // 2
                + x0
                + window_m // 2,
            ]
            amp_zoom = np.abs(field_zoom)
            amp_zoom = amp_zoom / np.nanmax(amp_zoom)

            x_v = np.arange(amp_zoom.shape[1])
            y_v = np.arange(amp_zoom.shape[0])
            X_v, Y_v = np.meshgrid(x_v, y_v)
            inital_guess = (window_m, window_m, 10)
            try:
                popt, pcov = opt.curve_fit(
                    Vortex2D, (Y_v, X_v), amp_zoom.ravel(), p0=inital_guess
                )
            except RuntimeError:
                popt = np.array([window_m, window_m, np.nan])
            vortex_reshape = Vortex2D((Y_v, X_v), *popt).reshape(window_m, window_m)
            vort_rad = popt[2] * d_real

            extent = np.array(
                [
                    -amp_zoom.shape[-1] * d_real * 1e6 / (2 * 1),
                    amp_zoom.shape[-1] * d_real * 1e6 / (2 * 1),
                    -amp_zoom.shape[-2] * d_real * 1e6 / (2 * 1),
                    amp_zoom.shape[-2] * d_real * 1e6 / (2 * 1),
                ]
            )
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plt.suptitle(
                rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {vort_rad * 1e6:.0f} $\mu m$"
            )
            im0 = ax[0].imshow(amp_zoom, cmap="gray", extent=extent)
            im1 = ax[1].imshow(vortex_reshape, cmap="gray", extent=extent)
            for a in ax:
                a.set_xlabel(r"$x$ ($\mu m$)")
                a.set_ylabel(r"$y$ ($\mu m$)")
                a.set_aspect("equal")
            ax[0].set_title("Amplitude image")
            ax[1].set_title("Two-dimensional fit")
            plt.colorbar(im0, ax=ax[0])
            plt.colorbar(im1, ax=ax[1])
            plt.show()


def phase_x(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 30
    x0 = 0
    y0 = 10
    rad = 3
    line = 0
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    v_x_t = []
    phase_x_t = []

    v_x_t_roi = []
    phase_x_t_roi = []
    x_i_t_roi = []

    # which = np.arange(10, 40, 2)
    # which = np.array([16, 23, 30, 39])
    which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(which):
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)

            # field = field[
            #     field.shape[-2] // 2 - roi_xi : field.shape[-2] // 2 + roi_xi,
            #     field.shape[-1] // 2 - roi_xi : field.shape[-1] // 2 + roi_xi,
            # ]
            field = np.flip(field, axis=0)
            np.save(f"{scan}/field_processed_{i}_{j}.npy", field)

            amp_tot = np.abs(field)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field)

            extent = np.array(
                [
                    -field.shape[-1] * d_real / (2 * xi),
                    field.shape[-1] * d_real / (2 * xi),
                    -field.shape[-2] * d_real / (2 * xi),
                    field.shape[-2] * d_real / (2 * xi),
                ]
            )

            # plt.figure()
            # plt.imshow(
            #     phase_tot,
            #     extent=extent,
            #     cmap="twilight_shifted",
            #     interpolation="none",
            # )
            # plt.colorbar()
            # plt.title(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xi * 1e6:.0f} $\mu m$")
            # plt.xlabel(r"$x/\xi$")
            # plt.ylabel(r"$y/\xi$")
            # plt.show()

            phase_x = phase_tot[phase_tot.shape[0] // 2 + line, :]

            (u_tot, _, _) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi, plot=False
            )
            v_x = u_tot[0]  # x-component of the velocity field
            velo_x = v_x[v_x.shape[0] // 2 + line, :]
            velo_x_roi = velo_x[
                velo_x.shape[0] // 2 - roi_xi : velo_x.shape[0] // 2 + roi_xi
            ]

            # do a subplot of the two figures above
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im = ax[0].imshow(
                phase_tot,
                # extent=extent,
                cmap="twilight_shifted",
                interpolation="none",
            )
            ax[0].axhline(
                y=phase_tot.shape[0] // 2 + line, color="white", linestyle="--"
            )
            ax[0].set_title("Phase field")
            ax[0].set_xlabel(r"$x/\xi$")
            ax[0].set_ylabel(r"$y/\xi$")
            plt.colorbar(im, ax=ax[0])
            ax[1].plot(
                np.unwrap(phase_x),
                label=rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xi * 1e6:.0f} $\mu m$",
            )
            ax[1].plot(
                velo_x,
                label=rf"Velocity",
            )
            ax[1].set_title("Phase profile")
            ax[1].set_xlabel(r"$x/\xi$")
            ax[1].set_ylabel(r"$\phi$")
            plt.legend()
            plt.tight_layout()
            plt.show()

            v_x_t += [velo_x]
            phase_x_unwrap = np.unwrap(phase_x)
            phase_x_unwrap -= phase_x_unwrap[phase_x_unwrap.shape[0] // 2]
            phase_x_unwrap_roi = phase_x_unwrap[
                phase_x_unwrap.shape[0] // 2 - roi_xi : phase_x_unwrap.shape[0] // 2
                + roi_xi
            ]

            phase_x_t += [phase_x_unwrap]
            v_x_t_roi += [velo_x_roi]
            phase_x_t_roi += [phase_x_unwrap_roi]

    v_x_t = np.array(v_x_t)
    phase_x_t = np.array(phase_x_t)
    v_x_t_roi = np.array(v_x_t_roi)
    phase_x_t_roi = np.array(phase_x_t_roi)
    # np.save(f"{scan}/phase_x_t.npy", phase_x_t)
    # np.save(f"{scan}/v_x_t.npy", v_x_t)
    # np.save(f"{scan}/phase_x_t_roi.npy", phase_x_t_roi)
    # np.save(f"{scan}/v_x_t_roi.npy", v_x_t_roi)

    which2 = np.array([16, 29, 30, 39])
    # which2 = np.array([1, 4, 7, 12, 14, 16])
    c_max = 0.85
    cmap_base = plt.get_cmap("viridis")
    colors = cmap_base(np.linspace(0, c_max, len(which2)))

    dphi = []

    fig, ax = plt.subplots(1, 2, figsize=[15, 5])
    for j, i in enumerate(which):
        phase_i = phase_x_t_roi[i]
        dphi += [phase_i[-1] - phase_i[0]]
        if i not in which2:
            continue
        color = colors[np.where(which2 == i)[0][0]]
        x_i = np.linspace(-roi, roi, len(v_x_t[i]))
        x_j = np.linspace(-roi, roi, len(v_x_t_roi[j]))
        phase_j = phase_x_t_roi[j]

        ax[0].plot(
            x_j,
            phase_j / (np.pi) - np.pi / 2 + 0.3,
            label=rf"$\tau$={taus[i]:.0f}",
            color=color,
            linewidth=3,
        )
        ax[1].plot(
            x_j,
            v_x_t_roi[j],
            color=color,
            linewidth=3,
        )

    ax[0].set_ylabel(r"$\phi_x / \pi$")
    ax[1].set_ylabel(r"$u_x^{tot} / c_s$")
    ax[0].set_title("Phase")
    ax[1].set_title("Tot velocity")
    ax[0].set_xlabel(r"$x/\xi$")
    ax[1].set_xlabel(r"$x/\xi$")
    ax[0].legend()
    ax[1].legend()
    for a in ax:
        # a.set_aspect("equal")
        a.set_xlim(-5, 12)
        # a.set_ylim(-0.73, 0.73)
    plt.tight_layout()
    plt.show()

    dphi = np.array(dphi)
    u0 = np.flip(4 * 10 / np.linspace(20, 50, phase_x_t_roi.shape[-1]))
    plt.figure()
    plt.plot(
        u0[:16],
        dphi[:16] / (np.pi),
        marker="o",
    )
    plt.xlabel(r"$u_0$")
    plt.ylabel(r"$\Delta \phi_x / \pi$")
    plt.title("Phase difference")
    plt.show()


def vortex_jrs_analytic():
    Q1 = 1
    Q2 = 1
    Q3 = 1
    delta_r1 = 6.0
    delta_r2 = 15
    delta_r3 = 6.0
    x1 = 0
    x2 = 5
    x3 = 5
    Nbr = 5
    N = 100 * Nbr
    roi = 2 * Nbr
    line = 0
    # Generate grid (without shifting X and Y)
    x = np.linspace(-roi, roi, N)
    y = np.linspace(-roi, roi, N)
    X, Y = np.meshgrid(x, y)

    ampv1, phasev1 = Vortex(X, Y, delta_r1 / 2, delta_r1 / 2, Q1, 1)
    ampv2, phasev2 = Vortex(X, Y, -delta_r1 / 2, -delta_r1 / 2, Q1, 1)

    ampj, phasej = JRS_padé(X, Y)

    # amptot = ampv1 + ampv2
    # phasetot = phasev1 + phasev2
    amptot = ampj
    phasetot = phasej
    psitot = amptot * np.exp(1j * (phasetot))
    amptot = np.abs(psitot)
    phasetot = np.angle(psitot)
    velotot = velocity.velocity(phasetot, dx=roi * 2 / N)

    plt.figure()
    plt.imshow(
        amptot,
        extent=[x.min(), x.max(), y.max(), y.min()],
        cmap="gray",
        interpolation="none",
    )
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(
        phasetot,
        extent=[x.min(), x.max(), y.max(), y.min()],
        cmap="twilight_shifted",
        interpolation="none",
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.colorbar()
    plt.streamplot(X, Y, velotot[0], velotot[1], color="white", linewidth=1, density=1)
    plt.show()


def equiphase(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    # fields = np.load(f"{scan}/field.npy")
    fields = load_cached_array(
        filepath=f"{scan}", filename="field.npy", cache_folder_path=path_cache
    )
    print("Loading data fields ref...")
    # fields_ref = np.load(f"{scan}/field_ref.npy")
    fields_ref = load_cached_array(
        filepath=f"{scan}", filename="field_ref.npy", cache_folder_path=path_cache
    )
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis.npy")
    roi = 7.5
    x0 = 20
    y0 = 0
    rad = 3
    line = 0
    N_times = fields_ref.shape[0]
    # N_avg = fields_ref.shape[1]
    N_avg = 1

    # which = np.arange(10, 40, 2)
    which = np.array([26, 27, 28, 29, 30])
    # which = np.arange(0, N_times, 1)
    for i in tqdm.tqdm(which):
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in np.array([0]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]

            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            # field = np.flip(field, axis=0)

            amp_tot = np.abs(field)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field)

            (
                _,
                u__inc,
                u_comp,
            ) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi, plot=False
            )

            u_inc_zoom_0 = u__inc[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_inc_zoom_1 = u__inc[
                1,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            u_comp_zoom_0 = u_comp[
                0,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]
            u_comp_zoom_1 = u_comp[
                1,
                field.shape[-2] // 2 - y0 - roi_xi : field.shape[-2] // 2 - y0 + roi_xi,
                field.shape[-1] // 2 + x0 - roi_xi : field.shape[-1] // 2 + x0 + roi_xi,
            ]

            amp_tot_zoom = amp_tot[
                amp_tot.shape[-2] // 2 - roi_xi + y0 : amp_tot.shape[-2] // 2
                + roi_xi
                + y0,
                amp_tot.shape[-1] // 2 - roi_xi + x0 : amp_tot.shape[-1] // 2
                + roi_xi
                + x0,
            ]
            phase_tot_zoom = phase_tot[
                phase_tot.shape[-2] // 2 - roi_xi + y0 : phase_tot.shape[-2] // 2
                + roi_xi
                + y0,
                phase_tot.shape[-1] // 2 - roi_xi + x0 : phase_tot.shape[-1] // 2
                + roi_xi
                + x0,
            ]

            # phase_unwrapped = restoration.unwrap_phase(
            #     phase_tot_zoom, wrap_around=(True, True)
            # )

            # phase_unwrapped = np.unwrap(phase_tot_zoom, axis=0)
            # ny, nx = phase_tot_zoom.shape
            # center_col = nx // 2
            # for j in range(center_col + 1, nx):
            #     col_prev = phase_unwrapped[:, j - 1]
            #     col_curr = phase_unwrapped[:, j]
            #     diff = col_curr - col_prev
            #     median_diff = np.median(diff)
            #     k = np.round(median_diff / (2 * np.pi))
            #     phase_unwrapped[:, j] -= k * 2 * np.pi
            # for j in range(center_col - 1, -1, -1):
            #     col_next = phase_unwrapped[:, j + 1]
            #     col_curr = phase_unwrapped[:, j]
            #     diff = col_curr - col_next
            #     median_diff = np.median(diff)
            #     k = np.round(median_diff / (2 * np.pi))
            #     phase_unwrapped[:, j] -= k * 2 * np.pi

            # x = np.arange(phase_unwrapped.shape[1])
            # y = np.arange(phase_unwrapped.shape[0])
            # X, Y = np.meshgrid(x, y)

            # # # Définir la colormap noir → blanc → noir
            # # colors = [(0, 0, 0), (1, 1, 1), (0, 0, 0)]
            # # cmap_bw = LinearSegmentedColormap.from_list(
            # #     "black_white_black", colors, N=256
            # # )

            # plt.figure()
            # # plt.contour(X, Y, phase_unwrapped, levels=20, cmap=cmap_bw)
            # plt.contour(X, Y, phase_unwrapped, levels=20, cmap="viridis")
            # # square aspect
            # plt.gca().set_aspect("equal")
            # plt.title(f"t={taus[i]:.0f}")
            # plt.show()

            def reconstruct_stream_function(v_inc, dx=1.0):
                """
                Reconstruit la fonction de courant ψ associée à v_inc.
                C'est cet objet qui porte la topologie (vortex), et non une phase.

                v_inc = (vx_inc, vy_inc)
                """

                vx, vy = v_inc
                Ny, Nx = vx.shape

                # vorticité ω = ∂x vy - ∂y vx
                omega = np.gradient(vy, dx, axis=1) - np.gradient(vx, dx, axis=0)

                # Grilles de Fourier
                kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
                ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
                KX, KY = np.meshgrid(kx, ky)

                K2 = KX**2 + KY**2
                K2[0, 0] = 1e-15  # éviter div/0

                # FFT
                omega_k = np.fft.fft2(omega)

                # ψ_k = - ω_k / k^2
                psi_k = -omega_k / K2
                psi_k[0, 0] = 0.0  # choisir un gauge

                psi = np.real(np.fft.ifft2(psi_k))

                return psi

            def reconstruct_phi_from_vcomp(v_comp, dx=1.0):
                """
                Reconstruit la phase dé-topologisée phi_comp à partir du champ
                de vitesse compressible v_comp = (vx, vy).

                Paramètres
                ----------
                v_comp : array shape (2, Ny, Nx)
                    Champ de vitesse compressible.
                dx : float
                    Pas spatial.

                Retour
                ------
                phi : array shape (Ny, Nx)
                    Phase lisse, sans vortex (de-topologisée).
                """

                vx, vy = v_comp

                Ny, Nx = vx.shape

                # Calcul de la divergence du champ compressible
                div_v = np.gradient(vx, dx, axis=1) + np.gradient(vy, dx, axis=0)

                # Grilles en espace de Fourier
                kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
                ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
                KX, KY = np.meshgrid(kx, ky)

                K2 = KX**2 + KY**2
                K2[0, 0] = 1e-15  # éviter la division par zéro pour k=0

                # Résolution de Poisson : φ = - div(v) / k^2 en espace de Fourier
                div_k = fft2(div_v)
                phi_k = -div_k / K2

                # k=0 → définir phase moyenne à zéro (choix de gauge)
                phi_k[0, 0] = 0.0

                # Retour espace réel
                phi = np.real(ifft2(phi_k))

                return phi

            u_inc_zoom = np.array(
                [
                    u_inc_zoom_0,
                    u_inc_zoom_1,
                ]
            )

            u_comp_zoom = np.array(
                [
                    u_comp_zoom_0,
                    u_comp_zoom_1,
                ]
            )
            phi_inc = reconstruct_stream_function(u_inc_zoom)
            phi_comp = reconstruct_phi_from_vcomp(u_comp_zoom)

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            x = np.arange(phi_comp.shape[1])
            y = np.arange(phi_comp.shape[0])
            X, Y = np.meshgrid(x, y)
            im0 = ax[0].imshow(
                phase_tot_zoom,
                cmap="twilight_shifted",
                interpolation="none",
            )
            ax[0].contour(
                X, Y, phase_tot_zoom, levels=20, colors="black", linewidths=0.5
            )
            ax[0].set_title(rf"Phase field, $\tau$ = {taus[i]:.0f}")
            plt.colorbar(im0, ax=ax[0])
            im1 = ax[1].imshow(
                phi_inc,
                cmap="twilight_shifted",
                interpolation="none",
            )
            ax[1].contour(X, Y, phi_inc, levels=20, colors="black", linewidths=0.5)
            ax[1].set_title(rf"De-topologized phase field, $\tau$ = {taus[i]:.0f}")
            plt.colorbar(im1, ax=ax[1])
            plt.show()

            # plt.figure()
            # x = np.arange(phi_comp.shape[1])
            # y = np.arange(phi_comp.shape[0])
            # X, Y = np.meshgrid(x, y)
            # plt.contour(X, Y, phi_comp, levels=20, cmap="viridis")
            # # square aspect
            # plt.gca().set_aspect("equal")
            # plt.title(f"t={taus[i]:.0f}")
            # plt.show()


# %% Manual data processing

if __name__ == "__main__":
    # scan = f"04302233_sheep_feedbackloop"  # Far Figure 1
    scan = f"03201801_sheep_jrs"  # Close Figure 2
    # scan = f"04151542_sheep_big"  # Clusters Figure 3

    # compute_phase_time(scan, plot=True)
    # compute_phase_time_single(
    #     scan, name_dn="dn_mvortex", name_field="field_mvortex", plot=False
    # )
    # compute_n2_Isat(scan, plot=True)
    # compute_final_xi(scan, plot=False)

    # plot_fields(scan, window=20, plot=False)
    # energy_time(scan, plot=False)
    # plot_energy(scan, plot=False)
    # velocity_fields(scan, plot=True)
    # velocity_x(scan, plot=True)
    # velocity_decomp(scan, plot=True)
    # rho_x(scan, plot=True)

    # vortex_tracking(scan, windows_size=20, plot=False)
    # make_mp4_and_gif(scan, fps=2)
    # vortex_tracking_corrected(scan, windows_size=20, plot=True)

    # plot_traj(scan)

    # one jrs only
    # scan = f"C:/Users/33652/Desktop/Thèse/VortexCollision-main/Data/spectral_analysis"
    # one_jrs(scan, plot=True)

    # vortex_jrs_analytic()
    # single_shot(plot=True)
    # point_vortex()
    # point_vortex_BEC()

    # field_velocity_x(scan, plot=True)
    # field_velocity_x_fit(scan, plot=True)

    # velocity_bgd(scan, plot=True)

    # velocity_bgd_test(scan, plot=True)

    # vortex_fit(scan)

    # phase_x(scan, plot=True)

    equiphase(scan, plot=True)
