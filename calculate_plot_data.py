#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_plot_data.py
----------------------
This script calculates and extracts plotting data for Figures 1-4 from raw/derived data.
It exports the results to CSV and HDF5 formats without generating plots.

Required files (Please set the paths in the GLOBAL CONFIGURATION section below):
1. SPAN-I L3 Moment Parquet: psp_swp_spi_sf00_L3_mom_v04.parquet
2. SPC L3 Moment Parquet: psp_sweap_spc_l3_vp_moment.parquet
3. IPC directory (Fig 1, 2, 4): Contains 01au_isotemporal_*.ipc files
4. IPC directory (Fig 3): Contains 01au_isotemporal_*.ipc files (e.g. 01au_isotemporal_0.1_3415.ipc)
5. Timestamp list: num_timestamp_starttimestamp_list.txt
6. SPICE Meta Kernel: psp-private-mk.tm (and associated .bsp files)
"""

import os
from pathlib import Path
import datetime
import gc
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import polars as pl
import h5py
from scipy import signal
from scipy.stats import chi2
from scipy import constants as scipy_const

import spiceypy as spice
from astropy.time import Time
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates.builtin_frames import HeliocentricMeanEcliptic
from sunpy.coordinates import HeliographicCarrington


# ==========================================
# GLOBAL CONFIGURATION (EDIT PATHS HERE)
# ==========================================
SPI_PARQUET_FILE = Path("./data/psp_swp_spi_sf00_L3_mom_v04.parquet")
SPC_PARQUET_FILE = Path("./data/psp_sweap_spc_l3_vp_moment.parquet")
IPC_DIR = Path("./data/on_working/0_001au/")
IPC_DIR_FIG3 = Path("./data/on_working/")
TIMESTAMP_LIST_FILE = Path("./data/on_working/0_001au/num_timestamp_starttimestamp_list.txt")
SPICE_MK_FILE = Path("./spice_kernel/psp/kernels/mk/psp-private-mk.tm")
OUTPUT_DIR = Path("./plot_data")


# ==========================================
# CORE UTILITIES & SPICE CONTEXT
# ==========================================
@contextmanager
def ch_working_directory(path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

class SpiceContext:
    def __init__(self, meta_kernel_path: str | Path):
        self.meta_kernel_path = Path(meta_kernel_path)

    def __enter__(self):
        print("Loading SPICE kernels from meta-kernel: {}".format(self.meta_kernel_path))
        with ch_working_directory(self.meta_kernel_path.parent):
            spice.furnsh(self.meta_kernel_path.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        spice.kclear()

def calc_RTN_velocity(c_1000: coord.SkyCoord):
    r_vec = np.array([c_1000.cartesian.x.to(u.km).value, c_1000.cartesian.y.to(u.km).value, c_1000.cartesian.z.to(u.km).value])
    v_vec = np.array([c_1000.velocity.d_x.to(u.km/u.s).value, c_1000.velocity.d_y.to(u.km/u.s).value, c_1000.velocity.d_z.to(u.km/u.s).value])
    R_hat = r_vec / np.linalg.norm(r_vec)
    N_hat = np.cross(r_vec, v_vec)
    N_hat /= np.linalg.norm(N_hat)
    T_hat = np.cross(N_hat, R_hat)
    V_R = np.dot(v_vec, R_hat) * u.km / u.s
    V_T = np.dot(v_vec, T_hat) * u.km / u.s
    V_N = np.dot(v_vec, N_hat) * u.km / u.s
    return V_R, V_T, V_N

def get_position(timestamp_str: str):
    et = spice.str2et(timestamp_str)
    state, lt = spice.spkezr("PARKER SOLAR PROBE", et, "J2000", "NONE", "SUN")
    pos_vec = coord.CartesianRepresentation(state[0]*u.km, state[1]*u.km, state[2]*u.km)
    vel_vec = coord.CartesianDifferential(state[3]*u.km/u.s, state[4]*u.km/u.s, state[5]*u.km/u.s)
    c_j2000 = coord.SkyCoord(pos_vec.with_differentials(vel_vec), representation_type="cartesian", frame="icrs")
    astropy_time = Time(timestamp_str)
    c_se = c_j2000.transform_to(HeliocentricMeanEcliptic(obstime=astropy_time))
    c_carr = c_j2000.transform_to(HeliographicCarrington(obstime=astropy_time, observer="self"))
    R_AU = c_se.distance.to(u.AU)
    SE_LAT = c_se.lat.to(u.deg)
    SE_LON = c_se.lon.to(u.deg)
    V_R, V_T, V_N = calc_RTN_velocity(c_j2000) 
    HG_LAT = c_carr.lat.to(u.deg)
    HG_LON = c_carr.lon.to(u.deg)
    return R_AU.value, SE_LAT.value, SE_LON.value, V_R.value, V_T.value, V_N.value, HG_LAT.value, HG_LON.value

def flatten_timearray(nested_stuff):
    for item in nested_stuff:
        if isinstance(item, str): yield item
        elif isinstance(item, Time):
            try:
                for t in item: yield t
            except TypeError: yield item
        elif isinstance(item, Iterable): yield from flatten_timearray(item)
        else: yield item

def get_position_flatten_timestamp(timestamp_nested_list, spice_mk_file: str | Path = SPICE_MK_FILE):
    flat_timestamp = list(flatten_timearray(timestamp_nested_list))
    flat_timestamp_str = [t.iso if isinstance(t, Time) else t for t in flat_timestamp]
    R_AU_list, SE_LAT_list, SE_LON_list, V_R_list, V_T_list, V_N_list, HG_LAT_list, HG_LON_list = [], [], [], [], [], [], [], []
    with SpiceContext(spice_mk_file):
        for t_str in flat_timestamp_str:
            R_AU, SE_LAT, SE_LON, V_R, V_T, V_N, HG_LAT, HG_LON = get_position(t_str)
            R_AU_list.append(R_AU); SE_LAT_list.append(SE_LAT); SE_LON_list.append(SE_LON)
            V_R_list.append(V_R); V_T_list.append(V_T); V_N_list.append(V_N)
            HG_LAT_list.append(HG_LAT); HG_LON_list.append(HG_LON)
    return np.array(R_AU_list), np.array(SE_LAT_list), np.array(SE_LON_list), np.array(V_R_list), np.array(V_T_list), np.array(V_N_list), np.array(HG_LAT_list), np.array(HG_LON_list)

def get_timestamp_head(au_list: np.ndarray) -> np.ndarray:
    data = np.char.strip(np.loadtxt(TIMESTAMP_LIST_FILE, delimiter=',', dtype=str))
    au_values = data.T[0, :].astype(np.float64)
    timestamp_str = data.T[2, :]
    idx = []
    for candidate in au_list:
        matched_indices = np.where(np.isclose(au_values, candidate))[0][-1]
        idx.append(matched_indices)
    selected_timestamps = timestamp_str[idx]
    dt_list = np.array([datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in selected_timestamps])
    return dt_list

def def_au_lists() -> list:
    au_list_3414 = np.arange(0.100, 0.162, 0.002)
    au_list_6827 = np.arange(0.162, 0.262, 0.002)
    au_list_109227 = np.arange(0.262, 0.784, 0.002)
    au_list_removed = np.array([0.276, 0.434, 0.436, 0.438, 0.440, 0.442, 0.444, 0.446, 0.448, 0.450, 0.452, 0.492, 0.494, 0.496, 0.498, 0.500, 0.510, 0.512, 0.518, 0.520, 0.522, 0.526, 0.528, 0.530, 0.532, 0.538, 0.540, 0.544, 0.546, 0.548, 0.558, 0.560, 0.562, 0.568, 0.570, 0.572, 0.574, 0.576, 0.588, 0.590, 0.592, 0.602, 0.604, 0.606, 0.610, 0.614, 0.616, 0.628, 0.630, 0.636, 0.638, 0.640, 0.648, 0.650, 0.652, 0.658, 0.660, 0.662, 0.668, 0.670, 0.678, 0.680, 0.684, 0.690, 0.696, 0.706, 0.708, 0.716, 0.724, 0.744, 0.754])
    mask_109227 = ~np.any(np.isclose(au_list_109227[:, None], au_list_removed[None, :]), axis=1)
    au_list_109227_filtered = au_list_109227[mask_109227]
    return [au_list_3414, au_list_6827, au_list_109227_filtered]

def def_au_list() -> np.ndarray:
    return np.concatenate(def_au_lists())

def load_df(au):
    if au<=0.160 or np.isclose(au, 0.160):
        filename = IPC_DIR / '01au_isotemporal_{:.3f}_0.003414.ipc'.format(au)
    elif au<=0.260 or np.isclose(au, 0.260):
        filename = IPC_DIR / '01au_isotemporal_{:.3f}_0.006827.ipc'.format(au)
    else:
        filename = IPC_DIR / '01au_isotemporal_{:.3f}.ipc'.format(au)
    lf = pl.scan_ipc(filename)
    df = lf.collect()
    return df

def load_lf(au : float) -> pl.LazyFrame:
    if au<=0.160 or np.isclose(au, 0.160):
        filename = IPC_DIR / '01au_isotemporal_{:.3f}_0.003414.ipc'.format(au)
    elif au<=0.260 or np.isclose(au, 0.260):
        filename = IPC_DIR / '01au_isotemporal_{:.3f}_0.006827.ipc'.format(au)
    else:
        filename = IPC_DIR / '01au_isotemporal_{:.3f}.ipc'.format(au)
    lf = pl.scan_ipc(filename)
    return lf

def load_df_fig3(au):
    """Original load_df logic from psd_helicity_confidence_interval_align_2.py for Fig 3"""
    if au==0.1:
        filename = IPC_DIR_FIG3 / '01au_isotemporal_{}_3415.ipc'.format(au)
    else:
        filename = IPC_DIR_FIG3 / '01au_isotemporal_{}.ipc'.format(au)
    lf = pl.scan_ipc(filename)
    df = lf.collect()
    return df


# --- Physics Functions ---
def get_flow_speed_mean_spi(timestamp_head : datetime.datetime, interval: datetime.timedelta = datetime.timedelta(hours=1), filename: str | Path = SPI_PARQUET_FILE):
    filename = Path(filename)
    lf = pl.scan_parquet(filename)
    lf = lf.filter(pl.col("timestamp").is_between(timestamp_head, timestamp_head + interval))

    stats = (
        lf.select([
            pl.col("vel_rtn_sun_R").mean().alias("mean_R").fill_null(float("nan")),
            pl.col("vel_rtn_sun_T").mean().alias("mean_T").fill_null(float("nan")),
            pl.col("vel_rtn_sun_N").mean().alias("mean_N").fill_null(float("nan")),
            pl.col("vel_rtn_sun_R").std().alias("std_R").fill_null(float("nan")),
            pl.col("vel_rtn_sun_T").std().alias("std_T").fill_null(float("nan")),
            pl.col("vel_rtn_sun_N").std().alias("std_N").fill_null(float("nan")),
            ])
            .collect()
            )
    
    mean_R = stats["mean_R"][0]
    mean_T = stats["mean_T"][0]
    mean_N = stats["mean_N"][0]
    std_R  = stats["std_R"][0]
    std_T  = stats["std_T"][0]
    std_N  = stats["std_N"][0]

    mean = (mean_R, mean_T, mean_N)
    std = (std_R, std_T, std_N)
    return mean, std

def get_proton_density_mean_spi(timestamp_head : datetime.datetime, interval: datetime.timedelta = datetime.timedelta(hours=1), filename: str | Path = SPI_PARQUET_FILE):
    filename = Path(filename)
    lf = pl.scan_parquet(filename)
    lf = lf.filter(pl.col("timestamp").is_between(timestamp_head, timestamp_head + interval))

    stats = (
        lf.select([
            pl.col("partial_density").mean().alias("mean_density").fill_null(float("nan")),
            pl.col("partial_density").std().alias("std_density").fill_null(float("nan")),
            ])
            .collect()
            )
    
    mean_density = stats["mean_density"][0]
    std_density  = stats["std_density"][0]
    return mean_density, std_density

def get_proton_temperature_mean_spi(timestamp_head : datetime.datetime, interval: datetime.timedelta = datetime.timedelta(hours=1), filename: str | Path = SPI_PARQUET_FILE):
    filename = Path(filename)
    lf = pl.scan_parquet(filename)
    lf = lf.filter(pl.col("timestamp").is_between(timestamp_head, timestamp_head + interval))

    stats = (
        lf.select([
            pl.col("temperature").mean().alias("mean_temperature").fill_null(float("nan")),
            pl.col("temperature").std().alias("std_temperature").fill_null(float("nan")),
            ])
            .collect()
            )
    
    mean_temperature = stats["mean_temperature"][0]
    std_temperature  = stats["std_temperature"][0]
    return mean_temperature, std_temperature

def get_t_tensor_mean_spi(timestamp_head : datetime.datetime, interval: datetime.timedelta = datetime.timedelta(hours=1), filename: str | Path = SPI_PARQUET_FILE):
    filename = Path(filename)
    lf = pl.scan_parquet(filename)
    lf = lf.filter(pl.col("timestamp").is_between(timestamp_head, timestamp_head + interval))

    stats = (
        lf.select([
            pl.col("t_tensor_rtn_rr").mean().alias("mean_rr").fill_null(float("nan")),
            pl.col("t_tensor_rtn_tt").mean().alias("mean_tt").fill_null(float("nan")),
            pl.col("t_tensor_rtn_nn").mean().alias("mean_nn").fill_null(float("nan")),
            pl.col("t_tensor_rtn_rt").mean().alias("mean_rt").fill_null(float("nan")),
            pl.col("t_tensor_rtn_tn").mean().alias("mean_tn").fill_null(float("nan")),
            pl.col("t_tensor_rtn_rn").mean().alias("mean_rn").fill_null(float("nan")),
            pl.col("t_tensor_rtn_rr").std().alias("std_rr").fill_null(float("nan")),
            pl.col("t_tensor_rtn_tt").std().alias("std_tt").fill_null(float("nan")),
            pl.col("t_tensor_rtn_nn").std().alias("std_nn").fill_null(float("nan")),
            pl.col("t_tensor_rtn_rt").std().alias("std_rt").fill_null(float("nan")),
            pl.col("t_tensor_rtn_tn").std().alias("std_tn").fill_null(float("nan")),
            pl.col("t_tensor_rtn_rn").std().alias("std_rn").fill_null(float("nan")),
            ])
            .collect()
            )
    
    mean = np.array([stats["mean_rr"][0], stats["mean_tt"][0], stats["mean_nn"][0], stats["mean_rt"][0], stats["mean_tn"][0], stats["mean_rn"][0]])
    std = np.array([stats["std_rr"][0], stats["std_tt"][0], stats["std_nn"][0], stats["std_rt"][0], stats["std_tn"][0], stats["std_rn"][0]])
    return mean, std

def calc_magnetic_field_unit_vector(lf: pl.LazyFrame) -> np.ndarray:
    lf_unit = lf.select([
        pl.col("R").mean().alias("R_mean"),
        pl.col("T").mean().alias("T_mean"),
        pl.col("N").mean().alias("N_mean")
    ]).with_columns([
        (pl.col("R_mean")**2 + pl.col("T_mean")**2 + pl.col("N_mean")**2).sqrt().alias("mag")
    ]).with_columns([
        (pl.col("R_mean") / pl.col("mag")).alias("R_hat"),
        (pl.col("T_mean") / pl.col("mag")).alias("T_hat"),
        (pl.col("N_mean") / pl.col("mag")).alias("N_hat")
    ]).select(["R_hat", "T_hat", "N_hat"])

    return lf_unit.collect().to_numpy()[0]

def calc_background_mag_field(lf: pl.LazyFrame) -> np.ndarray:
    lf_mean = lf.select([
        pl.col("R").mean().alias("R_mean"),
        pl.col("T").mean().alias("T_mean"),
        pl.col("N").mean().alias("N_mean"),
    ])
    lf_mag = lf_mean.with_columns(abs_mag_field = ((pl.col("R_mean")**2 + pl.col("T_mean")**2 + pl.col("N_mean")**2).sqrt()))
    background_mag_field = lf_mag.collect()["abs_mag_field"][0]
    return background_mag_field

def calc_alfven_speed(background_mag_field : float, mean_density : float) -> float:
    B_si = background_mag_field * 1e-9       # nT to Tesla
    n_si = mean_density * 1e6                 # cm^-3 to m^-3
    m_p = scipy_const.m_p                     # Proton mass in kg
    mu0 = scipy_const.mu_0                    # Permeability of free space
    V_A_si = B_si / np.sqrt(mu0 * n_si * m_p)
    V_A_km_s = V_A_si / 1e3
    return V_A_km_s

def calculate_anisotropy(B_vec, T_tensor_row):
    T = np.array([
        [T_tensor_row[0], T_tensor_row[3], T_tensor_row[5]],  # RR, RT, RN
        [T_tensor_row[3], T_tensor_row[1], T_tensor_row[4]],  # RT, TT, TN
        [T_tensor_row[5], T_tensor_row[4], T_tensor_row[2]]   # RN, TN, NN
    ])
    B_mag = np.linalg.norm(B_vec)
    if B_mag == 0: return np.nan
    b_hat = B_vec / B_mag
    T_para = np.dot(b_hat.T, np.dot(T, b_hat))
    T_trace = np.trace(T)
    T_perp = (T_trace - T_para) / 2.0
    A = T_perp / T_para
    return A, T_para, T_perp

def calculate_beta_parallel_scipy(n_cm3, T_para_ev, B_nt):
    n_si = n_cm3 * 1e6                    # cm^-3 -> m^-3
    T_j  = T_para_ev * scipy_const.e      # eV -> Joule
    B_si = B_nt * 1e-9                    # nT -> Tesla
    mu0 = scipy_const.mu_0
    p_thermal = n_si * T_j
    p_magnetic = (B_si**2) / (2 * mu0)
    beta_para = p_thermal / p_magnetic
    return beta_para

def get_beta_parallel_mean_spi(timestamp_head : datetime.datetime, au: float, interval: datetime.timedelta = datetime.timedelta(hours=1), filename: str | Path = SPI_PARQUET_FILE) -> float:
    filename = Path(filename)
    lf = pl.scan_parquet(filename)
    lf = lf.filter(pl.col("timestamp").is_between(timestamp_head, timestamp_head + interval))

    df = lf.select([
        pl.col("partial_density"),
        pl.col("t_tensor_rtn_rr"), pl.col("t_tensor_rtn_tt"), pl.col("t_tensor_rtn_nn"),
        pl.col("t_tensor_rtn_rt"), pl.col("t_tensor_rtn_tn"), pl.col("t_tensor_rtn_rn"),
    ]).collect()

    lf_B = load_lf(au)
    lf_B_mean = lf_B.select([pl.col("R").mean().alias("R_mean"), pl.col("T").mean().alias("T_mean"), pl.col("N").mean().alias("N_mean")])
    
    B_vec = lf_B_mean.collect()["R_mean", "T_mean", "N_mean"].to_numpy()[0]
    B_nt = np.linalg.norm(B_vec)
    n_cm3 = df["partial_density"].mean()
    if n_cm3 is None: n_cm3 = float("nan")
    T_tensor_row = np.array([df["t_tensor_rtn_rr"], df["t_tensor_rtn_tt"], df["t_tensor_rtn_nn"], df["t_tensor_rtn_rt"], df["t_tensor_rtn_tn"], df["t_tensor_rtn_rn"]]).mean(axis=1)
    A_mean, T_para_mean, T_perp_mean = calculate_anisotropy(B_vec, T_tensor_row)
    beta_para_mean = calculate_beta_parallel_scipy(n_cm3, T_para_mean, B_nt)
    return beta_para_mean

def get_anisotropy_mean_spi(timestamp_head : datetime.datetime, au: float, interval: datetime.timedelta = datetime.timedelta(hours=1), filename: str | Path = SPI_PARQUET_FILE) -> tuple[float, float]:
    filename = Path(filename)
    lf = pl.scan_parquet(filename)
    lf = lf.filter(pl.col("timestamp").is_between(timestamp_head, timestamp_head + interval))

    df = lf.select([
        pl.col("t_tensor_rtn_rr"), pl.col("t_tensor_rtn_tt"), pl.col("t_tensor_rtn_nn"),
        pl.col("t_tensor_rtn_rt"), pl.col("t_tensor_rtn_tn"), pl.col("t_tensor_rtn_rn"),
    ]).collect()

    lf_B = load_lf(au)
    lf_B_mean = lf_B.select([pl.col("R").mean().alias("R_mean"), pl.col("T").mean().alias("T_mean"), pl.col("N").mean().alias("N_mean")])
    
    B_vec = lf_B_mean.collect()["R_mean", "T_mean", "N_mean"].to_numpy()[0]

    T_tensor_row = np.array([df["t_tensor_rtn_rr"], df["t_tensor_rtn_tt"], df["t_tensor_rtn_nn"], df["t_tensor_rtn_rt"], df["t_tensor_rtn_tn"], df["t_tensor_rtn_rn"]]).mean(axis=1)
    A_mean, T_para_mean, T_perp_mean = calculate_anisotropy(B_vec, T_tensor_row)

    return A_mean

def calc_mag_psd_welch(df):
    timestamp_head = df.select("timestamp").head(2).to_numpy()
    dt = np.float64(timestamp_head[1,0] - timestamp_head[0,0]) / 1e6
    mag_data_R = df["R"].to_numpy()
    mag_data_T = df["T"].to_numpy()
    mag_data_N = df["N"].to_numpy()
    mag_data = np.sqrt(mag_data_R**2 + mag_data_T**2 + mag_data_N**2)
    del mag_data_R, mag_data_T, mag_data_N
    gc.collect()

    segment_length = 2**13
    overlap = segment_length // 2

    freqs, psd = signal.welch(mag_data, fs=1.0/dt, window='hann', nperseg=segment_length, noverlap=overlap)
    step_size = segment_length - overlap
    segment_num  = int(np.floor((len(mag_data) - overlap) / step_size))
    nu = 2 * segment_num / (1 + 2/9 - 2/(9 * segment_num)) 
    chi2_lower_quantile = chi2.ppf(0.05 / 2, nu)
    chi2_upper_quantile = chi2.ppf(1 - 0.05 / 2, nu)
    ci_lower = (nu * psd) / chi2_upper_quantile
    ci_upper = (nu * psd) / chi2_lower_quantile
    ci_min = np.minimum(ci_lower, ci_upper)
    ci_max = np.maximum(ci_lower, ci_upper)

    return freqs, psd, ci_min, ci_max

def calc_mag_abs(df):
    df = df.with_columns(
        (pl.col("R") ** 2 + pl.col("T") ** 2 + pl.col("N") ** 2).sqrt().alias("mag")
    )
    result = df.select([
        pl.col("mag").mean().alias("mean"),
        pl.col("mag").std().alias("std")
    ])
    return result["mean"][0], result["std"][0]

def calc_mag_helicity_normalized_welch(df):
    timestamp_head = df.select("timestamp").head(2).to_numpy()
    dt = np.float64(timestamp_head[1,0] - timestamp_head[0,0]) / 1e6
    mag_data_R = df["R"].to_numpy()
    mag_data_T = df["T"].to_numpy()
    mag_data_N = df["N"].to_numpy()

    segment_length = 2**11
    overlap = segment_length // 2
    freqs, csd_T_N = signal.csd(mag_data_T, mag_data_N, fs=1.0/dt, window='hann', nperseg=segment_length, noverlap=overlap)
    freqs, psd_R = signal.welch(mag_data_R, fs=1.0/dt, window='hann', nperseg=segment_length, noverlap=overlap)
    freqs, psd_T = signal.welch(mag_data_T, fs=1.0/dt, window='hann', nperseg=segment_length, noverlap=overlap)
    freqs, psd_N = signal.welch(mag_data_N, fs=1.0/dt, window='hann', nperseg=segment_length, noverlap=overlap)
    helicity_normalized = 2 * (csd_T_N.imag) / (psd_T + psd_N)

    step_size = segment_length - overlap
    segment_num  = int(np.floor((len(mag_data_T) - overlap) / step_size))
    nu = 2 * segment_num / (1 + 2/9 - 2/(9 * segment_num)) 
    w_T = psd_T / (psd_T + psd_N)
    w_N = psd_N / (psd_T + psd_N)
    nu_normalized_helicity = nu / (1 + w_T**2 + w_N**2) 

    chi2_lower_quantile = chi2.ppf(0.05 / 2, nu_normalized_helicity)
    chi2_upper_quantile = chi2.ppf(1 - 0.05 / 2, nu_normalized_helicity)
    ci_lower = (nu_normalized_helicity * helicity_normalized) / chi2_upper_quantile
    ci_upper = (nu_normalized_helicity * helicity_normalized) / chi2_lower_quantile
    ci_min = np.minimum(ci_lower, ci_upper)
    ci_max = np.maximum(ci_lower, ci_upper)

    return freqs, helicity_normalized, ci_min, ci_max

def calc_proton_cyclotron_frequency(B_nT: float, n_p:float = 5.0, Vsw: float = 400) -> float:
    q = scipy_const.e
    m_p = scipy_const.m_p
    mu0 = scipy_const.mu_0
    B_tesla = B_nT * 1e-9
    n_p_m3 = n_p * 1e6
    Vsw_m = Vsw * 1e3
    V_A = B_tesla / np.sqrt(mu0 * n_p_m3 * m_p)
    f_ci = (1 / (2 * np.pi)) * (q * B_tesla / m_p)
    f_ci_obs_para = f_ci * ( 1 + Vsw_m / V_A)
    f_ci_obs_anti = f_ci * abs(1 - Vsw_m / V_A)
    return f_ci, f_ci_obs_para, f_ci_obs_anti

def calc_di_frequency(n_p: float = 5.0, Vsw: float = 400) -> float:
    m_p = scipy_const.m_p
    q = scipy_const.e
    n_p_m3 = n_p * 1e6
    Vsw_m = Vsw * 1e3
    omega_pi = np.sqrt((n_p_m3 * q**2) / (scipy_const.epsilon_0 * m_p))
    d_i = scipy_const.c / omega_pi
    f_di = Vsw_m / (2 * np.pi * d_i)
    return f_di

def get_proton_density_mean_spi_interp() -> np.ndarray:
    au_list = def_au_list()
    timestamp_head_list = get_timestamp_head(au_list)
    mean_densities = []
    for timestamp_head in timestamp_head_list:
        mean_density, std_density = get_proton_density_mean_spi(timestamp_head, filename=SPI_PARQUET_FILE)
        mean_densities.append(mean_density)

    mean_densities = np.array(mean_densities)
    n = len(mean_densities)
    t = np.arange(n)
    mask = ~np.isnan(mean_densities)
    mean_densities_interp = np.interp(t, t[mask], mean_densities[mask])
    return mean_densities_interp

def get_flow_speed_mean_spi_interp() -> np.ndarray:
    au_list = def_au_list()
    timestamp_head_list = get_timestamp_head(au_list)
    mean_flow_speeds = []
    for timestamp_head in timestamp_head_list:
        mean_flow_speed, std_flow_speed = get_flow_speed_mean_spi(timestamp_head, filename=SPI_PARQUET_FILE)
        mean_flow_speeds.append(np.sqrt(mean_flow_speed[0]**2 + mean_flow_speed[1]**2 + mean_flow_speed[2]**2))

    mean_flow_speeds = np.array(mean_flow_speeds)
    n = len(mean_flow_speeds)
    t = np.arange(n)
    mask = ~np.isnan(mean_flow_speeds)
    mean_flow_speeds_interp = np.interp(t, t[mask], mean_flow_speeds[mask])
    return mean_flow_speeds_interp


# ==========================================
# EXTRACTION FUNCTIONS
# ==========================================

def calc_figure1():
    print("Calculating data for Figure 1...")
    au_list = def_au_list()
    timestamp_head_list = get_timestamp_head(au_list)

    means_R, means_T, means_N = [], [], []
    stds_R, stds_T, stds_N = [], [], []
    means_density, stds_density = [], []
    means_temperature, stds_temperature = [], []
    t_tensor_mean, t_tensor_std = [], []
    temperature_anisotropy, mean_beta_parallel = [], []
    alfven_mach_number, shift_k_coef, taylor_epsilon = [], [], []

    timestamp_head_list_astropy = Time(timestamp_head_list)
    R_AU_np, SE_LAT_np, SE_LON_np, V_R_np, V_T_np, V_N_np, HG_LAT_np, HG_LON_np = get_position_flatten_timestamp(timestamp_head_list_astropy, SPICE_MK_FILE)
    spacecraft_velocity_np = np.vstack((V_R_np, V_T_np, V_N_np)).T

    for timestamp_head, au, spacecraft_velocity in zip(timestamp_head_list, au_list, spacecraft_velocity_np):
        lf = load_lf(au)

        mean, std = get_flow_speed_mean_spi(timestamp_head, filename=SPI_PARQUET_FILE)
        mean_density, std_density = get_proton_density_mean_spi(timestamp_head, filename=SPI_PARQUET_FILE)
        mean_temperature, std_temperature = get_proton_temperature_mean_spi(timestamp_head, filename=SPI_PARQUET_FILE)
        mean_t_tensor, std_t_tensor = get_t_tensor_mean_spi(timestamp_head, filename=SPI_PARQUET_FILE)
        mean_anisotropy = get_anisotropy_mean_spi(timestamp_head, au, filename=SPI_PARQUET_FILE)
        mean_beta_parallel_value = get_beta_parallel_mean_spi(timestamp_head, au, filename=SPI_PARQUET_FILE)

        solar_wind_velocity = np.array([mean[0], mean[1], mean[2]])
        relative_velocity = solar_wind_velocity - spacecraft_velocity
        B_hat = calc_magnetic_field_unit_vector(lf)
        relative_velocity_perp = np.linalg.norm(np.cross(relative_velocity, B_hat))

        background_mag_field = calc_background_mag_field(lf)
        alfven_mach_number_value = mean[0] / calc_alfven_speed(background_mag_field, mean_density)
        shift_k_coef_value = 1 + calc_alfven_speed(background_mag_field, mean_density) / relative_velocity_perp
        taylor_epsilon_value = np.linalg.norm(std) / (np.sqrt(2) * relative_velocity_perp )

        means_R.append(mean[0]); means_T.append(mean[1]); means_N.append(mean[2])
        stds_R.append(std[0]); stds_T.append(std[1]); stds_N.append(std[2])
        means_density.append(mean_density); stds_density.append(std_density)
        means_temperature.append(mean_temperature); stds_temperature.append(std_temperature)
        t_tensor_mean.append(mean_t_tensor); t_tensor_std.append(std_t_tensor)
        temperature_anisotropy.append(mean_anisotropy)
        mean_beta_parallel.append(mean_beta_parallel_value)
        alfven_mach_number.append(alfven_mach_number_value)
        shift_k_coef.append(shift_k_coef_value)
        taylor_epsilon.append(taylor_epsilon_value)

    # Export to CSV
    df_fig1 = pl.DataFrame({
        'au': au_list,
        'mean_R': means_R, 'mean_T': means_T, 'mean_N': means_N,
        'std_R': stds_R, 'std_T': stds_T, 'std_N': stds_N,
        'mean_density': means_density, 'std_density': stds_density,
        'mean_temperature': means_temperature, 'std_temperature': stds_temperature,
        'beta_parallel': mean_beta_parallel,
        'anisotropy': temperature_anisotropy,
        'taylor_epsilon': taylor_epsilon,
        'shift_k_coef': shift_k_coef
    })
    df_fig1.write_csv(OUTPUT_DIR / 'fig1_data.csv')
    print("-> Saved plot_data/fig1_data.csv")

def calc_figure2():
    print("Calculating data for Figure 2...")
    au_lists = def_au_lists()
    
    # Process Top Panel
    freqs_list_top, magnetic_fields_list, B_mean_list, B_std_list = [], [], [], []
    au_list_np_flat = np.array([au for au_list in au_lists for au in au_list])
    
    for au_list in au_lists:
        for au in au_list:
            df = load_df(au)
            freqs, magnetic_field, ci_lower, ci_upper = calc_mag_psd_welch(df)
            B_mean, B_std = calc_mag_abs(df)
            freqs_list_top.append(freqs)
            magnetic_fields_list.append(magnetic_field)
            B_mean_list.append(B_mean)
            B_std_list.append(B_std)

    # Process Bottom Panel
    au_list_bot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7823]
    freqs_list_bot, psd_list_bot, ci_lower_list_bot, ci_upper_list_bot = [], [], [], []
    for au in au_list_bot:
        df = load_df(au)
        freqs, psd, ci_lower, ci_upper = calc_mag_psd_welch(df)
        freqs_list_bot.append(freqs)
        psd_list_bot.append(psd)
        ci_lower_list_bot.append(ci_lower)
        ci_upper_list_bot.append(ci_upper)

    # Export to HDF5
    with h5py.File(OUTPUT_DIR / 'fig2_data.h5', 'w') as f:
        grp_top = f.create_group('top_panel')
        grp_top.create_dataset('Distance_AU', data=au_list_np_flat)
        grp_top.create_dataset('Frequency_Hz', data=np.array(freqs_list_top))
        grp_top.create_dataset('PSD_Colormap', data=np.array(magnetic_fields_list))
        grp_top.create_dataset('B_Mean', data=np.array(B_mean_list))
        grp_top.create_dataset('B_Std', data=np.array(B_std_list))
        
        grp_bot = f.create_group('bottom_panel')
        grp_bot.create_dataset('Distance_AU', data=np.array(au_list_bot))
        grp_bot.create_dataset('Frequency_Hz', data=np.array(freqs_list_bot))
        grp_bot.create_dataset('PSD', data=np.array(psd_list_bot))
        grp_bot.create_dataset('CI_Lower', data=np.array(ci_lower_list_bot))
        grp_bot.create_dataset('CI_Upper', data=np.array(ci_upper_list_bot))
    print("-> Saved plot_data/fig2_data.h5")

def calc_figure3():
    print("Calculating data for Figure 3...")
    au_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7823]
    freqs_list, psd_list, ci_lower_list, ci_upper_list = [], [], [], []
    
    # Restored exact load_df_fig3 logic from psd_helicity_confidence_interval_align_2.py
    for au in au_list:
        df = load_df_fig3(au)
        freqs, psd, ci_lower, ci_upper = calc_mag_helicity_normalized_welch(df)
        freqs_list.append(freqs)
        psd_list.append(psd)
        ci_lower_list.append(ci_lower)
        ci_upper_list.append(ci_upper)
        
    with h5py.File(OUTPUT_DIR / 'fig3_data.h5', 'w') as f:
        f.create_dataset('Distance_AU', data=np.array(au_list))
        f.create_dataset('Frequency_Hz', data=np.array(freqs_list))
        f.create_dataset('Helicity_Norm', data=np.array(psd_list))
        f.create_dataset('CI_Lower', data=np.array(ci_lower_list))
        f.create_dataset('CI_Upper', data=np.array(ci_upper_list))
    print("-> Saved plot_data/fig3_data.h5")

def calc_figure4():
    print("Calculating data for Figure 4...")
    au_lists = def_au_lists()
    
    # 1. Top Panel (from psd_helicity_vs_au_wavenumber_obtainedflow_2.py)
    top_au_list = []
    top_freqs_list = []
    top_hel_list = []
    for au_list in au_lists:
        for au in au_list:
            df = load_df(au)
            freqs, hel_normalized, ci_lower, ci_upper = calc_mag_helicity_normalized_welch(df)
            top_au_list.append(au)
            top_freqs_list.append(freqs)
            top_hel_list.append(hel_normalized)

    # 2. Middle Panel (f/f_ci norm, from helicity_flow_analysis_2.py)
    mean_densities_interp = get_proton_density_mean_spi_interp()
    mean_flow_speeds_interp = get_flow_speed_mean_spi_interp()

    mid_au_list = []
    mid_freqs_list = []
    mid_hel_list = []
    for au_list in au_lists:
        for i, au in enumerate(au_list):
            lf = load_lf(au)
            df = lf.select(["timestamp", "R", "T", "N"]).collect()
            freqs, hel_normalized, ci_lower, ci_upper = calc_mag_helicity_normalized_welch(df)

            lf = load_lf(au)
            background_mag_field = calc_background_mag_field(lf)
            f_ci, f_ci_obs_para, f_ci_obs_anti = calc_proton_cyclotron_frequency(
                background_mag_field, 
                n_p=mean_densities_interp[i], 
                Vsw=mean_flow_speeds_interp[i]
            )

            mid_au_list.append(au)
            mid_freqs_list.append(np.array(freqs) / f_ci_obs_para)
            mid_hel_list.append(hel_normalized)

    # 3. Bottom Panel (k d_i norm, from helicity_flow_analysis_2.py)
    bot_au_list = []
    bot_freqs_list = []
    bot_hel_list = []
    for au_list in au_lists:
        for i, au in enumerate(au_list):
            lf = load_lf(au)
            df = lf.select(["timestamp", "R", "T", "N"]).collect()
            freqs, hel_normalized, ci_lower, ci_upper = calc_mag_helicity_normalized_welch(df)

            lf = load_lf(au)
            background_mag_field = calc_background_mag_field(lf)
            f_ci, f_ci_obs_para, f_ci_obs_anti = calc_proton_cyclotron_frequency(
                background_mag_field, 
                n_p=mean_densities_interp[i], 
                Vsw=mean_flow_speeds_interp[i]
            )
            f_di = calc_di_frequency(
                n_p=mean_densities_interp[i], 
                Vsw=mean_flow_speeds_interp[i]
            )

            bot_au_list.append(au)
            bot_freqs_list.append(np.array(freqs) / f_di)
            bot_hel_list.append(hel_normalized)

    # Save to HDF5
    with h5py.File(OUTPUT_DIR / 'fig4_data.h5', 'w') as f:
        grp_top = f.create_group('top_panel')
        grp_top.create_dataset('Distance_AU', data=np.array(top_au_list))
        grp_top.create_dataset('Frequency_Hz', data=np.array(top_freqs_list))
        grp_top.create_dataset('Helicity_Norm', data=np.array(top_hel_list))

        grp_mid = f.create_group('middle_panel')
        grp_mid.create_dataset('Distance_AU', data=np.array(mid_au_list))
        grp_mid.create_dataset('Freq_fci_norm', data=np.array(mid_freqs_list))
        grp_mid.create_dataset('Helicity_Norm', data=np.array(mid_hel_list))

        grp_bot = f.create_group('bottom_panel')
        grp_bot.create_dataset('Distance_AU', data=np.array(bot_au_list))
        grp_bot.create_dataset('Freq_di_norm', data=np.array(bot_freqs_list))
        grp_bot.create_dataset('Helicity_Norm', data=np.array(bot_hel_list))
    print("-> Saved plot_data/fig4_data.h5")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calc_figure1()
    calc_figure2()
    calc_figure3()
    calc_figure4()
    print("All plotting data successfully calculated and exported!")

if __name__ == '__main__':
    main()