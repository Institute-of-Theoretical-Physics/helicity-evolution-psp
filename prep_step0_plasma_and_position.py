#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0: Process Ephemeris and Plasma (SPC / SPAN-I) CDF files into Parquet / IPC formats.
Preserves original logic from source files (cdf_to_ipc_flow_velocity.py, cdf_to_ipc_position.py, helicity_flow_analysis.py).
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from spacepy import pycdf
import polars as pl
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SPC_L3_CDF_DIR = Path("/data/00/data_sweap_spc_L3/pub/data/sci/sweap/spc/L3/")
SPI_L3_CDF_DIR = Path("/data/00/data_sweap_spc_L3/pub/data/sci/sweap/spi/L3/spi_sf00")
EPHEMERIS_CDF_FILE = Path("/data/00/wget_ephemeris/helio1hr/psp_helio1hr_position_20180813_v01.cdf")

DATA_DIR = Path("./data")


def list_files_in_directory(directory: Path, pattern: str = "*.cdf") -> list[Path]:
    return sorted(directory.rglob(pattern))


# --- Ephemeris CDF to IPC ---
def read_ephemeris_cdf(cdf_file: Path) -> pl.DataFrame:
    with pycdf.CDF(cdf_file.as_posix()) as f:
        df_timestamp = pl.from_numpy(f["Epoch"][...].astype("datetime64[us]"), schema={"timestamp": pl.Datetime})
        df_rad_au = pl.from_numpy(f["RAD_AU"][...], schema={"RAD_AU": pl.Float32})
        df_se_lat = pl.from_numpy(f["SE_LAT"][...], schema={"SE_LAT": pl.Float32})
        df_se_lon = pl.from_numpy(f["SE_LON"][...], schema={"SE_LON": pl.Float32})
        df_concat = pl.concat([df_timestamp, df_rad_au, df_se_lat, df_se_lon], how="horizontal")
        df_concat = df_concat.with_columns(
            pl.col("RAD_AU").fill_nan(None).alias("RAD_AU"),
            pl.col("SE_LAT").fill_nan(None).alias("SE_LAT"),
            pl.col("SE_LON").fill_nan(None).alias("SE_LON")
        )
    return df_concat


def process_ephemeris():
    print("Processing Ephemeris Position CDF...")
    df = read_ephemeris_cdf(EPHEMERIS_CDF_FILE)
    df = df.lazy().drop_nulls().sort("timestamp").collect()
    out_file = DATA_DIR / "psp_helio1hr_position_20180813_v01.ipc"
    df.rechunk().write_ipc(out_file)
    print(f"Saved: {out_file}")


# --- SPC Flow Velocity to IPC ---
def read_spc_flow_cdf(cdf_file: str) -> pl.DataFrame:
    with pycdf.CDF(cdf_file) as f:
        df_timestamp = pl.from_numpy(f["Epoch"][...].astype("datetime64[us]"), schema={"timestamp": pl.Datetime})
        df_flow = pl.from_numpy(f["vp_fit_RTN"][...].T[0], schema={"vp_fit_R": pl.Float32})
        df_flow_unc = pl.from_numpy(f["vp_fit_RTN_uncertainty"][...].T[0], schema={"vp_fit_R_uncertainty": pl.Float32})
        df_concat = pl.concat([df_timestamp, df_flow, df_flow_unc], how="horizontal")
        df_concat = df_concat.with_columns(
            pl.col("vp_fit_R").fill_nan(None).alias("vp_fit_R"),
            pl.col("vp_fit_R_uncertainty").fill_nan(None).alias("vp_fit_R_uncertainty")
        )
    return df_concat


def process_spc_flow():
    print("Processing SPC Flow Velocity CDFs...")
    file_list = [os.path.join(p, f) for p, d, fs in os.walk(SPC_L3_CDF_DIR) for f in fs if f.endswith(".cdf")]
    file_list.sort()
    df = pl.DataFrame({"timestamp": [], "vp_fit_R": [], "vp_fit_R_uncertainty": []}, 
                      schema={"timestamp": pl.Datetime, "vp_fit_R": pl.Float32, "vp_fit_R_uncertainty": pl.Float32})
    for filename in file_list:
        df_onefile = read_spc_flow_cdf(filename)
        df.extend(df_onefile)
    df = df.lazy().drop_nulls().sort("timestamp").collect()
    out_file = DATA_DIR / "psp_sweap_spc_l3.ipc"
    df.rechunk().write_ipc(out_file)
    print(f"Saved: {out_file}")


# --- SPC L3 Moments CDF to Parquet ---
def scan_cdf_spc(path: Path) -> pl.LazyFrame:
    with pycdf.CDF(path.as_posix()) as f:
        epoch = f["Epoch"][...].astype("datetime64[us]")
        vp = f["vp_moment_RTN"][...].T
        vp_high = f["vp_moment_RTN_deltahigh"][...].T
        vp_low = f["vp_moment_RTN_deltalow"][...].T
        cols = [
            pl.from_numpy(epoch, schema={"timestamp": pl.Datetime}),
            pl.from_numpy(f["general_flag"][...].T, schema={"general_flag": pl.Int8}),
            pl.from_numpy(vp[0], schema={"proton_bulk_velocity_moment_R": pl.Float32}),
            pl.from_numpy(vp[1], schema={"proton_bulk_velocity_moment_T": pl.Float32}),
            pl.from_numpy(vp[2], schema={"proton_bulk_velocity_moment_N": pl.Float32}),
            pl.from_numpy(vp_high[0], schema={"proton_bulk_velocity_moment_R_deltahigh": pl.Float32}),
            pl.from_numpy(vp_high[1], schema={"proton_bulk_velocity_moment_T_deltahigh": pl.Float32}),
            pl.from_numpy(vp_high[2], schema={"proton_bulk_velocity_moment_N_deltahigh": pl.Float32}),
            pl.from_numpy(vp_low[0], schema={"proton_bulk_velocity_moment_R_deltalow": pl.Float32}),
            pl.from_numpy(vp_low[1], schema={"proton_bulk_velocity_moment_T_deltalow": pl.Float32}),
            pl.from_numpy(vp_low[2], schema={"proton_bulk_velocity_moment_N_deltalow": pl.Float32}),
            pl.from_numpy(f["np_moment"][...].T, schema={"proton_density_moment": pl.Float32}),
            pl.from_numpy(f["np_moment_deltahigh"][...].T, schema={"proton_density_moment_deltahigh": pl.Float32}),
            pl.from_numpy(f["np_moment_deltalow"][...].T, schema={"proton_density_moment_deltalow": pl.Float32}),
            pl.from_numpy(f["wp_moment"][...].T, schema={"proton_thermal_speed_moment_R": pl.Float32}),
            pl.from_numpy(f["wp_moment_deltahigh"][...].T, schema={"proton_thermal_speed_moment_R_deltahigh": pl.Float32}),
            pl.from_numpy(f["wp_moment_deltalow"][...].T, schema={"proton_thermal_speed_moment_R_deltalow": pl.Float32})
        ]
        df = pl.concat(cols, how="horizontal")
    lf = df.lazy()
    schema = lf.collect_schema()
    cols = [c for c, dtype in schema.items() if dtype.is_numeric()]
    lf = lf.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
    return lf


def process_spc_moments():
    print("Processing SPC L3 Moment CDFs...")
    filelist = list_files_in_directory(SPC_L3_CDF_DIR, pattern="*.cdf")
    with ProcessPoolExecutor() as executor:
        lfs = list(tqdm(executor.map(scan_cdf_spc, filelist), total=len(filelist)))
    lf = pl.concat(lfs).filter(pl.col("general_flag") == 0).drop_nulls().sort("timestamp")
    lf = lf.with_columns(pl.col("timestamp").diff().alias("time_diff"))
    df = lf.collect()
    out_file = DATA_DIR / "psp_sweap_spc_l3_vp_moment.parquet"
    df.rechunk().write_parquet(out_file, statistics="full", compression="zstd")
    print(f"Saved: {out_file}")


# --- SPAN-I L3 Moments CDF to Parquet ---
def quat_to_matrix(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def add_t_tensor_rtn(df: pl.DataFrame, rotmat_sc_inst: np.ndarray) -> pl.DataFrame:
    t_tensor_inst = df.select(["T_xx", "T_yy", "T_zz", "T_xy", "T_xz", "T_yz"]).to_numpy()
    t_tensor_rtn = np.empty((t_tensor_inst.shape[0], 3, 3), dtype=t_tensor_inst.dtype)
    for i, row in enumerate(df.iter_rows(named=True)):
        qw, qx, qy, qz = row["quat_sc_to_rtn_w"], row["quat_sc_to_rtn_x"], row["quat_sc_to_rtn_y"], row["quat_sc_to_rtn_z"]
        if any(v is None or np.isnan(v) for v in [qw, qx, qy, qz]):
            t_tensor_rtn[i] = None
            continue
        R_sc2rtn = quat_to_matrix(qw, qx, qy, qz)
        R_inst2rtn = R_sc2rtn @ rotmat_sc_inst
        T_inst = np.array([
            [row["T_xx"], row["T_xy"], row["T_xz"]],
            [row["T_xy"], row["T_yy"], row["T_yz"]],
            [row["T_xz"], row["T_yz"], row["T_zz"]]
        ])
        if any(v is None for v in T_inst.flatten()):
            t_tensor_rtn[i] = None
            continue
        t_tensor_rtn[i] = R_inst2rtn @ T_inst @ R_inst2rtn.T

    df = df.with_columns([
        pl.Series("t_tensor_rtn_rr", t_tensor_rtn[:,0,0], dtype=pl.Float32).fill_nan(None),
        pl.Series("t_tensor_rtn_tt", t_tensor_rtn[:,1,1], dtype=pl.Float32).fill_nan(None),
        pl.Series("t_tensor_rtn_nn", t_tensor_rtn[:,2,2], dtype=pl.Float32).fill_nan(None),
        pl.Series("t_tensor_rtn_rt", t_tensor_rtn[:,0,1], dtype=pl.Float32).fill_nan(None),
        pl.Series("t_tensor_rtn_tn", t_tensor_rtn[:,1,2], dtype=pl.Float32).fill_nan(None),
        pl.Series("t_tensor_rtn_rn", t_tensor_rtn[:,0,2], dtype=pl.Float32).fill_nan(None),
    ])
    return df


def scan_cdf_spi(path: Path):
    with pycdf.CDF(path.as_posix()) as f:
        epoch = f["Epoch"][...].astype("datetime64[us]")
        cols = [
            pl.from_numpy(epoch, schema={"timestamp": pl.Datetime}),
            pl.from_numpy(f["QUALITY_FLAG"][...].T, schema={"quality_flag": pl.UInt16}),
            pl.from_numpy(f["DENS"][...].T, schema={"partial_density": pl.Float32}),
            pl.from_numpy(f["VEL_RTN_SUN"][...].T[0], schema={"vel_rtn_sun_R": pl.Float32}),
            pl.from_numpy(f["VEL_RTN_SUN"][...].T[1], schema={"vel_rtn_sun_T": pl.Float32}),
            pl.from_numpy(f["VEL_RTN_SUN"][...].T[2], schema={"vel_rtn_sun_N": pl.Float32}),
            pl.from_numpy(f["SC_VEL_RTN_SUN"][...].T[0], schema={"sc_vel_rtn_sun_R": pl.Float32}),
            pl.from_numpy(f["SC_VEL_RTN_SUN"][...].T[1], schema={"sc_vel_rtn_sun_T": pl.Float32}),
            pl.from_numpy(f["SC_VEL_RTN_SUN"][...].T[2], schema={"sc_vel_rtn_sun_N": pl.Float32}),
            pl.from_numpy(f["VEL_SC"][...].T[0], schema={"vel_sc_X": pl.Float32}),
            pl.from_numpy(f["VEL_SC"][...].T[1], schema={"vel_sc_Y": pl.Float32}),
            pl.from_numpy(f["VEL_SC"][...].T[2], schema={"vel_sc_Z": pl.Float32}),
            pl.from_numpy(f["VEL_INST"][...].T[0], schema={"vel_inst_X": pl.Float32}),
            pl.from_numpy(f["VEL_INST"][...].T[1], schema={"vel_inst_Y": pl.Float32}),
            pl.from_numpy(f["VEL_INST"][...].T[2], schema={"vel_inst_Z": pl.Float32}),
            pl.from_numpy(f["T_TENSOR_INST"][...].T[0], schema={"T_xx": pl.Float32}),
            pl.from_numpy(f["T_TENSOR_INST"][...].T[1], schema={"T_yy": pl.Float32}),
            pl.from_numpy(f["T_TENSOR_INST"][...].T[2], schema={"T_zz": pl.Float32}),
            pl.from_numpy(f["T_TENSOR_INST"][...].T[3], schema={"T_xy": pl.Float32}),
            pl.from_numpy(f["T_TENSOR_INST"][...].T[4], schema={"T_xz": pl.Float32}),
            pl.from_numpy(f["T_TENSOR_INST"][...].T[5], schema={"T_yz": pl.Float32}),
            pl.from_numpy(f["TEMP"][...].T, schema={"temperature": pl.Float32}),
            pl.from_numpy(f["QUAT_SC_TO_RTN"][...].T[0], schema={"quat_sc_to_rtn_w": pl.Float64}),
            pl.from_numpy(f["QUAT_SC_TO_RTN"][...].T[1], schema={"quat_sc_to_rtn_x": pl.Float64}),
            pl.from_numpy(f["QUAT_SC_TO_RTN"][...].T[2], schema={"quat_sc_to_rtn_y": pl.Float64}),
            pl.from_numpy(f["QUAT_SC_TO_RTN"][...].T[3], schema={"quat_sc_to_rtn_z": pl.Float64})
        ]
        df = pl.concat(cols, how="horizontal")
        rotmat_sc_inst = f["ROTMAT_SC_INST"][...].copy()
    lf = df.lazy()
    schema = lf.collect_schema()
    cols = [c for c, dtype in schema.items() if dtype.is_numeric()]
    lf = lf.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
    return lf, rotmat_sc_inst


def process_spi_moments():
    print("Processing SPAN-I L3 Moment CDFs...")
    filelist = list_files_in_directory(SPI_L3_CDF_DIR, pattern="*_v04.cdf")
    with ProcessPoolExecutor() as executor:
        result = list(tqdm(executor.map(scan_cdf_spi, filelist), total=len(filelist)))
        lfs, rotmats = zip(*result)
    
    drop_mask = (
        ((pl.col("quality_flag") & (1 << 0)) != 0) |
        ((pl.col("quality_flag") & (1 << 3)) != 0) |
        ((pl.col("quality_flag") & (1 << 8)) != 0) |
        ((pl.col("quality_flag") & (1 << 10)) != 0) |
        ((pl.col("quality_flag") & (1 << 11)) != 0)
    )
    lf = pl.concat(lfs).filter(~drop_mask)
    df = lf.collect()
    df = add_t_tensor_rtn(df, rotmats[0])
    
    lf = df.lazy().drop([
        "vel_sc_X", "vel_sc_Y", "vel_sc_Z", "vel_inst_X", "vel_inst_Y", "vel_inst_Z",
        "sc_vel_rtn_sun_R", "sc_vel_rtn_sun_T", "sc_vel_rtn_sun_N",
        "T_xx", "T_yy", "T_zz", "T_xy", "T_xz", "T_yz",
        "quat_sc_to_rtn_w", "quat_sc_to_rtn_x", "quat_sc_to_rtn_y", "quat_sc_to_rtn_z"
    ]).drop_nulls().sort("timestamp")
    lf = lf.with_columns(pl.col("timestamp").diff().alias("time_diff"))
    
    df = lf.collect()
    out_file = DATA_DIR / "psp_swp_spi_sf00_L3_mom_v04.parquet"
    df.rechunk().write_parquet(out_file, statistics="full", compression="zstd")
    print(f"Saved: {out_file}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    process_ephemeris()
    process_spc_flow()
    process_spc_moments()
    process_spi_moments()


if __name__ == '__main__':
    main()