#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Extract 1-hour isotemporal time series (01au_isotemporal_*.ipc) per heliocentric distance.
Preserves original logic from source files (interpolation_each0001au.py, interpolation_each01au.py).
"""

import sys
import gc
import datetime
from pathlib import Path
import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

DATA_DIR = Path("./data")
ON_WORKING_0001_DIR = DATA_DIR / "on_working" / "0_001au"
ON_WORKING_DIR = DATA_DIR / "on_working"

TARGET_FILENAME = DATA_DIR / "cat" / "with_head_timedelta" / "psp_fld_l2_mag_RTN_2021_v02_cat.ipc"
POSITION_FILENAME = DATA_DIR / "cat" / "with_head_timedelta" / "psp_fld_l2_mag_RTN_2021_v02_cat_position.ipc"


def filter_timedelta_position(lf: pl.LazyFrame, rad_au_th: float) -> pl.LazyFrame:
    lf = lf.with_columns(
        (pl.col("timestamp") - pl.col("timestamp").min()).alias("time_delta")
    )
    lf = lf.with_columns(
        (pl.col("time_delta").cast(pl.Float64) / 1000000.0).cast(pl.Float64).alias("time_delta_seconds")
    )
    lf = lf.drop("time_delta")
    lf = lf.with_columns(
        (pl.col("time_delta_seconds") - pl.col("time_delta_seconds").shift(1)).cast(pl.Float64).alias("diff")
    )
    lf = lf.filter((pl.col("RAD_AU") < rad_au_th))
    lf = lf.filter((pl.col("timestamp") - pl.col("timestamp").min() < datetime.timedelta(hours=1)) & 
                   (pl.col("timestamp") - pl.col("timestamp").min() >= datetime.timedelta(hours=0)))

    lf = lf.filter(pl.col("diff").is_not_null())
    lf = lf.drop("RAD_AU").drop("time_delta_seconds").drop("diff").head(1)
    return lf


def interpolate_func(ts: np.ndarray, df_org: pl.DataFrame, target_row: str):
    vals = df_org[target_row].to_numpy()
    spline_func = CubicSpline(ts, vals)
    return spline_func


def get_interpolate_func(lf: pl.LazyFrame, target_row_1: str, target_row_2: str, target_row_3: str):
    df = lf.select("timestamp", target_row_1, target_row_2, target_row_3).collect().sort("timestamp")
    ts = df["timestamp"].to_numpy().astype(np.float64)
    num_ts = ts.shape[0]
    interp_func_1 = interpolate_func(ts, df, target_row_1)
    interp_func_2 = interpolate_func(ts, df, target_row_2)
    interp_func_3 = interpolate_func(ts, df, target_row_3)
    return interp_func_1, interp_func_2, interp_func_3, num_ts


def timestamp_isotemporal(lf: pl.LazyFrame, interval: float) -> pl.DataFrame:
    result_df = lf.select([
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp")
    ]).collect()
    min_ts = np.datetime64(result_df["min_timestamp"][0])
    max_ts = np.datetime64(result_df["max_timestamp"][0])
    interval_td = np.timedelta64(int(interval * 1e6), "us")

    list_ts = np.arange(min_ts, max_ts, interval_td).astype(np.datetime64)
    df_ts = pl.from_numpy(list_ts, schema={"timestamp": pl.Datetime("us")})
    return df_ts


def extract_isotemporal_chunk(au: float, interval: float):
    ON_WORKING_0001_DIR.mkdir(parents=True, exist_ok=True)
    
    if interval is None:
        # Suffix-less format for Fig 3 (0.1 - 0.7823 au) or far distances
        save_filename = ON_WORKING_DIR / f"01au_isotemporal_{au}.ipc" if au in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7823] else ON_WORKING_0001_DIR / f"01au_isotemporal_{au:.3f}.ipc"
        interval = 0.109227
    else:
        save_filename = ON_WORKING_0001_DIR / f"01au_isotemporal_{au:.3f}_{interval:.6f}.ipc"

    save_num_timestamp_filename = ON_WORKING_0001_DIR / "num_timestamp_starttimestamp_list.txt"
    
    target_row_1, target_row_2, target_row_3 = "R", "T", "N"
    datetime_from = datetime.datetime(2021, 9, 29, 0, 0, 0)
    datetime_to = datetime.datetime(2021, 11, 22, 0, 0, 0)

    lf_position = pl.scan_ipc(POSITION_FILENAME)
    lf_position = lf_position.filter(pl.col("timestamp").is_between(datetime_from, datetime_to)).drop("SE_LAT").drop("SE_LON")
    lf_position = filter_timedelta_position(lf_position, au)
    timestamp_th = lf_position.collect()["timestamp"][0]

    lf_org = pl.scan_ipc(TARGET_FILENAME)
    lf_org = lf_org.filter(pl.col("timestamp").is_between(timestamp_th, timestamp_th + datetime.timedelta(hours=1)))
    df_ts = timestamp_isotemporal(lf_org, interval)
    interp_func_1, interp_func_2, interp_func_3, num_ts = get_interpolate_func(lf_org, target_row_1, target_row_2, target_row_3)
    del lf_org
    gc.collect()

    lf_isotemp = df_ts.lazy()
    lf_isotemp = lf_isotemp.with_columns([
        pl.col("timestamp").map_batches(lambda ts: interp_func_1(ts.to_numpy().astype(np.float64)).astype(np.float32)).alias(target_row_1),
        pl.col("timestamp").map_batches(lambda ts: interp_func_2(ts.to_numpy().astype(np.float64)).astype(np.float32)).alias(target_row_2),
        pl.col("timestamp").map_batches(lambda ts: interp_func_3(ts.to_numpy().astype(np.float64)).astype(np.float32)).alias(target_row_3)
    ])
    df_isotemp = lf_isotemp.collect()
    df_isotemp.rechunk().write_ipc(save_filename)

    with open(save_num_timestamp_filename, "a") as f:
        f.write(f"{au}, {num_ts}, {timestamp_th}\n")
    print(f"Saved: {save_filename}")


def main():
    if len(sys.argv) >= 3:
        au = float(sys.argv[1])
        interval = float(sys.argv[2]) if sys.argv[2] != "None" else None
        extract_isotemporal_chunk(au, interval)
    else:
        # Default execution for 0.100 to 0.160 au
        for val in np.arange(0.114, 0.162, 0.002):
            extract_isotemporal_chunk(val, 0.003414)
        for val in np.arange(0.162, 0.262, 0.002):
            extract_isotemporal_chunk(val, 0.006827)


if __name__ == '__main__':
    main()