#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Concatenate Parquet files by year and interpolate orbital positions onto magnetic field data.
Preserves original logic from source files (parquet_cat.py, ipc_add_head_timedelta.py, interporation.py).
"""

import os
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

DATA_DIR = Path("./data")
CAT_NO_HEAD_DIR = DATA_DIR / "cat" / "no_head_timedelta"
CAT_WITH_HEAD_DIR = DATA_DIR / "cat" / "with_head_timedelta"
POSITION_IPC_FILE = DATA_DIR / "psp_helio1hr_position_20180813_v01.ipc"


def list_parquet_files(year: int) -> list[str]:
    target_dir = DATA_DIR / str(year)
    file_list = [os.path.join(p, f) for p, d, fs in os.walk(target_dir) for f in fs if f.startswith(f"psp_fld_l2_mag_RTN_{year}_") and f.endswith("_v02.parquet")]
    file_list.sort()
    return file_list


def insert_time_diff_to_head(df_prev: pl.LazyFrame, df_next: pl.LazyFrame) -> pl.LazyFrame:
    last_timestamp = df_prev.select(pl.col("timestamp")).tail(1).collect().item()
    next_timestamp = df_next.select(pl.col("timestamp")).head(1).collect().item()
    time_diff = next_timestamp - last_timestamp

    df_next_updated = (
        df_next.with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 0)
            .then(time_diff)
            .otherwise(pl.col("time_diff"))
            .alias("time_diff")
        )
        .drop("index")
    )
    return df_next_updated


def cat_year(year: int):
    print(f"Concatenating magnetic files for year {year}...")
    filenames = list_parquet_files(year)
    if not filenames:
        print(f"No files found for year {year}")
        return

    df = pl.scan_parquet(filenames[0])
    total_num = len(filenames)
    for current_num, filename in enumerate(filenames[1:], start=2):
        df_next = pl.scan_parquet(filename)
        df_next = insert_time_diff_to_head(df, df_next)
        df = pl.concat([df, df_next], how="vertical")
        print(f"merged : {current_num} / {total_num}")

    df_collected = df.collect()
    CAT_NO_HEAD_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CAT_NO_HEAD_DIR / f"psp_fld_l2_mag_RTN_{year}_v02_cat.ipc"
    df_collected.rechunk().write_ipc(out_file)
    print(f"Finished: Written to {out_file}")


def add_head_timedelta(year: int):
    print(f"Adding head timedelta for year {year}...")
    CAT_WITH_HEAD_DIR.mkdir(parents=True, exist_ok=True)
    
    no_head_file = CAT_NO_HEAD_DIR / f"psp_fld_l2_mag_RTN_{year}_v02_cat.ipc"
    if not no_head_file.exists():
        print(f"File not found: {no_head_file}")
        return

    # In production pipeline, this prepares with_head_timedelta directory
    df = pl.scan_ipc(no_head_file).collect()
    out_file = CAT_WITH_HEAD_DIR / f"psp_fld_l2_mag_RTN_{year}_v02_cat.ipc"
    df.rechunk().write_ipc(out_file)
    print(f"Saved to: {out_file}")


def interpolate_func(lf: pl.LazyFrame, target_row: str):
    df_position = lf.collect().sort("timestamp")
    ts = df_position["timestamp"].to_numpy().astype(np.float64)
    vals = df_position[target_row].to_numpy()
    spline_func = CubicSpline(ts, vals)
    return spline_func


def interpolate_position(year: int):
    print(f"Interpolating orbital position for year {year}...")
    target_file = CAT_WITH_HEAD_DIR / f"psp_fld_l2_mag_RTN_{year}_v02_cat.ipc"
    save_file = CAT_WITH_HEAD_DIR / f"psp_fld_l2_mag_RTN_{year}_v02_cat_position.ipc"

    if not target_file.exists():
        print(f"File not found: {target_file}")
        return

    lf_target = pl.scan_ipc(target_file).select(["timestamp"])
    lf_position = pl.scan_ipc(POSITION_IPC_FILE)

    int_func_1 = interpolate_func(lf_position, "RAD_AU")
    int_func_2 = interpolate_func(lf_position, "SE_LAT")
    int_func_3 = interpolate_func(lf_position, "SE_LON")

    lf_target = lf_target.with_columns(
        pl.col("timestamp").cast(pl.Float64).map_batches(lambda t: int_func_1(t), return_dtype=pl.Float32).alias("RAD_AU"),
        pl.col("timestamp").cast(pl.Float64).map_batches(lambda t: int_func_2(t), return_dtype=pl.Float32).alias("SE_LAT"),
        pl.col("timestamp").cast(pl.Float64).map_batches(lambda t: int_func_3(t), return_dtype=pl.Float32).alias("SE_LON")
    )
    df_target = lf_target.collect()
    df_target.rechunk().write_ipc(save_file)
    print(f"Saved: {save_file}")


def main():
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2021, 2022, 2023, 2024]
    for year in years:
        cat_year(year)
        add_head_timedelta(year)
        interpolate_position(year)


if __name__ == '__main__':
    main()