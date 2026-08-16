#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Convert FIELDS L2 Mag (RTN) CDF files to split Parquets.
Preserves original logic from source files (cdf_to_parquet_split.py, cdf_to_parquet.py).
"""

import os
import sys
import gc
from pathlib import Path
from spacepy import pycdf
import polars as pl

TARGET_DIR_ROOT = Path("/data/00/wget_sci/data/psp/data/sci/fields/l2/mag_RTN/")
DATA_DIR = Path("./data")


def list_files_in_directory(directory: Path) -> list[str]:
    file_list = [os.path.join(pathname, filename) for pathname, dirnamess, filenames in os.walk(directory) for filename in filenames if filename.endswith("_v02.cdf")]
    file_list.sort()
    return file_list


def read_cdf_to_dataframe(cdf_file: str) -> pl.DataFrame:
    with pycdf.CDF(cdf_file) as f:
        df_mag = pl.from_numpy(f["psp_fld_l2_mag_RTN"][...], schema={"R": pl.Float32, "T": pl.Float32, "N": pl.Float32})
        df_mag = df_mag.with_columns(
            pl.col("R").fill_nan(None).alias("R"),
            pl.col("T").fill_nan(None).alias("T"),
            pl.col("N").fill_nan(None).alias("N")
        )
        df_timestamp = pl.from_numpy(f["epoch_mag_RTN"][...].astype("datetime64[us]"), schema={"timestamp": pl.Datetime})
        df_concat = pl.concat([df_timestamp, df_mag], how="horizontal")
    return df_concat


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    df = df.lazy().drop_nulls().sort("timestamp")
    df = df.with_columns(pl.col("timestamp").diff().alias("time_diff"))
    return df.collect()


def cdf_to_parquet(filenames: list[str], year: int, num_packet: int, total_num_packets: int):
    df = pl.DataFrame()
    num_files = len(filenames)
    num_current = 0
    for filename in filenames:
        df_onefile = read_cdf_to_dataframe(filename)
        df = pl.concat([df, df_onefile], how="vertical")
        num_current += 1
        print(f"loaded ({num_current}/{num_files}) (packet {num_packet}/{total_num_packets}) : {filename}")
    
    df = preprocessing(df)
    out_dir = DATA_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"psp_fld_l2_mag_RTN_{year}_{num_packet:04d}_v02.parquet"
    df.rechunk().write_parquet(out_file, compression="lz4")
    print(f"The DataFrame written to {out_file}")


def cdf_to_parquet_split(year: int, split_size: int = 100):
    target_directory = TARGET_DIR_ROOT / str(year)
    filenames = list_files_in_directory(target_directory)
    filenames_split = [filenames[i:i + split_size] for i in range(0, len(filenames), split_size)]
    num_packet = 0
    for filenames_packet in filenames_split:
        cdf_to_parquet(filenames_packet, year, num_packet, len(filenames_split))
        num_packet += 1
        gc.collect()


def main():
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2020, 2021, 2022, 2023, 2024]
    for year in years:
        cdf_to_parquet_split(year, split_size=100)


if __name__ == '__main__':
    main()