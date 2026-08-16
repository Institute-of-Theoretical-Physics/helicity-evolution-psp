#!/usr/bin/env bash
# ==============================================================================
# Master Preprocessing Pipeline Script
# Executes Steps 0 through 3 sequentially to generate all required datasets.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "=== STEP 0: Processing Plasma Moments and Ephemeris ==="
python3 prep_step0_plasma_and_position.py

echo "=== STEP 1: Converting FIELDS L2 Mag CDFs to Split Parquets ==="
python3 prep_step1_mag_cdf_to_parquet.py 2021 2022 2023 2024

echo "=== STEP 2: Concatenating Parquets and Interpolating Position ==="
python3 prep_step2_concat_and_position.py 2021 2022 2023 2024

echo "=== STEP 3: Extracting Isotemporal Time Series (0.100 - 0.782 AU) ==="
# 1. First Range: 0.100 to 0.160 AU (interval = 0.003414 s)
for val in $(seq 0.100 0.002 0.160)
do
    echo "Extracting: ${val} au (interval: 0.003414 s)"
    python3 prep_step3_generate_isotemporal.py $val 0.003414
done

# 2. Second Range: 0.162 to 0.260 AU (interval = 0.006827 s)
for val in $(seq 0.162 0.002 0.260)
do
    echo "Extracting: ${val} au (interval: 0.006827 s)"
    python3 prep_step3_generate_isotemporal.py $val 0.006827
done

# 3. Third Range: 0.262 to 0.782 AU (interval = 0.109227 s, suffix-less format)
for val in $(seq 0.262 0.002 0.782)
do
    echo "Extracting: ${val} au (interval: 0.109227 s)"
    python3 prep_step3_generate_isotemporal.py $val None
done

# 4. Discrete distances for Fig 3 (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7823)
for val in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.7823
do
    echo "Extracting discrete Fig3 distance: ${val} au"
    python3 prep_step3_generate_isotemporal.py $val None
done

echo "=================================================================="
echo " All Preprocessing Steps Completed Successfully!"
echo " Next step: Run calculate_plot_data.py to extract figure datasets."
echo "=================================================================="