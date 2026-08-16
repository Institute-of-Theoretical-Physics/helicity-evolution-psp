# Code and Data for "Non-local Evolution of Magnetic Helicity across the MHD-Kinetic Interface"

[![DOI](https://zenodo.org/badge/1335526282.svg)](https://doi.org/10.5281/zenodo.21960159)

This repository contains the Python scripts and processed datasets required to reproduce the figures (Figures 1 through 4) presented in the paper **"Non-local Evolution of Magnetic Helicity across the MHD-Kinetic Interface"**.

To promote open science and ensure full methodological transparency, this repository provides everything from the raw Parker Solar Probe (PSP) preprocessing pipeline to lightweight figure plotting scripts.

## 📊 Levels of Reproducibility

To accommodate different user needs and follow standard heliophysics/astrophysics data processing conventions (where Level 1 corresponds to data processing closest to raw measurements and Level 3 represents high-level derived products), this repository is structured into three levels:

* **Level 1: Raw Data Preprocessing (For Reference)**
The `prep_stepX` scripts demonstrate how the raw L2/L3 CDF files were converted, concatenated, and interpolated into intermediate datasets. **Disclaimer:** These preprocessing scripts are provided primarily for *logical transparency*. They assume a specific local directory structure mirroring official data archives and are not intended as out-of-the-box standalone tools.
* **Level 2: Plot Data Extraction & Calculation**
If you have the intermediate processed data (`.ipc` and `.parquet`), you can run `calculate_plot_data.py` to recalculate the physical quantities and re-extract the lightweight `plot_data/` files (`.csv` and `.h5`).
* **Level 3: Quick Figure Reproduction (Recommended for Most Users)**
The quickest way to reproduce the paper's figures. Lightweight, pre-extracted datasets (`.csv` and `.h5`) are already provided in the `plot_data/` directory. You only need to run the `plot_figureX.py` scripts to instantly generate the final plots.

---

## 📁 Repository Structure

```text
.
├── calculate_plot_data.py              # [Level 2] Calculates & extracts data for plotting
├── fig/                                # Output directory for generated figures
│   ├── Figure1.pdf             
│   ├── Figure2_top.png         
│   ├── Figure2_bottom.png      
│   ├── Figure3.png             
│   ├── Figure4_au_hel_normalized_wavenumber.png
│   ├── Figure4_au_hel_normalized_fci_norm.png
│   ├── Figure4_au_hel_normalized_fci_norm_expand.png
│   ├── Figure4_au_hel_normalized_di_norm.png
│   └── Figure4_au_hel_normalized_di_norm_expand.png
├── plot_data/                          # [Level 3] Lightweight extracted data for plotting
│   ├── fig1_data.csv                   # 1D time-series/distance data
│   ├── fig2_data.h5                    # PSD and Magnetic field data
│   ├── fig3_data.h5                    # 1D Helicity spectra
│   └── fig4_data.h5                    # 2D Helicity colormaps
├── plot_figure1.py                     # [Level 3] Generates Figure 1
├── plot_figure2.py                     # [Level 3] Generates Figure 2
├── plot_figure3.py                     # [Level 3] Generates Figure 3
├── plot_figure4.py                     # [Level 3] Generates Figure 4
├── prep_step0_plasma_and_position.py   # [Level 1] Raw CDF to Parquet/IPC conversion
├── prep_step1_mag_cdf_to_parquet.py    # [Level 1] MAG CDF splitting
├── prep_step2_concat_and_position.py   # [Level 1] Concat and interpolate coordinates
├── prep_step3_generate_isotemporal.py  # [Level 1] Isotemporal 1-hr chunk extraction
├── run_all_preprocessing.sh            # [Level 1] Master shell script for prep steps
└── spice_kernel/               
    └── psp/kernel/mk/
        └── psp-private-mk.tm           # SPICE Meta-Kernel with download instructions

```

---

## 💻 Environment & Dependencies

The code is written in Python 3. To run the scripts, you will need the following libraries:

* `numpy`
* `scipy`
* `matplotlib` (Note: Ensure the **"Myriad Pro"** font is installed on your system for exact typographic reproduction)
* `polars` (Used for fast tabular data manipulation)
* `h5py` (Used for reading/writing multi-dimensional spectral data)
* `astropy`, `sunpy` (For coordinate transformations)
* `spacepy` (For reading `.cdf` files in Level 1)
* `spiceypy` (For SPICE orbit calculations)

You can install the core dependencies via pip:

```bash
pip install numpy scipy matplotlib polars h5py astropy sunpy spacepy spiceypy

```

---

## 🚀 Usage Guide

### Level 3: Reproducing the Figures (Recommended)

You do not need to download any raw satellite data to perform this step. The required data is already included in the `plot_data/` folder.

Simply execute the plotting scripts from the root directory:

```bash
python plot_figure1.py
python plot_figure2.py
python plot_figure3.py
python plot_figure4.py

```

The resulting `.pdf` and `.png` files will be saved in the root directory (or `fig/` directory), matching the exact layouts, axis limits, and annotations found in the paper.

### Level 2: Recalculating the Plot Data

If you wish to re-evaluate the physics parameters (e.g., fractional helicity, Taylor hypothesis parameters, Alfvén Mach numbers), you can run:

```bash
python calculate_plot_data.py

```

*Note: This step requires the intermediate `.parquet` and `.ipc` files, which are too large to host on GitHub. You must generate them via Level 1 or obtain them separately.*

### Level 1: Raw Data Preprocessing

For those intending to trace the data pipeline from the very beginning (Level 1), you can examine the `prep_stepX` scripts and the `run_all_preprocessing.sh` wrapper.

**Important Disclaimers for Level 1:**

1. **Local Directory Assumptions:** The preprocessing scripts assume that the raw Level-2 and Level-3 CDF files (SWEAP SPC, SPAN-I, FIELDS MAG RTN, Ephemeris) are stored in specific local directory trees (e.g., `/data/00/...`). You will need to edit the `GLOBAL CONFIGURATION` paths inside the scripts to match your local environment.
2. **Server Load:** We do not provide automated bulk download scripts (like `wget` one-liners) to prevent unnecessary strain on the official data provider servers (such as CDAWeb or SWEAP archives). We assume users will mirror the required directories locally using their preferred, responsible methods.
3. **SPICE Kernels:** Trajectory data is processed using SPICE. The `.bsp`, `.tls`, and other necessary binary kernels are not included due to size. Please open `spice_kernel/psp/kernel/mk/psp-private-mk.tm` using a text editor; all necessary download URLs and instructions for the required SPICE kernels are documented in the header of that file. Once downloaded, update the meta-kernel paths accordingly.
