#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reproduce Figure 2 (Top, Middle, and Bottom panels).
Requirements: h5py, matplotlib, numpy
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

def plot_figure2_top(f):
    """Plot the top and middle panels (Magnetic field intensity & PSD Colormap)"""
    print("Plotting Figure 2 (Top/Middle panels)...")
    
    # 1. Load data from HDF5
    au_list_top = f['top_panel']['Distance_AU'][:]
    freqs_top = f['top_panel']['Frequency_Hz'][:]
    psd_cmap = f['top_panel']['PSD_Colormap'][:]
    B_mean = f['top_panel']['B_Mean'][:]
    B_std = f['top_panel']['B_Std'][:]

    # [Safety Check] 
    # If the HDF5 extraction was placed inside a loop, the length of au_list_top
    # might be larger than the colormap data. Match the lengths from the tail if necessary.
    n_data = psd_cmap.shape[0]
    if len(au_list_top) != n_data:
        au_list_top = au_list_top[-n_data:]
        B_mean = B_mean[-n_data:]
        B_std = B_std[-n_data:]

    # Create 2D mesh for pcolormesh
    au_mesh = np.tile(au_list_top, (freqs_top.shape[1], 1)).T

    # 2. Plotting Setup
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.4*0.6), sharex="col", 
                            gridspec_kw={'hspace': 0.05, 'height_ratios': [1, 2]}, 
                            constrained_layout=False)

    # 3. Top Panel: Magnetic Field Intensity
    axs[0].plot(au_list_top, B_mean, '-', color='black', linewidth=1)
    axs[0].fill_between(au_list_top, B_mean - B_std, B_mean + B_std, facecolor='#999999', linewidth=0.5)
    axs[0].set_xlim(0.100, 0.782)
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'$|\boldsymbol{B}|$ [nT]', fontweight='bold')

    # 4. Middle Panel: PSD Colormap
    mesh = axs[1].pcolormesh(au_mesh, freqs_top, psd_cmap, cmap="Greys", norm=LogNorm(vmin=1e-5, vmax=1e1))
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-1, 145)
    axs[1].set_ylabel('Frequency [Hz]', fontweight='bold')
    axs[1].set_xlabel('Distance from the sun [au]', fontweight='bold')

    # 5. Formatting axes
    for a in axs:
        for spine in a.spines.values(): 
            spine.set_linewidth(1.5)
        a.tick_params(direction='out', width=1.5, length=5)
        for label in a.get_yticklabels(): 
            label.set_fontweight('bold')
    for label in axs[1].get_xticklabels(): 
        label.set_fontweight('bold')

    # Colorbar
    pos = axs[1].get_position()
    cax = fig.add_axes([pos.x1 - 0.05, pos.y0 + 0.10, 0.03, pos.height])
    pp = plt.colorbar(mesh, cax=cax, orientation="vertical", fraction=0.05, pad=0.02)
    pp.set_label("PSD [nT$^{2}$Hz$^{-1}$]", fontweight='bold')
    
    fig.subplots_adjust(top=0.995, left=0.17, right=0.83, bottom=0.21)
    fig.align_ylabels(axs)
    
    plt.savefig("Figure2_top.png", dpi=600)
    print("-> Saved Figure2_top.png")
    plt.close(fig)


def plot_figure2_bottom(f):
    """Plot the bottom panels (1D Spectra at specific distances)"""
    print("Plotting Figure 2 (Bottom panels)...")
    
    # 1. Load data from HDF5
    au_list = f['bottom_panel']['Distance_AU'][:]
    freqs_list = f['bottom_panel']['Frequency_Hz'][:]
    psd_list = f['bottom_panel']['PSD'][:]
    ci_lower_list = f['bottom_panel']['CI_Lower'][:]
    ci_upper_list = f['bottom_panel']['CI_Upper'][:]

    n = len(au_list)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, 3.4 * 1.2), sharex="col", gridspec_kw={'hspace': 0})
    
    # Inertial range coefficients used in original code
    coef_inert = [2e0, 4e-1, 7e-2, 3e-2, 1e-2, 7e-3, 1e-3, 1e-3]
    x_inert = np.logspace(-3, 2, 1000)

    # 2. Plotting each panel
    for i, ax in enumerate(axes):
        ax.plot(freqs_list[i], psd_list[i], color="#000000", linewidth=1.0)
        ax.fill_between(freqs_list[i], ci_lower_list[i], ci_upper_list[i], facecolor='#999999', linewidth=0.5)
        
        # Plot inertial range
        y_inert = coef_inert[i] * x_inert**(-5/3)
        ax.plot(x_inert, y_inert, color="#0526FF", linewidth=1.0, linestyle='--')
        
        # Formatting
        ax.set_xlim(3e-3, 2e2)
        ax.set_ylim(1e-5, 1e2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(direction='in', which='both', width=1.5, length=3)
        ax.tick_params(axis='x', which='minor', bottom=False)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(8)
            label.set_horizontalalignment('left')
            label.set_x(-0.07)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            
        ax.set_yticks([1e-3, 1e0])
        ax.set_yticklabels(['$\\mathdefault{10^{-3}}$', '$\\mathdefault{10^{0}}$'])
        ax.text(0.99, 0.9, f"{au_list[i]} au", transform=ax.transAxes, ha='right', va='top', fontsize=10, fontweight='bold')

    axes[-1].set_xlabel("Frequency [Hz]", fontsize=10, fontweight='bold')
    axes[n//2].set_ylabel("PSD [nT$^2$ Hz$^{-1}$]", fontsize=10, fontweight='bold')
    axes[n//2].yaxis.set_label_coords(-0.10, 0.9)
    
    fig.subplots_adjust(top=0.99, left=0.14, bottom=0.09, right=0.99)
    plt.savefig("Figure2_bottom.png", dpi=600)
    print("-> Saved Figure2_bottom.png")
    plt.close(fig)

def main():
    data_path = "plot_data/fig2_data.h5"
    
    # Configure Matplotlib globally
    plt.style.use('default')
    plt.rcParams["font.family"] = "Myriad Pro"
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = 10

    try:
        with h5py.File(data_path, 'r') as f:
            if 'top_panel' in f:
                plot_figure2_top(f)
            else:
                print("Error: 'top_panel' group not found in HDF5.")
                
            if 'bottom_panel' in f:
                plot_figure2_bottom(f)
            else:
                print("Error: 'bottom_panel' group not found in HDF5.")
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")

if __name__ == "__main__":
    main()