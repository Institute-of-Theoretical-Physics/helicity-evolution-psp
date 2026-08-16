#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reproduce Figure 4 (Top, Middle, Bottom, and Expands).
Requirements: h5py, matplotlib, numpy
Outputs 5 independent figure files.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

def get_custom_cmap():
    """Generates the custom seismic colormap with a blended white center"""
    N = 256
    org_cmap = plt.get_cmap("seismic")
    colors = org_cmap(np.linspace(0, 1, N))
    half_width = int((0.2 * N) / 2)
    center = N // 2
    for i in range(center - half_width, center + half_width):
        blend = 1 - abs(i - center) / half_width
        colors[i] = blend * np.array([1, 1, 1, 1]) + (1 - blend) * colors[i]
    return mcolors.LinearSegmentedColormap.from_list("custom_seismic", colors, N=N)

def plot_top(f, cmap, norm):
    print("Plotting Figure 4 (Top panel)...")
    au_list = f['top_panel']['Distance_AU'][:]
    freqs = f['top_panel']['Frequency_Hz'][:]
    hel = f['top_panel']['Helicity_Norm'][:]
    
    au_mesh = np.tile(au_list, (freqs.shape[1], 1)).T
    condensed_font = mpl.font_manager.FontProperties(family="Myriad Pro", stretch="condensed", weight="bold")

    fig, ax = plt.subplots(figsize=(3.4, 3.4*0.8), constrained_layout=True)
    mesh = ax.pcolormesh(au_mesh, freqs, hel, cmap=cmap, norm=norm, edgecolors='face', shading='auto', antialiased=False)
    
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 145)
    ax.set_xlim(0.100, 0.782)
    ax.set_facecolor('black')

    # Formatting
    ax.tick_params(direction='out', which='both', width=1.5, length=3)
    ax.tick_params(axis='x', which='minor', bottom=False)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(10); label.set_horizontalalignment('left'); label.set_x(-0.10)
    for label in ax.get_xticklabels(): label.set_fontweight('bold')

    # Yellow Curve
    curve_list = 2.0e1 * np.exp(- 9.0 * au_list)
    mask = (curve_list > 0) & (au_list < 0.5)
    ax.plot(au_list[mask], curve_list[mask], color='yellow', linewidth=3.0, linestyle='-', zorder=10)

    # Secondary Axis
    flow_speed = 441.4
    secax = ax.secondary_yaxis('right', functions=(lambda f: 2 * np.pi * f / flow_speed, lambda k: k * flow_speed / (2 * np.pi)))
    secax.set_ylabel('Wavenumber (fixed flow speed) [rad/km]', fontproperties=condensed_font)
    for label in secax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(8)
    secax.tick_params(direction='out', which='both', width=1.5, length=3)
    secax.tick_params(axis='x', which='minor', bottom=False)

    # Colorbar
    pp = fig.colorbar(mesh, orientation="horizontal", location="top", shrink=0.8, aspect=30, pad=0.02)
    pp.set_label(r"Normalized magnietic helicity $\boldsymbol{\sigma}_{\mathbf{m}}$", fontproperties=condensed_font, fontsize=8)
    pp.outline.set_linewidth(1.5)
    pp.ax.xaxis.set_ticks_position('bottom')
    pp.ax.xaxis.set_tick_params(width=1.5, length=1.5)
    for label in pp.ax.get_xticklabels():
        label.set_fontweight('bold'); label.set_fontsize(8)

    ax.set_xlabel('Distance from the sun [au]', fontweight='bold')
    ax.set_ylabel('Frequency [Hz]', fontweight='bold')

    plt.savefig("Figure4_au_hel_normalized_wavenumber.png", dpi=600)
    print("-> Saved Figure4_au_hel_normalized_wavenumber (.pdf / .png)")
    plt.close(fig)

def plot_middle(f, cmap, norm):
    print("Plotting Figure 4 (Middle panel & Expand)...")
    au_list = f['middle_panel']['Distance_AU'][:]
    freqs = f['middle_panel']['Freq_fci_norm'][:]
    hel = f['middle_panel']['Helicity_Norm'][:]
    au_mesh = np.tile(au_list, (freqs.shape[1], 1)).T

    # 1. Normal Plot
    fig, ax = plt.subplots(figsize=(3.4, 3.4*0.65), constrained_layout=False)
    ax.pcolormesh(au_mesh, freqs, hel, cmap=cmap, norm=norm, edgecolors='face', shading='auto', antialiased=False)
    
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e1)
    ax.set_xlim(0.100, 0.782)
    ax.set_facecolor('black')

    ax.tick_params(direction='out', which='both', width=1.5, length=3)
    ax.tick_params(axis='x', which='minor', bottom=False)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(10); label.set_horizontalalignment('left'); label.set_x(-0.10)
    for label in ax.get_xticklabels(): label.set_fontweight('bold')

    ax.set_xlabel('Distance from the sun [au]', fontweight='bold')
    ax.set_ylabel(r'Normalized frequency $\boldsymbol{f/}\boldsymbol{f}_{\mathbf{ci}}$', fontweight='bold')
    ax.axhline(y=1.0, color='black', linestyle='--', zorder=10, linewidth=1.5, dashes=[2, 2])

    fig.subplots_adjust(top=0.95, right=0.845, left=0.16, bottom=0.18)
    plt.savefig("Figure4_au_hel_normalized_fci_norm.png", dpi=600)
    print("-> Saved Figure4_au_hel_normalized_fci_norm (.pdf / .png)")
    plt.close(fig)

    # 2. Expand Plot
    fig, ax = plt.subplots(figsize=(3.4*0.5, 3.4*0.65), constrained_layout=False)
    ax.pcolormesh(au_mesh, freqs, hel, cmap=cmap, norm=norm, edgecolors='face', shading='auto', antialiased=False)
    
    ax.set_yscale('log')
    ax.set_ylim(3.0e-1, 2.0e0)
    ax.set_xlim(0.100, 0.200)
    ax.set_facecolor('black')

    ax.tick_params(direction='out', which='both', width=1.5, length=3)
    ax.tick_params(axis='x', which='minor', bottom=False)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)

    ax.set_yticks([0.3, 1.0, 2.0])
    ax.set_yticklabels(['0.3', '1.0', '2.0'])
    ax.tick_params(axis='y', which='minor', labelleft=False)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(10); label.set_horizontalalignment('left'); label.set_x(-0.12)
    for label in ax.get_xticklabels(): label.set_fontweight('bold')

    ax.axhline(y=1.0, color='black', linestyle='--', zorder=10, linewidth=1.5, dashes=[2, 2])

    fig.subplots_adjust(top=0.98, right=0.92, left=0.15, bottom=0.09)
    plt.savefig("Figure4_au_hel_normalized_fci_norm_expand.png", dpi=600)
    print("-> Saved Figure4_au_hel_normalized_fci_norm_expand (.png)")
    plt.close(fig)

def plot_bottom(f, cmap, norm):
    print("Plotting Figure 4 (Bottom panel & Expand)...")
    au_list = f['bottom_panel']['Distance_AU'][:]
    freqs = f['bottom_panel']['Freq_di_norm'][:]
    hel = f['bottom_panel']['Helicity_Norm'][:]
    au_mesh = np.tile(au_list, (freqs.shape[1], 1)).T

    # 1. Normal Plot
    fig, ax = plt.subplots(figsize=(3.4, 3.4*0.65), constrained_layout=False)
    ax.pcolormesh(au_mesh, freqs, hel, cmap=cmap, norm=norm, edgecolors='face', shading='auto', antialiased=False)
    
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e1)
    ax.set_xlim(0.100, 0.782)
    ax.set_facecolor('black')

    ax.tick_params(direction='out', which='both', width=1.5, length=3)
    ax.tick_params(axis='x', which='minor', bottom=False)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(10); label.set_horizontalalignment('left'); label.set_x(-0.10)
    for label in ax.get_xticklabels(): label.set_fontweight('bold')

    ax.set_xlabel('Distance from the sun [au]', fontweight='bold')
    ax.set_ylabel(r'Normalized wavenumber $\boldsymbol{k} \boldsymbol{d}_{\mathbf{i}}$', fontweight='bold')
    ax.axhline(y=1.0, color='black', linestyle='--', zorder=10, linewidth=1.5, dashes=[2, 2])

    fig.subplots_adjust(top=0.95, right=0.845, left=0.16, bottom=0.18)
    plt.savefig("Figure4_au_hel_normalized_di_norm.png", dpi=600)
    print("-> Saved Figure4_au_hel_normalized_di_norm (.pdf / .png)")
    plt.close(fig)

    # 2. Expand Plot
    fig, ax = plt.subplots(figsize=(3.4*0.5, 3.4*0.65), constrained_layout=False)
    ax.pcolormesh(au_mesh, freqs, hel, cmap=cmap, norm=norm, edgecolors='face', shading='auto', antialiased=False)
    
    ax.set_yscale('log')
    ax.set_ylim(3.0e-1, 3.0e0)
    ax.set_xlim(0.100, 0.200)
    ax.set_facecolor('black')

    ax.tick_params(direction='out', which='both', width=1.5, length=3)
    ax.tick_params(axis='x', which='minor', bottom=False)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)

    ax.set_yticks([0.3, 1.0, 3.0])
    ax.set_yticklabels(['0.3', '1.0', '3.0'])
    ax.tick_params(axis='y', which='minor', labelleft=False)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(10); label.set_horizontalalignment('left'); label.set_x(-0.12)
    for label in ax.get_xticklabels(): label.set_fontweight('bold')

    ax.axhline(y=1.0, color='black', linestyle='--', zorder=10, linewidth=1.5, dashes=[2, 2])

    fig.subplots_adjust(top=0.98, right=0.92, left=0.15, bottom=0.09)
    plt.savefig("Figure4_au_hel_normalized_di_norm_expand.png", dpi=600)
    print("-> Saved Figure4_au_hel_normalized_di_norm_expand (.png)")
    plt.close(fig)

def main():
    data_path = "plot_data/fig4_data.h5"
    
    # Global plotting parameters
    plt.rcParams["font.family"] = "Myriad Pro"
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.size"] = 10
    
    cmap = get_custom_cmap()
    norm = mcolors.TwoSlopeNorm(0.0, vmin=-1.0, vmax=1.0)
    
    try:
        with h5py.File(data_path, 'r') as f:
            if 'top_panel' in f:
                plot_top(f, cmap, norm)
            if 'middle_panel' in f:
                plot_middle(f, cmap, norm)
            if 'bottom_panel' in f:
                plot_bottom(f, cmap, norm)
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")

if __name__ == "__main__":
    main()