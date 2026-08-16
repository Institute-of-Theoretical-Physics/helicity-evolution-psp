#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reproduce Figure 3 (1D Normalized magnetic helicity density spectra).
Requirements: h5py, matplotlib, numpy
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    data_path = "plot_data/fig3_data.h5"
    try:
        with h5py.File(data_path, 'r') as f:
            au_list = f['Distance_AU'][:]
            freqs_list = f['Frequency_Hz'][:]
            hel_list = f['Helicity_Norm'][:]
            ci_lower_list = f['CI_Lower'][:]
            ci_upper_list = f['CI_Upper'][:]
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

    plt.style.use('grayscale')
    plt.rcParams["font.family"] = "Myriad Pro"
    
    n = len(au_list)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, 3.4), sharex="col", gridspec_kw={'hspace': 0})
    
    for i, ax in enumerate(axes):
        ax.plot(freqs_list[i], hel_list[i], color="#000000", linewidth=1.0)
        ax.fill_between(freqs_list[i], ci_lower_list[i], ci_upper_list[i], facecolor='#999999', linewidth=0.5)
        
        ax.set_xlim(3e-3, 2e2)
        ax.set_ylim(-1, 1)
        ax.set_xscale('log')
        
        ax.tick_params(direction='in', which='both', width=1.5, length=3)
        ax.tick_params(axis='x', which='minor', bottom=False)
        for spine in ax.spines.values(): spine.set_linewidth(1.5)
        for label in ax.get_yticklabels(): label.set_fontweight('bold'); label.set_fontsize(8)
        for label in ax.get_xticklabels(): label.set_fontweight('bold')
        
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', zorder=0, alpha=0.5)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_yticklabels(['$\\mathdefault{-0.5}$', '$\\mathdefault{0}$', '$\\mathdefault{0.5}$'])
        ax.text(0.99, 0.9, f"{au_list[i]} au", transform=ax.transAxes, ha='right', va='top', fontsize=8, fontweight='bold')

    # Recreate the color spans indicating wave modes (matches the original plotting coordinates)
    axes[0].axvspan(1.9e1, 7e1, ymin=0.5, ymax=1, facecolor='magenta', alpha=0.25, zorder=0, linewidth=0)
    axes[0].axvspan(5e0, 1.9e1, ymin=-1, ymax=0.5, facecolor='cyan', alpha=0.25, zorder=0, linewidth=0)
    axes[1].axvspan(1.5e0, 4.73e0, ymin=-1, ymax=0.5, facecolor='cyan', alpha=0.25, zorder=0, linewidth=0)
    axes[2].axvspan(2.6e0, 4.73e0, ymin=0.5, ymax=1, facecolor='magenta', alpha=0.25, zorder=0, linewidth=0)
    axes[2].axvspan(9e-1, 2.6e0, ymin=-1, ymax=0.5, facecolor='cyan', alpha=0.25, zorder=0, linewidth=0)
    axes[3].axvspan(2.2e0, 4.73e0, ymin=0.5, ymax=1, facecolor='magenta', alpha=0.25, zorder=0, linewidth=0)
    axes[3].axvspan(6e-1, 2.2e0, ymin=-1, ymax=0.5, facecolor='cyan', alpha=0.25, zorder=0, linewidth=0)
    axes[4].axvspan(6e-1, 4.73e0, ymin=0.5, ymax=1, facecolor='magenta', alpha=0.25, zorder=0, linewidth=0)
    axes[4].axvspan(2e0, 2.7e0, facecolor='green', alpha=0.25, zorder=0, linewidth=0)
    axes[5].axvspan(1.35e0, 1.85e0, facecolor='green', alpha=0.25, zorder=0, linewidth=0)
    axes[6].axvspan(1.1e0, 1.95e0, facecolor='green', alpha=0.25, zorder=0, linewidth=0)
    axes[6].axvspan(3.4e0, 4.3e0, facecolor='green', alpha=0.25, zorder=0, linewidth=0)
    axes[7].axvspan(2.3e0, 2.9e0, facecolor='green', alpha=0.25, zorder=0, linewidth=0)

    axes[-1].set_xlabel("Frequency [Hz]", fontsize=10, fontweight='bold')
    axes[n//2].set_ylabel(r"$\boldsymbol{\sigma}_{\boldsymbol{\mathrm{m}}}$", fontsize=10, fontweight='bold')
    axes[n//2].yaxis.set_label_coords(-0.12,0.9)
    
    fig.subplots_adjust(top=0.99, left=0.15, bottom=0.12, right=0.99)
    plt.savefig("Figure3.png", dpi=600)
    print("Saved Figure3.png")

if __name__ == "__main__":
    main()