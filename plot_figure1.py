#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reproduce Figure 1.
Requirements: polars, matplotlib, numpy
"""
import polars as pl
import matplotlib.pyplot as plt

def main():
    # 1. Load the pre-processed data
    try:
        df = pl.read_csv("plot_data/fig1_data.csv")
    except Exception as e:
        print(f"Failed to load data. Please ensure 'plot_data/fig1_data.csv' exists. Error: {e}")
        return

    # Extract columns matching the actual CSV header
    au_list = df['au'].to_numpy()
    
    # 2. Plotting setup
    plt.rcParams["font.family"] = "Myriad Pro" # Requires Myriad Pro installed
    plt.rcParams['mathtext.default'] = 'regular'
    
    fig, ax = plt.subplots(7, 1, figsize=(3.4, 3.4*1.1), sharex="col", 
                           gridspec_kw={'hspace': 0, 'height_ratios': [1]*7})
    labelpad_ylabel = 25

    # 3. Plot panels
    # Panel 1: Flow Speed
    ax[0].errorbar(au_list, df['mean_R'].to_numpy(), yerr=df['std_R'].to_numpy(), label="R", fmt='o', markersize=2, elinewidth=0.5, color='C6')
    ax[0].errorbar(au_list, df['mean_T'].to_numpy(), yerr=df['std_T'].to_numpy(), label="T", fmt='o', markersize=2, elinewidth=0.5, color='C7')
    ax[0].errorbar(au_list, df['mean_N'].to_numpy(), yerr=df['std_N'].to_numpy(), label="N", fmt='o', markersize=2, elinewidth=0.5, color='C8')
    ax[0].set_ylabel(r"$V_{\mathrm{sw}}$" + "\n" + r"(km/s)", fontweight='bold', va="top", labelpad=labelpad_ylabel)
    ax[0].legend(loc='upper right', prop={'weight': 'bold','size': 6}, handlelength=1.0, handletextpad=0.5, borderaxespad=0.5)

    # Panel 2: Density
    ax[1].errorbar(au_list, df['mean_density'].to_numpy(), yerr=df['std_density'].to_numpy(), label="Density", fmt='o', markersize=2, elinewidth=0.5, color='C5')
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$n_{\mathrm{p}}$" + "\n" + r"(cm$^{-3}$)", fontweight='bold', va="top", labelpad=labelpad_ylabel)

    # Panel 3: Temperature
    ax[2].errorbar(au_list, df['mean_temperature'].to_numpy(), yerr=df['std_temperature'].to_numpy(), label="Temperature", fmt='o', markersize=2, elinewidth=0.5, color='C4')
    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"$T_{\mathrm{p}}$" + "\n" + r"(eV)", fontweight='bold', va="top", labelpad=labelpad_ylabel)

    # Panel 4: Beta Parallel
    ax[3].errorbar(au_list, df['beta_parallel'].to_numpy(), label="Beta Parallel", fmt='o', markersize=2, color='C3')
    ax[3].set_yscale("log")
    ax[3].set_ylabel(r"$\beta_{\parallel}$", fontweight='bold', va="top", labelpad=labelpad_ylabel)

    # Panel 5: Anisotropy
    ax[4].errorbar(au_list, df['anisotropy'].to_numpy(), label="Anisotropy", fmt='o', markersize=2, color='C2')
    ax[4].set_yscale("log")
    ax[4].set_ylim(1e-1, 1e1)
    ax[4].set_ylabel(r"$T_{\perp} / T_{\parallel}$", fontweight='bold', va="top", labelpad=labelpad_ylabel)

    # Panel 6: Taylor Epsilon
    ax[5].errorbar(au_list, df['taylor_epsilon'].to_numpy(), label="Taylor Epsilon", fmt='o', markersize=2, color='C1')
    ax[5].set_yscale("log")
    ax[5].set_ylabel(r"$\epsilon$", fontweight='bold', va="top", labelpad=labelpad_ylabel)

    # Panel 7: Shift K Coef
    ax[6].errorbar(au_list, df['shift_k_coef'].to_numpy(), label="1 + Alfven Speed / Relative Velocity Perp", fmt='o', markersize=2, color='C0')
    ax[6].set_yscale("log")
    ax[6].set_ylabel(r"$1 + v_{\mathrm{A}} / v_{\perp}$", fontweight='bold', va="top", labelpad=labelpad_ylabel)
    ax[6].set_xlabel("Heliocentric Distance (au)", fontweight='bold')
    
    # 4. Formatting
    for i, a in enumerate(ax):
        for label in a.get_yticklabels():
            if i != 0: label.set_horizontalalignment('left'); label.set_x(-0.10)
            label.set_fontweight('bold')
        for spine in a.spines.values():
            spine.set_linewidth(1.5)
        a.tick_params(direction='out', width=1.5, length=5)
        if i != 6:
            a.spines['bottom'].set_linewidth(0.75)
            a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            for label in a.get_xticklabels(): label.set_fontweight('bold')

    plt.tight_layout(pad=0)
    fig.subplots_adjust(top=0.995, left=0.22)
    plt.savefig("Figure1.pdf", dpi=600)
    print("Saved Figure1.pdf")

if __name__ == "__main__":
    main()