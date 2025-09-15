# -*- coding: utf-8 -*-
"""
Publication-Quality Figure Generator (v2.3 - Final English Version)

This script implements a unified and robust analysis pipeline for both
D-Scaling and N-Scaling experiments. It automatically identifies the
"valid regime" and applies the correct fitting logic for each scaling law.

Key features:
- Automatically detects experiment type (D-Scaling vs. N-Scaling).
- Universally applies the "valid regime" logic.
- Adaptive fitting for H'sie (Logarithmic for N-Scaling, Exponential for D-Scaling).
- Professional 2x3 layout, increased font sizes, and high-resolution output.

Usage:
    python publication_figure_generator.py <path_to_your_csv_file.csv> [-o <output_name.png>]
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import argparse
import os

# --- Universal Fitting Functions ---
def power_law_fit(x, y):
    """Performs a power-law fit (y = c * x^s)."""
    mask = (y > 0) & (x > 0) & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2: return 0, 1, 0, np.full_like(x, np.nan, dtype=float)
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    s, i, r, p, _ = linregress(lx, ly); r2 = r**2
    return r2, p, s, 10**(s*np.log10(x)+i)

def exp_decay_fit(x, y):
    """Performs an exponential decay fit (y = c * exp(-d*x))."""
    mask = (y > 0) & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2: return 0, 1, 0, np.full_like(x, np.nan, dtype=float)
    lx, ly = x[mask], np.log(y[mask])
    s, i, r, p, _ = linregress(lx, ly); r2 = r**2
    return r2, p, -s, np.exp(s*x+i)

def logarithmic_fit(x, y):
    """Performs a logarithmic fit (y = a*log10(x) + b)."""
    mask = (x > 0) & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return 0, 1, 0, 0, np.full_like(x, np.nan, dtype=float)
    log_x = np.log10(x[mask])
    y_masked = y[mask]
    
    slope, intercept, r_value, p_value, _ = linregress(log_x, y_masked)
    r_squared = r_value**2
    y_fit = slope * np.log10(x) + intercept
    
    return r_squared, p_value, slope, intercept, y_fit

# --- Main Plotting Function ---
def create_plots(csv_path, output_path):
    # --- Data Loading and Universal Setup ---
    df = pd.read_csv(csv_path)

    # --- Experiment Type Detection ---
    if 'num_params_N' in df.columns:
        exp_type = 'N-Scaling'
        x_col = 'num_params_N'
        x_label_var = 'N'
        x_label_full = 'Number of Parameters (N)'
        df = df.sort_values(x_col).reset_index(drop=True)
    elif 'data_size_d' in df.columns:
        exp_type = 'D-Scaling'
        x_col = 'data_size_d'
        x_label_var = 'D'
        x_label_full = 'Dataset Size (D)'
        df = df.sort_values(x_col).reset_index(drop=True)
    else:
        raise ValueError("Error: CSV must contain either 'num_params_N' or 'data_size_d' column.")

    # --- Universal "Valid Regime" Logic ---
    min_loss_idx = df['final_test_loss'].idxmin()
    df_valid = df.loc[:min_loss_idx].copy()
    print(f"Detected {exp_type} data. Analysis based on the 'valid regime' containing {len(df_valid)} data points.")
    
    # --- Data Preparation for Reducible Metrics ---
    w1, w2 = 0.5, 0.5
    L_inf = df_valid['final_test_loss'].min()
    hsie_inf = df_valid['final_hsie'].min() # Asymptotic value for D-Scaling
    hsie_0 = df_valid['final_hsie'].min()   # Baseline value for N-Scaling
    htse_inf = df_valid['final_htse'].min() # Asymptotic value for N-Scaling
    htse_0 = df_valid['final_htse'].min()   # Baseline value for D-Scaling

    # For D-Scaling, H_0 is estimated from the first few points
    if exp_type == 'D-Scaling':
        n_est = 5
        if len(df_valid) > n_est:
            htse_0 = df_valid['final_htse'].head(n_est).mean()
    
    df_valid['reducible_loss'] = df_valid['final_test_loss'] - L_inf
    
    if exp_type == 'N-Scaling':
        df_valid['reducible_htse'] = df_valid['final_htse'] - htse_inf
        df_valid['reducible_hsie'] = df_valid['final_hsie'] - hsie_0
    else: # D-Scaling
        df_valid['reducible_htse'] = df_valid['final_htse'] - htse_0
        df_valid['reducible_hsie'] = df_valid['final_hsie'] - hsie_inf

    x_valid = df_valid[x_col].values

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(f'Cognitive Investment Model - Unified Theory Analysis ({exp_type})', fontsize=28, y=0.96)
    
    # --- Plot 1: Performance Law ---
    ax = axes[0, 0]
    r2_1, p_1, s_1, fit_1 = power_law_fit(x_valid, df_valid['reducible_loss'])
    ax.plot(df[x_col], df['final_test_loss'], 'o', color='grey', alpha=0.5, label='All Data')
    ax.plot(df_valid[x_col], df_valid['final_test_loss'], 'o', color='blue', label='Valid Regime')
    ax.plot(x_valid, fit_1 + L_inf, '--', color='red')
    text_1 = (f'$R^2={r2_1:.2f}, p={p_1:.1e}$\n'
              f'$L-L_\\infty \\propto {x_label_var}^{{{s_1:.2f}}}$')
    ax.text(0.95, 0.95, text_1, ha='right', va='top', transform=ax.transAxes, bbox=dict(fc='wheat', alpha=0.5), fontsize=16)
    ax.legend()

    # --- Plot 2: Abstraction Component ---
    ax = axes[0, 1]
    r2_2, p_2, s_2, fit_2 = power_law_fit(x_valid, df_valid['reducible_htse'])
    htse_baseline = htse_inf if exp_type == 'N-Scaling' else htse_0
    ax.plot(x_valid, df_valid['final_htse'], 's', color='green')
    ax.plot(x_valid, fit_2 + htse_baseline, '--', color='purple')
    htse_label = 'H-H_\\infty' if exp_type == 'N-Scaling' else 'H-H_0'
    text_2 = (f'$R^2={r2_2:.2f}, p={p_2:.1e}$\n'
              f'${htse_label} \\propto {x_label_var}^{{{s_2:.2f}}}$')
    text_pos_y = 0.05 if exp_type == 'N-Scaling' else 0.95 # Adjust text position based on curve direction
    text_va = 'bottom' if exp_type == 'N-Scaling' else 'top'
    ax.text(0.95, text_pos_y, text_2, ha='right', va=text_va, transform=ax.transAxes, bbox=dict(fc='wheat', alpha=0.5), fontsize=16)

    # --- Plot 3: Compression Component (Adaptive Fitting) ---
    ax = axes[0, 2]
    ax.plot(x_valid, df_valid['final_hsie'], '^', color='orange')
    
    if exp_type == 'N-Scaling':
        r2_3, p_3, a_3, b_3, fit_3 = logarithmic_fit(x_valid, df_valid['reducible_hsie'])
        hsie_baseline = hsie_0
        ax.plot(x_valid, fit_3 + hsie_baseline, '--', color='darkcyan')
        text_3 = (f'$R^2={r2_3:.3f}, p={p_3:.1e}$\n'
                  f'$H-H_0 \\propto \\log({x_label_var})$')
    else: # D-Scaling
        r2_3, p_3, d_rate_3, fit_3 = exp_decay_fit(x_valid, df_valid['reducible_hsie'])
        hsie_baseline = hsie_inf
        ax.plot(x_valid, fit_3 + hsie_baseline, '--', color='darkred')
        text_3 = (f'$R^2={r2_3:.2f}, p={p_3:.1e}$\n'
                  f'$H-H_\\infty \\propto e^{{-{d_rate_3:.1e}{x_label_var}}}$')

    ax.text(0.95, 0.95, text_3, ha='right', va='top', transform=ax.transAxes, bbox=dict(fc='wheat', alpha=0.5), fontsize=16)
    
    # --- Plot 4: Internal Cost Trend ---
    ax = axes[1, 0]
    htse_red_sq = np.maximum(0, df_valid['reducible_htse'])**2
    hsie_red_sq = np.maximum(0, df_valid['reducible_hsie'])**2
    df_valid['L_ideal_reducible'] = np.sqrt(w1 * htse_red_sq + w2 * hsie_red_sq)
    ax.plot(x_valid, df_valid['L_ideal_reducible'], 'p', color='purple')
    
    # --- Plot 5: The Core Law ---
    ax = axes[1, 1]
    r2_4, p_4, s_4, fit_4 = power_law_fit(df_valid['L_ideal_reducible'], df_valid['reducible_loss'])
    ax.plot(df_valid['L_ideal_reducible'], df_valid['reducible_loss'], 'd', color='black')
    ax.plot(df_valid['L_ideal_reducible'], fit_4, '--', color='magenta')
    text_5 = (f'Core Law:\n'
              f'$R^2={r2_4:.4f}, p={p_4:.1e}$\n'
              f'$L_{{red}} \\propto \\mathcal{{L}}_{{ideal, red}}^{{{s_4:.2f}}}$')
    ax.text(0.95, 0.95, text_5, ha='right', va='top', transform=ax.transAxes, bbox=dict(fc='wheat', alpha=0.5), fontsize=16)

    # --- Final Formatting and Saving ---
    axes[0,0].set_title('Law 1: Performance', fontsize=18)
    axes[0,0].set_ylabel('Final Test Loss', fontsize=16)
    axes[0,1].set_title("Component 1: Abstraction ($H'_{tse}$)", fontsize=18)
    axes[0,1].set_ylabel("Final $H'_{tse}$", fontsize=16)
    axes[0,2].set_title("Component 2: Compression ($H'_{sie}$)", fontsize=18)
    axes[0,2].set_ylabel("Final $H'_{sie}$", fontsize=16)
    axes[1,0].set_title('Internal Cost $\\mathcal{L}_{ideal, red}$ Trend', fontsize=18)
    axes[1,0].set_ylabel('Reducible Ideal Norm', fontsize=16)
    axes[1,1].set_title('The Core Law: Performance vs. Cost', fontsize=18)
    axes[1,1].set_ylabel('Reducible Test Loss', fontsize=16)
    axes[1,2].axis('off') # Turn off the empty subplot

    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2: continue
            ax = axes[i, j]
            
            # Set X-labels
            if i == 0 or (i == 1 and j == 0):
                ax.set_xlabel(x_label_full, fontsize=16)
            elif i == 1 and j == 1:
                ax.set_xlabel('Reducible Ideal Norm $\\mathcal{L}_{ideal, red}$', fontsize=16)
            
            # Set scales
            ax.set_xscale('log')
            if exp_type == 'N-Scaling' and j == 2 and i == 0: 
                ax.set_yscale('linear') # H'sie log growth is better on linear y-axis
            else:
                ax.set_yscale('log')
            
            ax.grid(True, which='both', linestyle='--')
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPublication-quality figure saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a 2x3 publication-quality figure from scaling law experiment data.", 
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", type=str, 
                        help="Path to save the output figure (default: 'scaling_law_summary_figure.png' in the same directory as the input CSV).")
    args = parser.parse_args()
    
    if args.output:
        output_path = args.output
    else:
        # Default output path logic
        input_dir = os.path.dirname(os.path.abspath(args.csv_path))
        base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        output_path = os.path.join(input_dir, f"{base_name}_summary_figure.png")
    
    create_plots(args.csv_path, output_path)
