"""
Generate Figure 5: Hybrid fusion method comparison
Data source: newoutput/results_hybrid_methods.csv
Style: Publication-ready grayscale with hatching patterns (matching 12.jpg reference)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set style for academic publication - Grayscale with Times New Roman
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Read the data
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'results_hybrid_methods.csv'))

max_cands_list = [50, 100, 200]
methods = ['RRF', 'Cascade', 'WeightedSum', 'Switching']

# Style for each method - matching reference image
method_styles = {
    'RRF': {
        'color': '0.15',      # Dark (almost black)
        'hatch': '',          # Solid
        'edgecolor': 'black'
    },
    'Cascade': {
        'color': '0.4',       # Medium dark gray
        'hatch': '///',       # Diagonal lines
        'edgecolor': 'black'
    },
    'WeightedSum': {
        'color': '0.7',       # Light gray
        'hatch': '...',       # Dots
        'edgecolor': 'black'
    },
    'Switching': {
        'color': '0.9',       # Very light gray
        'hatch': 'xxx',       # Crosshatch
        'edgecolor': 'black'
    },
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(max_cands_list))
width = 0.2  # Width of each bar
offsets = [-1.5, -0.5, 0.5, 1.5]  # Positions for 4 bars

for i, method in enumerate(methods):
    method_df = df[df['hybrid_method'] == method]
    
    means = [method_df[method_df['MAX_CANDS'] == mc]['ndcg_mean'].values[0] for mc in max_cands_list]
    stds = [method_df[method_df['MAX_CANDS'] == mc]['ndcg_std'].values[0] for mc in max_cands_list]
    
    style = method_styles[method]
    
    bars = ax.bar(
        x + offsets[i] * width,
        means,
        width,
        label=method,
        color=style['color'],
        hatch=style['hatch'],
        edgecolor=style['edgecolor'],
        linewidth=0.8,
        yerr=stds,
        capsize=3,
        error_kw={'elinewidth': 1.0, 'capthick': 1.0, 'color': 'black'}
    )

# Customize axes
ax.set_xlabel('MAX_CANDS', fontweight='normal')
ax.set_ylabel('nDCG@10', fontweight='normal')
ax.set_title('Comparison of hybrid fusion methods across candidate caps', fontweight='normal', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(max_cands_list)

# Set y-axis limits matching reference
ax.set_ylim(0.70, 0.90)
ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)

# Add legend in upper right
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)

plt.tight_layout()

# Save figures
plt.savefig(os.path.join(SCRIPT_DIR, 'fig5_hybrid_fusion_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SCRIPT_DIR, 'fig5_hybrid_fusion_comparison.pdf'), dpi=300, bbox_inches='tight')
print("Saved: fig5_hybrid_fusion_comparison.png and fig5_hybrid_fusion_comparison.pdf")

# Print data summary
print("\n" + "="*60)
print("DATA SUMMARY: Hybrid Fusion Methods (nDCG@10)")
print("="*60)
for method in methods:
    method_df = df[df['hybrid_method'] == method]
    print(f"\n{method}:")
    for mc in max_cands_list:
        row = method_df[method_df['MAX_CANDS'] == mc].iloc[0]
        print(f"  MC={mc}: nDCG={row['ndcg_mean']:.4f}±{row['ndcg_std']:.4f}")

# Print ranking
print("\n" + "="*60)
print("RANKING BY MAX_CANDS")
print("="*60)
for mc in max_cands_list:
    mc_df = df[df['MAX_CANDS'] == mc].sort_values('ndcg_mean', ascending=False)
    print(f"\nMAX_CANDS={mc}:")
    for i, (_, row) in enumerate(mc_df.iterrows(), 1):
        print(f"  {i}. {row['hybrid_method']}: {row['ndcg_mean']:.4f}")
