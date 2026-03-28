"""
Generate Figure 4: Performance comparison across model families and candidate caps.
Data source: newoutput/results_overall.csv
Style: Publication-ready grayscale with hatching patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set style for academic publication - Grayscale with Times New Roman
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Read the data from newoutput folder
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'results_overall.csv'))

# Data for best models per family
max_cands_list = [50, 100, 200]

# Extract data for each model family - Grayscale with distinct patterns
# Order: Best CBF (solid dark), Best CF (diagonal lines), Best Hybrid (dots)
data = {
    'Best CBF': {
        'model': 'CBF-multilingual-e5-large', 
        'color': '0.3',      # Dark gray (solid fill)
        'hatch': '',         # No hatch - solid
        'edgecolor': 'black'
    },
    'Best CF': {
        'model': 'CF-LightGCN', 
        'color': '0.6',      # Medium gray
        'hatch': '///',      # Diagonal lines
        'edgecolor': 'black'
    },
    'Best Hybrid': {
        'model': 'Hybrid-multilingual-e5-large-LightGCN', 
        'color': '0.85',     # Light gray
        'hatch': '...',      # Dots pattern
        'edgecolor': 'black'
    },
}

# Extract metrics for each family
for family, info in data.items():
    model_df = df[df['Model'] == info['model']]
    info['ndcg_mean'] = []
    info['ndcg_std'] = []
    info['mrr_mean'] = []
    info['mrr_std'] = []
    info['hr_mean'] = []
    info['hr_std'] = []

    for mc in max_cands_list:
        row = model_df[model_df['MAX_CANDS'] == mc].iloc[0]
        info['ndcg_mean'].append(row['ndcg_mean'])
        info['ndcg_std'].append(row['ndcg_std'])
        info['mrr_mean'].append(row['mrr_mean'])
        info['mrr_std'].append(row['mrr_std'])
        info['hr_mean'].append(row['hr_mean'])
        info['hr_std'].append(row['hr_std'])

# Create the figure with 3 panels - wider format
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

x = np.arange(len(max_cands_list))
width = 0.25

metrics = [
    ('ndcg', 'nDCG@10', '(a) nDCG@10'),
    ('mrr', 'MRR@10', '(b) MRR@10'),
    ('hr', 'HR@10', '(c) HR@10'),
]

for ax, (metric_key, ylabel, title) in zip(axes, metrics):
    for i, (family, info) in enumerate(data.items()):
        means = info[f'{metric_key}_mean']
        stds = info[f'{metric_key}_std']

        bars = ax.bar(
            x + (i - 1) * width, 
            means, 
            width,
            label=family, 
            color=info['color'], 
            hatch=info['hatch'],
            edgecolor=info['edgecolor'], 
            linewidth=0.8,
            yerr=stds, 
            capsize=3, 
            error_kw={'elinewidth': 1.0, 'capthick': 1.0, 'color': 'black'}
        )

    ax.set_xlabel('MAX_CANDS', fontweight='normal')
    ax.set_ylabel(ylabel, fontweight='normal')
    ax.set_title(title, fontweight='normal', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(max_cands_list)

    # Set y-axis limits based on metric (matching reference image style)
    if metric_key == 'hr':
        ax.set_ylim(0.70, 1.02)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    elif metric_key == 'mrr':
        ax.set_ylim(0.60, 0.90)
        ax.set_yticks([0.60, 0.70, 0.80, 0.90])
    else:  # ndcg
        ax.set_ylim(0.65, 0.95)
        ax.set_yticks([0.65, 0.75, 0.85, 0.95])

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)

# Add shared legend at the top center
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, 
           bbox_to_anchor=(0.5, 1.02), frameon=False,
           columnspacing=2.0, handletextpad=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SCRIPT_DIR, 'fig4_recommendation_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SCRIPT_DIR, 'fig4_recommendation_comparison.pdf'), dpi=300, bbox_inches='tight')
print("Saved: fig4_recommendation_comparison.png and fig4_recommendation_comparison.pdf")

# ============================================================
# Hybrid method comparison figure
# ============================================================
df_hybrid = pd.read_csv(os.path.join(SCRIPT_DIR, 'results_hybrid_methods.csv'))

methods = ['RRF', 'Cascade', 'WeightedSum', 'Switching']
method_styles = {
    'RRF': {'color': '0.1', 'hatch': ''},
    'Cascade': {'color': '0.35', 'hatch': '///'},
    'WeightedSum': {'color': '0.6', 'hatch': '...'},
    'Switching': {'color': '0.85', 'hatch': 'xxx'},
}

fig2, ax2 = plt.subplots(figsize=(8, 5))

x2 = np.arange(len(max_cands_list))
width2 = 0.2

for i, method in enumerate(methods):
    method_df = df_hybrid[df_hybrid['hybrid_method'] == method]
    means = [method_df[method_df['MAX_CANDS'] == mc]['ndcg_mean'].values[0] for mc in max_cands_list]
    stds = [method_df[method_df['MAX_CANDS'] == mc]['ndcg_std'].values[0] for mc in max_cands_list]

    ax2.bar(x2 + (i - 1.5) * width2, means, width2,
            label=method, color=method_styles[method]['color'],
            hatch=method_styles[method]['hatch'],
            edgecolor='black', linewidth=0.5,
            yerr=stds, capsize=2, error_kw={'elinewidth': 0.8, 'capthick': 0.8})

ax2.set_xlabel('MAX_CANDS')
ax2.set_ylabel('nDCG@10')
ax2.set_title('Comparison of hybrid fusion methods across candidate caps')
ax2.set_xticks(x2)
ax2.set_xticklabels(max_cands_list)
ax2.set_ylim(0.7, 0.9)
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_hybrid_method_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_hybrid_method_comparison.pdf'), dpi=300, bbox_inches='tight')
print("Saved: fig_hybrid_method_comparison.png and fig_hybrid_method_comparison.pdf")

# ============================================================
# Print summary of data used
# ============================================================
print("\n" + "="*60)
print("DATA SUMMARY (from newoutput)")
print("="*60)
for family, info in data.items():
    print(f"\n{family} ({info['model']}):")
    for i, mc in enumerate(max_cands_list):
        print(f"  MC={mc}: nDCG={info['ndcg_mean'][i]:.4f}±{info['ndcg_std'][i]:.4f}, "
              f"HR={info['hr_mean'][i]:.4f}, MRR={info['mrr_mean'][i]:.4f}")
