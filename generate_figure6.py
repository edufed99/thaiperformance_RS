"""
Generate Figure 6: Context policy violation across control strategies
Data source: newoutput/results_context_policy_violation.csv
Style: Publication-ready grayscale line chart (matching 13.jpg reference)
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
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'results_context_policy_violation.csv'))

# Use MAX_CANDS=100 as representative (values are same across MAX_CANDS for POP)
max_cands = 100

# Get data for each model
hybrid_row = df[(df['model_name'] == 'Hybrid-Best') & (df['MAX_CANDS'] == max_cands)].iloc[0]
cf_row = df[(df['model_name'] == 'CF-LightGCN') & (df['MAX_CANDS'] == max_cands)].iloc[0]
pop_row = df[(df['model_name'] == 'POP-Global (no filter)') & (df['MAX_CANDS'] == max_cands)].iloc[0]

# Prepare data
cutoffs = ['@1', '@5', '@10']
x = np.arange(len(cutoffs))

# Convert to percentages
hybrid_violations = [hybrid_row['Violation@1'] * 100, hybrid_row['Violation@5'] * 100, hybrid_row['Violation@10'] * 100]
cf_violations = [cf_row['Violation@1'] * 100, cf_row['Violation@5'] * 100, cf_row['Violation@10'] * 100]
pop_violations = [pop_row['Violation@1'] * 100, pop_row['Violation@5'] * 100, pop_row['Violation@10'] * 100]

# Create figure - more square aspect ratio like reference
fig, ax = plt.subplots(figsize=(5.5, 4))

# Plot lines with distinct styles (matching reference exactly)
# Best Hybrid - solid line with filled circles
ax.plot(x, hybrid_violations, 'o-', color='black', linewidth=1.2, markersize=6, 
        label='Best Hybrid', markerfacecolor='black', markeredgewidth=1)

# POP (no filter) - dash-dot line with filled triangles (plotted second to match legend order)
ax.plot(x, pop_violations, '^-.', color='black', linewidth=1.2, markersize=6, 
        label='POP (no filter)', markerfacecolor='black', markeredgewidth=1)

# Best CF - dashed line with filled squares
ax.plot(x, cf_violations, 's--', color='gray', linewidth=1.2, markersize=6, 
        label='Best CF', markerfacecolor='gray', markeredgewidth=1)

# Add value labels for POP line (since it has significant values)
for i, v in enumerate(pop_violations):
    ax.annotate(f'{v:.1f}', (x[i], v), textcoords="offset points", 
                xytext=(0, 8), ha='center', fontsize=8, fontweight='normal')

# Add annotation for gated models
ax.annotate('Gated models remain at 0 across all cutoffs', 
            xy=(1.95, 1), xytext=(1.1, 8),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.6))

# Customize axes
ax.set_xlabel('Rank cutoff', fontweight='normal')
ax.set_ylabel('Violation rate (%)', fontweight='normal')
ax.set_xticks(x)
ax.set_xticklabels(cutoffs)

# Set y-axis limits
ax.set_ylim(-2, 55)
ax.set_yticks([0, 10, 20, 30, 40, 50])

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)

# Add legend inside plot (upper left) with 2 columns - matching reference
ax.legend(loc='upper left', ncol=2, frameon=True, fancybox=False,
          edgecolor='black', framealpha=1, fontsize=8,
          handlelength=2.5, columnspacing=1)

plt.tight_layout()

# Save figures
plt.savefig(os.path.join(SCRIPT_DIR, 'fig6_context_policy_violation.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SCRIPT_DIR, 'fig6_context_policy_violation.pdf'), dpi=300, bbox_inches='tight')
print("Saved: fig6_context_policy_violation.png and fig6_context_policy_violation.pdf")

# Print data summary
print("\n" + "="*60)
print("DATA SUMMARY: Context Policy Violation (MAX_CANDS=100)")
print("="*60)
print(f"\nBest Hybrid (CTX_FILTER=ON):")
print(f"  @1: {hybrid_violations[0]:.1f}%")
print(f"  @5: {hybrid_violations[1]:.1f}%")
print(f"  @10: {hybrid_violations[2]:.1f}%")

print(f"\nBest CF (CTX_FILTER=ON):")
print(f"  @1: {cf_violations[0]:.1f}%")
print(f"  @5: {cf_violations[1]:.1f}%")
print(f"  @10: {cf_violations[2]:.1f}%")

print(f"\nPOP-Global (no filter, CTX_FILTER=OFF):")
print(f"  @1: {pop_violations[0]:.1f}%")
print(f"  @5: {pop_violations[1]:.1f}%")
print(f"  @10: {pop_violations[2]:.1f}%")

print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)
print("• Gated models (Hybrid, CF with pre-filter) achieve 0% violation")
print("• Ungated POP baseline shows ~43-46% policy violations")
print("• This demonstrates the effectiveness of eligibility-gated design")
