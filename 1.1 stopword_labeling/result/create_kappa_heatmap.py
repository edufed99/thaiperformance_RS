import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read the kappa matrix
df = pd.read_csv('model_pairwise_kappa.csv', index_col=0)

# Reorder for better presentation
order = ['Gemini', 'DeepSeek', 'GPT', 'Qwen']
df = df.reindex(index=order, columns=order)

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn',
            vmin=0.4, vmax=1.0,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': "Cohen's Kappa"},
            square=True, ax=ax)

plt.title("Cohen's Kappa Correlation Matrix\n(Pairwise Agreement between LLM Hierarchical Labels)",
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('cohens_kappa_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved: cohens_kappa_heatmap.png')

# Print formatted table
print()
print('Correlation Matrix Table:')
print(df.round(4).to_string())
