"""
*******************************************************************************

Script purpose:

    Visualize the outputs of biological and cellular -patheay enrichment

Note:
    The results coming from this script are shown in Fig.4a.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from globals import path_gene_results, path_fig

#------------------------------------------------------------------------------
# Function to generate colors with specified min and max
#------------------------------------------------------------------------------

def colors_from_values(values, palette_name, min_val = -0, max_val = 0):
    # Normalize the values to range [min_val, max_val]
    normalized = (values - min_val) / (max_val - min_val)
    # Ensure values are within [0, 1]
    normalized = np.clip(normalized, 0, 1)
    # Convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # Use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis = 0)

#------------------------------------------------------------------------------
# Load data - results of gene enrichment analysis are loaded
#------------------------------------------------------------------------------

# Change this to switch to the other map
name = 'GCEA_epicenter_10_GO_cellularComponentDirect-discrete'
#name = 'GCEA_epicenter_10_GO-biologicalProcessDirect-discrete'

cell = pd.read_csv(path_gene_results + name + '.csv')

#------------------------------------------------------------------------------
# Visualize the results
#------------------------------------------------------------------------------

pval = 0.05

cell_filtered = cell.sort_values(by='pValPermCorr')
cell_filtered = cell_filtered[cell_filtered.pValPermCorr <= 0.05]
cell_filtered = cell_filtered.sort_values(by = 'cScorePheno',
                                          ascending = False).head(30)


# Set the global font to Arial, size 6
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 6

cell_filtered = cell_filtered.sort_values(by = 'cScorePheno',
                                          ascending = False).head(30)

# Explicitly set the figure's background to white
fig, ax = plt.subplots(1, 1, figsize = (8, 3*4.6), facecolor='white')
ax.set_facecolor('white')
scatter_plot = ax.scatter(cell_filtered.cScorePheno, range(len(cell_filtered)),
                          c=colors_from_values(cell_filtered.cScorePheno, 'coolwarm'))

# Labels
for i, (x, label) in enumerate(zip(cell_filtered.cScorePheno, cell_filtered.cDesc)):
    ax.annotate(label,
                (x, i),
                xytext = (5, 0),
                textcoords = 'offset points',
                size = 12,
                va = 'center')

# Add horizontal lines
colors = colors_from_values(cell_filtered.cScorePheno, 'coolwarm')
for i, x in enumerate(cell_filtered.cScorePheno):
    ax.hlines(i,
              0,
              x,
              color = colors[i],
              alpha = 0.7,
              linestyles = 'dashed')

ax.axvline(0, color = 'k', linewidth = 1)
ax.set_xlabel('c-score', fontsize = 12)
ax.set_yticks(range(len(cell_filtered)))
ax.set_yticklabels(cell_filtered.cLabel)
plt.xlim(0.05, 0.25)  # cellular
#plt.xlim(0.10, 0.35) # biological
sns.despine(fig, left=True, bottom=True, right=True)
plt.savefig(path_fig + '/' + name + '.svg',
            bbox_inches = 'tight',
            dpi = 300)

#------------------------------------------------------------------------------
# END