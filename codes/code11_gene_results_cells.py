"""
*******************************************************************************
Purpose:

    Visualize the outputs of cell-type enrichment

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path        = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_gen_results = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_network/results/finalabanotate_GO/'
script_name      = os.path.basename(__file__).split('.')[0]
path_fig         = os.path.join(os.getcwd(), 'generated_figures/', script_name)
os.makedirs(path_fig, exist_ok = True)

#------------------------------------------------------------------------------
# function to generate colors
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    
    return np.array(palette).take(indices, axis = 0)
#------------------------------------------------------------------------------
# load data

name ='GCEA_PsychEncode-cellTypesUMI-discrete_Pearson_10'
cell = pd.read_csv(path_gen_results + name + '.csv')
pval = 0.05

# Sort by pValZ and keep the lowest 15
cell_filtered = cell.sort_values(by = 'pValZ').head(35)

fig, axes = plt.subplots(1, 1, figsize = (5, 15))
cell_plot = sns.barplot(data = cell_filtered,
                        x = -np.log(cell_filtered.pValZ),
                        y = 'cLabel',
                        ax = axes,
                        palette = colors_from_values(cell.cScorePheno, 'coolwarm'))

plt.savefig(path_fig + '/celltypes.svg',
            bbox_inches = 'tight',
            dpi = 300)

# labels
for i, p in enumerate(cell_plot.patches):
    x = p.get_x() + p.get_width()
    xy = (5, -15)
    
    label = cell_filtered.cLabel[i] + '*' if cell_filtered.pValPermCorr[i] < pval else cell_filtered.cLabel[i]
   
    weight = 'bold' if cell_filtered.pValPermCorr[i] < pval else 'normal'
    
    cell_plot.annotate(label, 
                       (x, p.get_y()),
                       xytext = xy,
                       textcoords = 'offset points',
                       size = 12,
                       weight = weight)


cell_plot.axvline(0, color = 'k', linewidth = 1)
cell_plot.set_xlabel('-log(p)', fontsize = 12)
cell_plot.set(yticklabels = [], ylabel = None)
cell_plot.tick_params(left = False)
sns.despine(fig, None, True, True, True, False)

# Draw a rectangle around significant bars
pval_threshold = 0.005
for i, p in enumerate(cell_plot.patches):
    if cell_filtered.pValPermCorr.iloc[i] < pval:
        # Coordinates and dimensions of the rectangle
        x = p.get_x()   # Slightly offset to the left for aesthetics
        y = p.get_y()
        width = p.get_width()  # Slightly wider than the bar
        height = p.get_height()
        # Create a rectangle
        rect = patches.Rectangle((x, y),
                                 width,
                                 height,
                                 linewidth = 3,
                                 edgecolor='black',
                                 facecolor='none')
        axes.add_patch(rect)
        # Style customizations
sns.despine(fig, None, True, True, True, False)

# cbar
cb = plt.colorbar(mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin = cell.cScorePheno.min(), 
                                                                  vmax = cell.cScorePheno.max()), 
                  cmap = 'coolwarm'),
                  orientation = 'vertical', 
                  cax = fig.add_axes([1.2, 0.15, 0.04, 0.3]))

cb.set_label("Z (Pearson correlation", size = 12)
# save
plt.savefig(path_fig + '/celltypes.svg',
            bbox_inches = 'tight',
            dpi = 300)

#------------------------------------------------------------------------------
# END
