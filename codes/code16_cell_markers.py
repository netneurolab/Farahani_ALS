"""
*******************************************************************************

Script purpose:

    The aim of this code to delve into details of gene expression analysis results.
    This is requested by a reviewer.

NOTE:
    The results coming from this script are shown in a supplementary figure.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from globals import path_gene_results, path_fig, path_gene, path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Needed function
#------------------------------------------------------------------------------

# Function to generate colors
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    
    return np.array(palette).take(indices, axis = 0)

#------------------------------------------------------------------------------
# Load CSV file of - cell types
#------------------------------------------------------------------------------

name = 'GCEA_epicenter_10_PsychEncode-cellTypesUMI-discrete'
cell = pd.read_csv(path_gene_results + name + '.csv')

#------------------------------------------------------------------------------
# Load the name and raw-data of genes
#------------------------------------------------------------------------------

diff_stability = scipy.io.loadmat(path_gene + 'gene_coexpression_ds_filtered.mat')['gene_coexpression_ds']
df_genes = scipy.io.loadmat(path_gene + 'gene_coexpression_filtered.mat')['gene_coexpression']
name_genes = scipy.io.loadmat(path_gene + 'names_genes_filtered.mat')['names']

# Flatten the name_genes array
name_genes_flattened = np.array([name[0] for name in name_genes.flatten()])
rank_data  = np.load(path_results + 'epicenter_all_rank.npy')

#------------------------------------------------------------------------------
# Generate a scatter plot for each significant cell type
#------------------------------------------------------------------------------

pval = 0.05

for idx, cell_type in cell.iterrows():
    if cell_type['pValPermCorr'] < pval:

        name_genes_cellType = cell.cGenes[idx]
        # Get maps
        name_genes_list = [gene.strip() for gene in name_genes_cellType.split(',')]
        # Create a boolean mask that is True for genes in name_genes_flattened that are in name_genes_list
        mask = np.isin(name_genes_flattened,
                       name_genes_list)
        # Use this mask to filter df_genes to get the gene maps corresponding to the genes in name_genes_cellType
        gene_maps = df_genes[:, mask]
        name_maps = name_genes_flattened[mask]
        diff_stability_maps = diff_stability.flatten()[mask.T]
        plt.figure(figsize = (16, 16))
        spearmanr_cor = np.zeros(sum(mask))
        for  j in range(sum(mask)):
            spearmanr_cor[j], _ = spearmanr(rank_data,
                                            gene_maps[:, j])

        # Create a scatter plot
        plt.scatter(diff_stability_maps, spearmanr_cor, color = 'gray', s = 100)

        # Annotate each point with the gene name
        for i, gname in enumerate(name_maps):
            gene_name_str = str(gname)
            plt.text(diff_stability_maps[i],
                     spearmanr_cor[i],
                     gene_name_str,
                     fontsize = 18,
                     ha = 'right')

        # Add labels and title
        plt.xlabel('Differential Stability')
        plt.ylabel('Spearman Correlation with Brain Map')
        plt.title(f'Scatter Plot for Cell Type: {cell_type["cLabel"]}')

        # Save the plot
        plt.savefig(path_fig + f'/scatter_celltype_{cell_type["cLabel"]}.svg',
                    bbox_inches = 'tight', dpi = 300)
        plt.show()

# Initialize a DataFrame to collect the data for plotting
boxplot_data = []
significant_cell_types = []

# Loop over each cell type
for idx, cell_type in cell.iterrows():
    # Filter the gene names associated with this cell type
    name_genes_cellType = [gene.strip() for gene in cell_type['cGenes'].split(',')]
    # Get the Spearman correlations for these genes
    mask = np.isin(name_genes_flattened, name_genes_cellType)
    # Use this mask to filter df_genes to get the gene maps corresponding to the genes in name_genes_cellType
    gene_maps = df_genes[:, mask]
    spearman_corrs = np.zeros((sum(mask)))
    for j in range(sum(mask)):
        spearman_corrs[j], _ = spearmanr(rank_data, gene_maps[:, j])

    # Prepare data for boxplot
    for corr in spearman_corrs:
        boxplot_data.append({'Cell Type': cell_type['cLabel'],
                             'Spearman Correlation': corr})
    # Mark significant cell types
    if cell_type['pValPermCorr'] < 0.05:
        significant_cell_types.append(cell_type['cLabel'])
    print(str(cell_type.cLabel))
    print(np.nanmedian(spearman_corrs))

# Convert the data into a DataFrame
boxplot_df = pd.DataFrame(boxplot_data)

# Create a custom palette for the boxplot
palette = {}
for cell_type in boxplot_df['Cell Type'].unique():
    if cell_type in significant_cell_types:
        palette[cell_type] = 'red'  # Significant cell types in red
    else:
        palette[cell_type] = 'gray' # Non-significant cell types in gray

# Plot the boxplot with the custom palette
plt.figure(figsize = (6, 6))
sns.boxplot(x = 'Cell Type',
            y = 'Spearman Correlation',
            data = boxplot_df,
            palette = palette)
plt.xticks(rotation = 90)  # Rotate cell type labels for better readability
plt.title('Spearman Correlations for Each Cell Type')
plt.xlabel('Cell Type')
plt.ylabel('Spearman Correlation')
plt.tight_layout()

# Save the plot
plt.savefig(path_fig + 'celltypes_spearman_boxplot.svg',
            bbox_inches = 'tight',
            dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END