"""
*******************************************************************************

Script purpose:

    Generate a CSV file to save correlation values of genes (from AHBA)
    and atrophy/epicenter maps.

Script output:

    Save the results as a .csv file:
        'gene_correlation_values.csv'

Note:

    The generated files are included in the supplementary information.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import scipy.io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from globals import path_results, path_gene
from statsmodels.stats.multitest import multipletests
from nilearn.datasets import fetch_atlas_schaefer_2018
from functions import (pval_cal,
                       vasa_null_Schaefer)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Load gene data
#------------------------------------------------------------------------------

diff_stability = scipy.io.loadmat(path_gene + 'gene_coexpression_ds_filtered.mat')['gene_coexpression_ds']
df_genes = scipy.io.loadmat(path_gene + 'gene_coexpression_filtered.mat')['gene_coexpression']
name_genes = scipy.io.loadmat(path_gene + 'names_genes_filtered.mat')['names']
name_genes_flattened = np.array([name[0] for name in name_genes.flatten()])
num_genes = len(name_genes.T)

#------------------------------------------------------------------------------
# Load atrophy and epicneter maps
#------------------------------------------------------------------------------

rank_data  = np.load(path_results + 'epicenter_all_rank.npy')
atrophy_data  = np.load(path_results + 'mean_w_score_all_schaefer.npy')

#------------------------------------------------------------------------------
# Calculate spearman correlation of epicenters and atrophy with gene maps
#------------------------------------------------------------------------------

spearmanr_cor_rank = np.zeros(num_genes)
spearmanr_cor_atrophy = np.zeros(num_genes)
boxplot_data = []
for gene in range(num_genes):
        spearmanr_cor_rank[gene], _ = spearmanr(rank_data,
                                                df_genes[:, gene])

        spearmanr_cor_atrophy[gene], _ =  spearmanr(atrophy_data,
                                                    df_genes[:, gene])

        boxplot_data.append({'Gene': name_genes_flattened[gene],
                             'Spearman correlation (Epicenter)': spearmanr_cor_rank[gene],
                             'Spearman correlation (Atrophy)': spearmanr_cor_atrophy[gene]})
        print(gene)

boxplot_df = pd.DataFrame(boxplot_data)

#------------------------------------------------------------------------------
# Do spin-test
#------------------------------------------------------------------------------

nnodes = 400
nspins = 1000
schaefer = fetch_atlas_schaefer_2018(n_rois = nnodes)
spins = vasa_null_Schaefer(nspins)

#------------------------------------------------------------------------------
# Calculate p-values
#------------------------------------------------------------------------------

# Get null values
nulls_rank = np.zeros((num_genes, nspins))
nulls_atrophy = np.zeros((num_genes, nspins))
for spin_ind in range(nspins):
    spinned_rank = rank_data[spins[:,spin_ind]]
    for label_ind in range(num_genes):
        nulls_rank[label_ind, spin_ind], _ = spearmanr(spinned_rank,
                                                       df_genes[:, gene])
    
    spinned_atrophy = atrophy_data[spins[:,spin_ind]]
    for label_ind in range(num_genes):
        nulls_atrophy[label_ind, spin_ind], _ = spearmanr(spinned_atrophy,
                                                       df_genes[:, gene])
    print(spin_ind)
    
# Get non-parametric p-value given the null distributions
p_value_spin_rank = np.zeros((num_genes,))
p_value_spin_atrophy = np.zeros((num_genes,))

for label_ind in range(num_genes):
    p_value_spin_rank[label_ind] = pval_cal(spearmanr_cor_rank[label_ind],
                                           nulls_rank[label_ind,:].flatten(),
                                           nspins)

    p_value_spin_atrophy[label_ind] = pval_cal(spearmanr_cor_atrophy[label_ind],
                                           nulls_atrophy[label_ind,:].flatten(),
                                           nspins)

p_value_spin_rank_corr = multipletests(p_value_spin_rank,
                                       method = 'fdr_bh')[1]

p_value_spin_atrophy_corr = multipletests(p_value_spin_atrophy,
                                          method = 'fdr_bh')[1]

# Add p-values to the boxplot_df DataFrame
boxplot_df['Uncorrected p-value (Epicenter)'] = p_value_spin_rank
boxplot_df['Corrected p-value (Epicenter)'] = p_value_spin_rank_corr
boxplot_df['Uncorrected p-value (Atrophy)'] = p_value_spin_atrophy
boxplot_df['Corrected p-value (Atrophy)'] = p_value_spin_atrophy_corr

boxplot_df.to_csv(path_results + 'gene_correlation_values.csv',
                             index = False)
#------------------------------------------------------------------------------
# Select some samples to discuss in the papar
#------------------------------------------------------------------------------

name_gene_sig = name_genes_flattened[p_value_spin_rank_corr < 0.05]
corr_gene_sig = spearmanr_cor_rank[p_value_spin_rank_corr < 0.05]

posname_gene_sig = name_gene_sig[corr_gene_sig > 0.50]
poscorr_gene_sig = corr_gene_sig[corr_gene_sig > 0.50]

pos_name_gene_sig = posname_gene_sig[np.argsort(poscorr_gene_sig)]

#------------------------------------------------------------------------------
# END