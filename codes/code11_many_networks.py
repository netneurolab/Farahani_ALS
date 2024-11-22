"""
*******************************************************************************

Script purpose:

    Assess the possibility of explaining the ALS atrophy based on biological
    similarity networks.

    Many networks - Metabolic connectivity has an important role.

Script output:

    ----------------------------------------------------------------------

    Results:n = 10,000 & include_sc = 0

    ----------------------------------------------------------------------

    network under study - order: dict_keys(['gene_coexpression',
                                            'receptor_similarity',
                                            'laminar_similarity',
                                            'metabolic_connectivity',
                                            'haemodynamic_connectivity'])


    pearson correlation of node and neighbor: [0.35488231
                                               0.33611294
                                               0.22102449
                                               0.453669
                                               0.24230391]

    spin test p-value: - fdr corrected [0.01549845
                                        0.01549845
                                        0.14048595
                                        0.0009999
                                        0.03024698]

    ----------------------------------------------------------------------

    Results n = 10,000 & include_sc = 1

    ----------------------------------------------------------------------

    network under study - order: dict_keys(['gene_coexpression',
                                            'receptor_similarity',
                                            'laminar_similarity',
                                            'metabolic_connectivity',
                                            'haemodynamic_connectivity'])

    pearson correlation of node and neighbor: [0.50139057
                                               0.50592875
                                               0.49190293
                                               0.57751331
                                               0.49859891]

    spin test p-value: - fdr corrected [0.00024998
                                        0.00024998
                                        0.00024998
                                        0.00024998
                                        0.00029997]
    ----------------------------------------------------------------------

Note:

    The results coming from this script are shown in Fig. 3 and also in a supplementary figure.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import random
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from functions import (show_on_surface,
                       pval_cal,
                       vasa_null_Schaefer)
from statsmodels.stats.multitest import multipletests
from globals import path_results, path_fig, path_networks, nnodes, path_sc

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
random.seed(999)

#------------------------------------------------------------------------------
# Contrasts
#------------------------------------------------------------------------------

subtype    = 'all' # Options: 'all', 'spinal', 'bulbar'
nspins     = 10000 # Number of null realizations
include_sc = 0     # If you plan on weighting the SC based on another network set to 1

positive_color = [0.872, 0.4758, 0.449]

#------------------------------------------------------------------------------
# Load atrophy data and visualize it
#------------------------------------------------------------------------------

disease_profile = np.load(path_results + 'mean_w_score_' + subtype + '_schaefer.npy')
show_on_surface(disease_profile, nnodes, -0.2, 0.2)
disease_profile = np.reshape(disease_profile, nnodes)

#------------------------------------------------------------------------------
# Load networks and combine the information into a single class "networks"
#------------------------------------------------------------------------------

gc = np.load(path_networks + 'gene_coexpression.npy')
rs = np.load(path_networks + 'receptor_similarity.npy')
ls = np.load(path_networks + 'laminar_similarity.npy')
mc = np.load(path_networks + 'metabolic_connectivity.npy')
hc = np.load(path_networks + 'haemodynamic_connectivity.npy')

networks = {"gene_coexpression"         : gc,
            "receptor_similarity"       : rs,
            "laminar_similarity"        : ls,
            "metabolic_connectivity"    : mc,
            "haemodynamic_connectivity" : hc}

#------------------------------------------------------------------------------
# Normalize networks - Use weights on the edges of the structural connectome
# Generate heatmaps for visualization purposes
#------------------------------------------------------------------------------

for network in networks.keys():

    net_temp = networks[network]
    net_temp = np.arctanh(networks[network]) # Normalization step
    net_temp[np.eye(nnodes).astype(bool)] = 0 # Make diagonal zero
    net_temp[net_temp < 0] = 0 # Remove neg values

    if include_sc == 1:
        # Load strucutral conenctome and save the mask as a png file
        sc = np.load(path_sc + 'adj_nolog.npy')
        net_temp[sc <= 0] = 0 # Filter the network based on sc
        sc[sc <= 0] = 0
        sc[sc > 0] = 1
        plt.figure()
        sns.heatmap(sc,
                    cbar=False,
                    square=True)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(path_fig + '/heatmap_binary_sc.png',
                bbox_inches = 'tight',
                dpi = 300,
                transparent = True)
        plt.show()

    networks[network] = net_temp
    plt.figure(figsize = (8, 8))
    # Std of remaining edges, across edges that actually exist
    sns.heatmap(networks[network],
                vmin= -3* np.std(net_temp[net_temp>0]),
                vmax=  3* np.std(net_temp[net_temp>0]),
                cmap='coolwarm',
                cbar=False,
                square=True)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(path_fig + '/' + network + '_heatmap_schaefer_include_sc_' + str(include_sc) +'.png',
            bbox_inches = 'tight',
            dpi = 300,
            transparent = True)
    plt.show()

#------------------------------------------------------------------------------
# Plot node-neighbor scatter plot per network
#------------------------------------------------------------------------------

nn_corrs = np.zeros((len(networks.keys()), 2))
generated_null = np.zeros((len(networks.keys()), nspins))

for n, network in enumerate(networks.keys()):
    temp_disease_profile = disease_profile # Copy disease pattern into this loop
    fig, ax =  plt.subplots(figsize = (10, 10))
    
    neighbour_abnormality = np.zeros((nnodes,))
    net = networks[network].copy()
    for i in range(nnodes):
         neighbour_abnormality[i] = np.nansum(temp_disease_profile * net[i, :])/(np.sum(net[i, :]))

    # It can happen that there is no count --> which will make nan or inf
    mask = ~np.isnan(neighbour_abnormality)
    neighbour_abnormality[~mask] = 0
    temp_disease_profile[~mask] = 0

    plt.scatter(disease_profile,
                neighbour_abnormality,
                color = 'gray',
                linewidth = 1,
                alpha = 0.7)

    nn_corrs[n, 0], _ = pearsonr(temp_disease_profile,
                           neighbour_abnormality)
    temp_disease_profile = disease_profile

    # Create spins
    spins = vasa_null_Schaefer(nspins)

    for nspin_ind in range(nspins):
        neighbour_abnormality_null = np.zeros((nnodes,)) # For each spin create a new temp
        disease_profile_null = temp_disease_profile[spins[:, nspin_ind]]
        for i in range(nnodes):
            neighbour_abnormality_null[i] = np.nansum(disease_profile_null * net[i, :])/(np.sum(net[i, :]))

        mask = ~np.isnan(neighbour_abnormality_null)
        neighbour_abnormality_null[~mask] = 0
        disease_profile_null[~mask] = 0

        generated_null[n, nspin_ind], _ = pearsonr(neighbour_abnormality_null,
                                                   disease_profile_null)

    nn_corrs[n, 1] = pval_cal(nn_corrs[n, 0], generated_null[n, :], nspins)
    plt.show()
    del mask, temp_disease_profile, net, neighbour_abnormality

# Perfrom fdr correction
nn_corrs[:, 1] = [multipletests(nn_corrs[:, 1], method = 'fdr_bh')][0][1]

def plot_heatmap(data, name_to_save):
    colormap = sns.color_palette("coolwarm")
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    sns.heatmap(data.reshape(len(networks.keys()), 1),
                annot = True,
                linewidths = 12,
                vmin = 0,
                vmax = 0.55,
                alpha = 0.9,
                cmap = colormap,
                yticklabels = networks.keys())
    plt.tight_layout()
    plt.savefig(os.path.join(path_fig, name_to_save + str(include_sc) + '.svg'),
                bbox_inches = 'tight',
                dpi = 300,
                transparent = True)

# Plot heatmap - values correlation
plot_heatmap(nn_corrs[:, 0], 'heatmap_corr_schaefer')
# Plot heatmap - values pval corrected
plot_heatmap(nn_corrs[:, 1], 'heatmap_pval_corrected_spins_schaefer')

#------------------------------------------------------------------------------
# Boxplot
#------------------------------------------------------------------------------

def plot_boxplot(data, nulls, name_to_save):
    fig, axes = plt.subplots(1, 1, figsize = (5, 10))
    # Calculate positions for both boxplots and scatter dots
    positions = np.arange((len(networks.keys()))) + 1 # Adjust the range as needed
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # Plot for Negative Gene Set
    axes.boxplot(np.squeeze(nulls[:, :].T),
                    positions = positions,
                    labels =  networks.keys(),
                    patch_artist = True,
                    boxprops = dict(facecolor = 'lightgray'))
    axes.scatter(positions,
                    data,
                    s = 80,
                    c = positive_color,
                    marker = 'o',
                    label = 'Data')
    axes.set_xticklabels(networks.keys(), rotation = 90)
    plt.tight_layout()
    plt.ylim([-0.2, 0.6])
    plt.savefig(path_fig + '/' + name_to_save + str(include_sc) + '.svg',
            bbox_inches = 'tight',
            dpi = 300,
            transparent = True)
    plt.show()

plot_boxplot(nn_corrs[:, 0], generated_null, 'boxplot_spins_schaefer')

#------------------------------------------------------------------------------
# Summary of the results
#------------------------------------------------------------------------------

print('----------------------------------------------------------------------')
print('network under study - order: ' + str(networks.keys()))
print('----------------------------------------------------------------------')
print('pearson correlation of node and neighbor: ' + str(nn_corrs[:, 0]))
print('----------------------------------------------------------------------')
print('spin test p-value: - fdr corrected ' + str(nn_corrs[:, 1]))
print('----------------------------------------------------------------------')

#------------------------------------------------------------------------------
# END