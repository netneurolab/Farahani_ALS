"""
*******************************************************************************

Purpose:

    Many networks - Metabolic connectivity has an important role.

    Results:n = 10,000 & include_sc = 0
        ----------------------------------------------------------------------
        network under study - order: dict_keys(['gene_coexpression', 'receptor_similarity', 'laminar_similarity', 'metabolic_connectivity', 'haemodynamic_connectivity'])
        ----------------------------------------------------------------------
        pearson correlation of node and neighbor: [0.35242867
                                                   0.3522382
                                                   0.22403488
                                                   0.43464916
                                                   0.2487004]
        ----------------------------------------------------------------------
        spin test p-value: - fdr corrected [0.01332001
                                            0.01248751
                                            0.13086913
                                            0.004995
                                            0.02747253]
        ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    Results n = 10,000 & include_sc = 1
        ----------------------------------------------------------------------
        network under study - order: dict_keys(['gene_coexpression', 'receptor_similarity', 'laminar_similarity', 'metabolic_connectivity', 'haemodynamic_connectivity'])
        ----------------------------------------------------------------------
        pearson correlation of node and neighbor: [0.48394705
                                                   0.48159473
                                                   0.47553752
                                                   0.51525627
                                                   0.46089281]
        ----------------------------------------------------------------------
        spin test p-value: - fdr corrected [0.00124875
                                            0.00124875
                                            0.00124875
                                            0.00124875
                                            0.001998  ]
        ----------------------------------------------------------------------
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
from IPython import get_ipython
from scipy.stats import pearsonr
from netneurotools.stats import gen_spinsamples
from statsmodels.stats.multitest import multipletests
from functions import (show_on_surface,
                       pval_cal)
#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
random.seed(999)

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path    = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_network = os.path.join(base_path,'data/Network/')
path_results = os.path.join(base_path,'results/')
path_sc      = os.path.join(base_path, 'data/SC/')
script_name  = os.path.basename(__file__).split('.')[0]
path_fig     = os.path.join(os.getcwd(), 'generated_figures/', script_name)
os.makedirs(path_fig, exist_ok = True)

#------------------------------------------------------------------------------
# Contrasts
#------------------------------------------------------------------------------

nnodes     = 400
subtype    = 'all' #'all' #'upper' #'lower' #'upper'
nspins     = 1000 # number of null realizations
include_sc = 0    # if you plan on weighting the SC based on another network

positive_color = [0.872, 0.4758, 0.449]

#------------------------------------------------------------------------------
# Load atrophy data and visualize it
#------------------------------------------------------------------------------

disease_profile = np.load(path_results + 'mean_w_score_' + subtype + '_schaefer.npy')

show_on_surface(disease_profile, nnodes, -0.2, 0.2)

disease_profile = np.reshape(disease_profile, nnodes)

#------------------------------------------------------------------------------
# Data needed for the Spin Tests
#------------------------------------------------------------------------------

coords = np.genfromtxt(base_path + 'data/schaefer_' + str(nnodes) + '.txt')
coords = coords[:, -3:]
nnodes = len(coords)
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1

#------------------------------------------------------------------------------
# Load networks and combine the information into a single class "networks"
#------------------------------------------------------------------------------

gc = np.load(path_network + 'gene_coexpression.npy')
rs = np.load(path_network + 'receptor_similarity.npy')
ls = np.load(path_network + 'laminar_similarity.npy')
mc = np.load(path_network + 'metabolic_connectivity.npy')
hc = np.load(path_network + 'haemodynamic_connectivity.npy')

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
    net_temp[np.eye(nnodes).astype(bool)] = 0 # make diagonal zero
    net_temp[net_temp < 0] = 0 # Remove neg values

    if include_sc == 1:
        # Load strucutral conenctome and save the mask as a png file
        sc = np.load(path_sc + 'adj.npy')
        net_temp[sc <= 0] = 0 # filter the network based on sc
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
    plt.figure(figsize=(8, 8))

    sns.heatmap(networks[network],
                vmin= -3* np.std(net_temp[net_temp>0]), # std of remaining edges, across edges that actually exist
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
    temp_disease_profile = disease_profile # copy disease pattern into this loop
    fig, ax =  plt.subplots(figsize = (10, 10))
    
    neighbour_abnormality = np.zeros((nnodes,))
    net = networks[network].copy()
    for i in range(nnodes):
         neighbour_abnormality[i] = np.nansum(temp_disease_profile * net[i, :])/(np.count_nonzero(net[i, :]))
    # it can happen that there is no count --> which will make nan or inf

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

    # create spins
    spins = gen_spinsamples(coords,
                            hemiid,
                            n_rotate = nspins,
                            seed = n*100,
                            method = 'vasa')

    for nspin_ind in range(nspins):
        neighbour_abnormality_null = np.zeros((nnodes,)) # for each spin create a new temp
        disease_profile_null = temp_disease_profile[spins[:, nspin_ind]]
        for i in range(nnodes):
            neighbour_abnormality_null[i] = np.nansum(disease_profile_null * net[i, :])/(np.count_nonzero(net[i, :]))

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

def plot_boxplot(data, nulls, name_to_save):
    fig, axes = plt.subplots(1, 1, figsize = (5, 10))
    # Calculate positions for both boxplots and scatter dots
    positions = np.arange((len(networks.keys()))) + 1  # Adjust the range as needed
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

# Create boxplot
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