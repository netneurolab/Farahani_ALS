"""
*******************************************************************************

Script purpose:

    Looking into the role of Structural Connectome disease progression
    evaluated using the Pearson corr

    Use Schaefer-800 as a validation parcellation in this case.

    Tested using:
        - spin test (spatial autocorrelation)
        - two different network null models
            degree preserving nulls
            edge-length and degree preserving nulls

Script output:

    n = 1,000; vasa

    ----------------------------------------------------------------------

    pearson correlation of node and neighbor - based on SC: 0.607325885189985

    ----------------------------------------------------------------------

    spin test p-value: 0.000999000999000999
    edge length and degree-preserving test p-value: 0.000999000999000999
    degree-preserving test p-value: 0.000999000999000999

    ----------------------------------------------------------------------

Note:

    The results are shown in Fig.S2.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from netneurotools.stats import gen_spinsamples
from scipy.spatial.distance import squareform, pdist
from functions import (match_length_degree_distribution,
                       randmio_und,
                       pval_cal,
                       load_nifti)
from globals import path_fig, path_results, path_atlas, path_sc

#------------------------------------------------------------------------------
# Functions that might be needed
#------------------------------------------------------------------------------

def load_data(path):
    '''
    Utility function to load pickled dictionary containing the data used in
    these experiments.

    Parameters
    ----------
    path: str
        File path to the pickle file to be loaded.

    Returns
    -------
    data: dict
        Dictionary containing the data used in these experiments
    '''

    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data
def plot_boxplot_combined(data, nulls, ax, boxplot_label):
    """
    Plot a single boxplot on the provided axes.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.boxplot(nulls)
    ax.set_ylim((-0.3, 0.7))
    ax.scatter([1], data,
               s = 80,
               c = 'red',
               marker = 'o',
               zorder = 2,
               edgecolor = 'black',
               linewidth = 0.5)
    ax.set_title(boxplot_label)

#------------------------------------------------------------------------------
# Load SC - noLog and Schaefer-800
#------------------------------------------------------------------------------

nnodes = 800 # Not imported from global this time as it is a validation with 800 nodes atlas
sc = load_data(path_sc +'human_SC_nolog.pickle')['adj']

# Make its diagonal elements equal to zero
sc[np.eye(nnodes).astype(bool)] = 0

# Load schaefer-800 parcellation
schafer_atlas = load_nifti(os.path.join(path_atlas, 'Schaefer800.nii.gz'))

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

null_cal = 1    # if 1 calculation of nulls should be performed
                # 0 is you want to reuse the already calculated nulls
subtype = 'all' # Options: 'all', 'spinal', 'bulbar'
nspins  = 1000
positive_color = [0.872, 0.4758, 0.449]

#------------------------------------------------------------------------------
# Load needed information for this analysis
#------------------------------------------------------------------------------

# Load atrophy data
w_score_mean = -1 * load_nifti(path_results + 'mean_w_score_' + subtype + '.nii.gz')

def calculate_mean_w_score(w_score, atlas, n_parcels):
    """
    Calculate mean w-score across parcels.
    """
    mean_w_score = np.zeros((int(n_parcels), 1))
    for i in range(1, int(n_parcels) + 1):
        mean_w_score[i - 1, :] = np.mean(w_score[atlas == i])
    return mean_w_score

disease_profile  = calculate_mean_w_score(w_score_mean, schafer_atlas, nnodes)
disease_profile = disease_profile.flatten()

#------------------------------------------------------------------------------
# Analysis: node-neighbor atrophy relationship
#------------------------------------------------------------------------------

neighbour_abnormality = np.zeros((nnodes,))
for i in range(nnodes):
    neighbour_abnormality[i] = np.nansum(disease_profile * sc[i, :])/(np.sum(sc[i, :]))

# Create the scatter plot
fig, ax =  plt.subplots(figsize = (10, 10))
sns.regplot(x = neighbour_abnormality,
            y = disease_profile,
            color = 'silver',
            scatter_kws={'s': 100, 'edgecolor': 'black', 'linewidth': 1})

# Labeling the axes
ax.set_xlim((-0.15, 0.25))
ax.set_ylim((-0.35, 0.45))
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Scatter Plot with Regression Line')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_fig + 'sc_scatterplot_w_score_800_nnodes.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

# Correlation value
val_corr, _ = pearsonr(disease_profile,
                       neighbour_abnormality)
print(val_corr)

#------------------------------------------------------------------------------
# 1. Spin tests
#------------------------------------------------------------------------------

# Info related to spin tests
coords = load_data(path_sc + 'human_SC.pickle')['coords']
nnodes = len(coords)
hemiid = np.zeros((nnodes,))
hemiid[:int(nnodes/2)] = 1
if null_cal == 1: #if you want to create the nulls
    spins = gen_spinsamples(coords,
                            hemiid,
                            n_rotate = nspins,
                            seed = 1234,
                            method = 'vasa')
    generated_null = np.zeros((nspins,))
    for n in range(nspins):
        neighbour_abnormality_null = np.zeros((nnodes,))
        disease_profile_null = disease_profile[spins[:, n]]
        for i in range(nnodes):
            neighbour_abnormality_null[i] = np.nansum(disease_profile_null * sc[i, :])/(np.sum(sc[i, :]))
        generated_null[n], _ = pearsonr(neighbour_abnormality_null, disease_profile_null)
        del disease_profile_null

    # Save the null
    np.save(path_results + 'null_spin_sc_800_nnodes.npy', generated_null)

else: # If you want to use the already saved nulls
    generated_null = np.load(path_results + 'null_spin_sc_800_nnodes.npy')

p_spin = pval_cal(val_corr, generated_null, nspins)

#------------------------------------------------------------------------------
# 2. Edge-length and degree preserving nulls
#------------------------------------------------------------------------------

coords =load_data(path_sc +'human_SC.pickle')['coords']
eu_distance = squareform(pdist(coords,
                               metric = "euclidean"))
eu_dist_rec = 1 / eu_distance
eu_dist_rec[np.eye(nnodes, dtype = bool)] = 0
distance_net = eu_dist_rec
if null_cal == 1: # If you want to create the nulls
    def process_spin(args):
        s_ind, net, distance_net, disease_profile, networks_spin_test = args
        _, net_rewired, _ = match_length_degree_distribution(net,
                                                             distance_net,
                                                             10,
                                                             nnodes*20)
        net_rewired[np.eye(nnodes).astype(bool)] = 0
        null_neighbour_abnormality = np.zeros(nnodes)
        for i in range(nnodes):  # for each node
            null_neighbour_abnormality[i] = np.nansum(disease_profile * net_rewired[i, :])/(np.sum(net_rewired[i, :]))

        null_corr, _ = pearsonr(null_neighbour_abnormality, disease_profile)
        return s_ind, net_rewired, null_neighbour_abnormality, null_corr

    nulls_1 = np.zeros((nspins, nnodes, nnodes))
    null_neighbour_abnormality_1 = np.zeros((nnodes, nspins))
    null_1 = np.zeros((nspins, 1))
    net = sc.copy()
    def wrapper(s_ind):
        args = [s_ind, net, distance_net, disease_profile, net]
        return process_spin(args)

    # Parallel execution
    results = Parallel(n_jobs = -1)(delayed(wrapper)(s_ind) for s_ind in range(nspins))
    for s_ind, result in enumerate(results):
        s_ind, net_rewired, null_neighbour_abnormality_single, null_corr_single = result
        nulls_1[s_ind, :] = net_rewired
        null_neighbour_abnormality_1[:, s_ind] = null_neighbour_abnormality_single
        null_1[s_ind, 0] = null_corr_single

    # Save the null
    np.save(path_results + 'null_degree_lenght_sc_800_nnodes.npy', null_1)

else: # If you want to use the already saved nulls
    null_1 = np.load(path_results + 'null_degree_lenght_sc_800_nnodes.npy')

pval_net_1 = pval_cal(val_corr, null_1, nspins)

#------------------------------------------------------------------------------
# 3. Degree preserving nulls
#------------------------------------------------------------------------------

if null_cal == 1: # If you want to create the nulls
    def process_spin_degree(args):
        s_ind, net, disease_profile, networks_spin_test = args
        net_rewired,_ = randmio_und(net, 10)
        net_rewired[np.eye(nnodes).astype(bool)] = 0
        null_neighbour_abnormality = np.zeros(nnodes)
        for i in range(nnodes):  # for each node
            null_neighbour_abnormality[i] = np.nansum(disease_profile * net_rewired[i, :])/(np.sum(net_rewired[i, :]))
        null_corr, _ = pearsonr(null_neighbour_abnormality, disease_profile)
        return s_ind, net_rewired, null_neighbour_abnormality, null_corr

    nulls_2 = np.zeros((nspins, nnodes, nnodes))
    null_neighbour_abnormality_2 = np.zeros((nnodes, nspins))
    null_2 = np.zeros((nspins, 1))

    def wrapper_degree(s_ind):
        args = [s_ind, net, disease_profile, net]
        print(s_ind)
        return process_spin_degree(args)

    # Parallel execution
    results = Parallel(n_jobs = -1)(delayed(wrapper_degree)(s_ind) for s_ind in range(nspins))
    for s_ind, result in enumerate(results):
        s_ind, net_rewired, null_neighbour_abnormality_single, null_corr_single = result
        nulls_2[s_ind, :] = net_rewired
        null_neighbour_abnormality_2[:, s_ind] = null_neighbour_abnormality_single
        null_2[s_ind, 0] = null_corr_single

    # Save the null
    np.save(path_results + 'null_degree_sc_800_nnodes.npy', null_2)

else: # If you want to use the already saved nulls
    null_2 = np.load(path_results + 'null_degree_sc_800_nnodes.npy')

pval_net_2 = pval_cal(val_corr, null_2, nspins)

#------------------------------------------------------------------------------
# Combined visualization
#------------------------------------------------------------------------------

# Create a figure with three nulls in it (axes)
fig, axes = plt.subplots(1, 3, figsize = (30, 10))

# Spin test nulls boxplot
plot_boxplot_combined(val_corr,
                      generated_null.reshape(nspins, 1),
                      axes[0],
                      'Spin Test')

# Edge-length and degree preserving nulls boxplot
plot_boxplot_combined(val_corr,
                      null_1,
                      axes[1],
                      'Length and Degree Preserving Nulls')

# Degree preserving nulls boxplot
plot_boxplot_combined(val_corr,
                      null_2,
                      axes[2],
                      'Degree Preserving Nulls')

plt.tight_layout()
plt.savefig(path_fig + 'sc_boxplot_all_tests_800_nnodes.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# Summary of the results
#------------------------------------------------------------------------------

print('----------------------------------------------------------------------')
print('pearson correlation of node and neighbor - based on SC: ' + str(val_corr))
print('----------------------------------------------------------------------')
print('spin test p-value: ' + str(p_spin))
print('edge length and degree-preserving test p-value: ' + str(pval_net_1))
print('degree-preserving test p-value: ' + str(pval_net_2))
print('----------------------------------------------------------------------')

#------------------------------------------------------------------------------
# END