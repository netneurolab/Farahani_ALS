"""
*******************************************************************************

Purpose:

    Looking into the role of Structural Connectome disease progression
    evaluated using the Pearson corr

    Tested using:
        - spin test (spatial autocorrelation)
        - two different network null models
            degree preserving nulls
            edge-length and degree preserving nulls

# Results:
    n = 1,000; vasa
    ----------------------------------------------------------------------
    pearson correlation of node and neighbor - based on SC: 0.4610690006423789
    ----------------------------------------------------------------------
    spin test p-value: 0.001998001998001998
    degree-preserving test p-value: 0.000999000999000999
    edge length and degree-preserving test p-value: 0.04495504495504495
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
from joblib import Parallel, delayed
from netneurotools.stats import gen_spinsamples
from scipy.spatial.distance import squareform, pdist
from functions import (match_length_degree_distribution,
                       randmio_und,
                       pval_cal)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
random.seed(5647)

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path    = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_data    = os.path.join(base_path, 'data/')
path_results = os.path.join(base_path, 'results/')
path_sc      = os.path.join(base_path, 'data/SC/')
script_name  = os.path.basename(__file__).split('.')[0]
path_fig     = os.path.join(os.getcwd(), 'generated_figures/', script_name)
os.makedirs(path_fig, exist_ok = True)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

nnodes  = 400   # Schaefer-400
subtype = 'all' #'all' #'bulbar' #'spinal'
nspins  = 1000  # number of null realizations

#------------------------------------------------------------------------------
# Needed Functions
#------------------------------------------------------------------------------

positive_color = [0.872, 0.4758, 0.449]

def plot_boxplot(data, nulls, name_to_save):
    """
    Create a box plot for the null distribution
    and visualize the actual value on top
    """
    fig, axes = plt.subplots(1, 1, figsize = (10, 10))
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.boxplot((nulls))
    positions = 1
    axes.scatter(positions,
                    data,
                    s = 80,
                    c = positive_color,
                    marker = 'o',
                    zorder = 2,
                    edgecolor = 'black',
                    linewidth = 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path_fig, name_to_save + '.png'))

def plot_boxplot_combined(data, nulls, ax, boxplot_label):
    """
    Plot a single boxplot on the provided axes.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.boxplot(nulls)
    ax.set_ylim((-0.3, 0.5))
    ax.scatter([1], data,
               s = 80,
               c = 'red',
               marker = 'o',
               zorder = 2,
               edgecolor = 'black',
               linewidth = 0.5)
    ax.set_title(boxplot_label)

#------------------------------------------------------------------------------
# Load needed information for this analysis
#------------------------------------------------------------------------------

# Load atrophy data
disease_profile = np.load(path_results + 'mean_w_score_' + str(subtype) + '_Schaefer.npy')
disease_profile = np.reshape(disease_profile, nnodes)

# Load structural network
# Make its diagonal elements equal to zero
sc = np.load(path_sc + 'adj.npy')
sc[sc <= 0] = 0
sc[np.eye(nnodes).astype(bool)] = 0

#------------------------------------------------------------------------------
# Analysis: Node-Neighbor Atrophy Relationship
#------------------------------------------------------------------------------

neighbour_abnormality = np.zeros((nnodes,))

for i in range(nnodes):
    neighbour_abnormality[i] = np.nansum(disease_profile * sc[i, :])/(np.count_nonzero(sc[i, :]))

# Create the scatter plot
fig, ax =  plt.subplots(figsize = (10, 10))
sns.regplot(x = neighbour_abnormality,
            y = disease_profile,
            color = 'silver',
            scatter_kws={'s': 100, 'edgecolor': 'black', 'linewidth': 1})

# Labeling the axes
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Scatter Plot with Regression Line')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim((-0.04, 0.10))
ax.set_ylim((-0.3, 0.4))
plt.tight_layout()
plt.savefig(path_fig + 'sc_scatterplot_w_score.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# 1. Spin Tests
#------------------------------------------------------------------------------

coords = np.genfromtxt(path_data + 'schaefer_' + str(nnodes) + '.txt')
coords = coords[:, -3:]
nnodes = len(coords)
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
spins = gen_spinsamples(coords,
                        hemiid,
                        n_rotate = nspins,
                        seed = 1234,
                        method = 'vasa')

# real value
val_corr, _ = pearsonr(disease_profile,
                       neighbour_abnormality)
print(val_corr)

generated_null = np.zeros((nspins,))
for n in range(nspins):
    neighbour_abnormality_null = np.zeros((nnodes,))
    disease_profile_null = disease_profile[spins[:, n]]
    for i in range(nnodes):
        neighbour_abnormality_null[i] = np.nansum(disease_profile_null * sc[i, :])/(np.count_nonzero(sc[i, :]))
    generated_null[n], _ = pearsonr(neighbour_abnormality_null, disease_profile_null)
    del disease_profile_null

p_spin = pval_cal(val_corr, generated_null, nspins)
plot_boxplot(val_corr, generated_null.reshape(nspins, 1), 'sc_boxplot_spintest')

#------------------------------------------------------------------------------
# 2. Edge-length and degree preserving nulls
#------------------------------------------------------------------------------

eu_distance = squareform(pdist(coords, metric = "euclidean"))
eu_dist_rec = 1 / eu_distance
eu_dist_rec[np.eye(nnodes, dtype = bool)] = 0
distance_net = eu_dist_rec

def process_spin(args):
    s_ind, net, distance_net, disease_profile, networks_spin_test = args
    _, net_rewired, _ = match_length_degree_distribution(net,
                                                         distance_net,
                                                         10,
                                                         nnodes*20)
    net_rewired[np.eye(nnodes).astype(bool)] = 0
    null_neighbour_abnormality = np.zeros(nnodes)
    for i in range(nnodes):  # for each node
        null_neighbour_abnormality[i] = np.nansum(disease_profile * net_rewired[i, :])/(np.count_nonzero(net_rewired[i, :]))

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

pval_net_1 = pval_cal(val_corr, null_1, nspins)
plot_boxplot(val_corr, null_1, 'sc_boxplot_network_null_model_conservative')

#------------------------------------------------------------------------------
# 3. Degree preserving nulls
#------------------------------------------------------------------------------

def process_spin_degree(args):
    s_ind, net, disease_profile, networks_spin_test = args
    net_rewired,_ = randmio_und(net, 10)
    net_rewired[np.eye(nnodes).astype(bool)] = 0
    null_neighbour_abnormality = np.zeros(nnodes)
    for i in range(nnodes):  # for each node
        null_neighbour_abnormality[i] = np.nansum(disease_profile * net_rewired[i, :])/(np.count_nonzero(net_rewired[i, :]))
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

pval_net_2 = pval_cal(val_corr, null_2, nspins)
plot_boxplot(val_corr, null_2,' sc_boxplot_network_null_model_degree')

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
plt.savefig(path_fig + 'sc_boxplot_all_tests.svg',
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
print('degree-preserving test p-value: ' + str(pval_net_2))
print('edge length and degree-preserving test p-value: ' + str(pval_net_1))
print('----------------------------------------------------------------------')

#------------------------------------------------------------------------------
# END