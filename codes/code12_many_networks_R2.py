"""
*******************************************************************************

Script purpose:

    Assess the influence of adding a regressor based on biological similarity matrices
    in predicting the atrophy.
    We once built the model using the neighbour regressor coming from SC and then
    add the neighbour atrophy map coming from the inter-regional similarity matrices
    we next assess the significance of an increase in R2 as a result of adding the regressor.

Script output:

    -----------------------------------------------------------------------
    Masked by structural connectome

        0.2647764423878821
        [0.27122314]
        [0.27866305]
        [0.27403189]
        [0.33144551]
        [0.27022747]

    p_spin_corr - fdr corrected
        [0.0269946 ,
         0.0089982 ,
         0.01266413,
         0.0009998 ,
         0.02919416])

    -----------------------------------------------------------------------
    Not masked by structural connectome:

         0.26679800208993987
         [0.2663641 ]
         [0.26521575]
         [0.26647204]
         [0.27952604]
         [0.26680799]

    p_spin_corr - fdr corrected
        [0.97680464,
         0.97680464,
         0.97680464,
         0.014997  ,
         0.97680464]
    -----------------------------------------------------------------------

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import seaborn as sns
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from functions import vasa_null_Schaefer
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from globals import nnodes, path_fig, path_results, path_networks, path_sc

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype    = 'all' # Options: 'all', 'spinal', 'bulbar'
nspins     = 5000  # Number of null realizations
include_sc = 0     # Masked based on sc if 1 else 0

#------------------------------------------------------------------------------
# Load needed information
#------------------------------------------------------------------------------

# Load atrophy data
disease_profile = np.load(path_results + 'mean_w_score_' + str(subtype) + '_Schaefer.npy')
disease_profile = np.reshape(disease_profile, nnodes)

# Load structural network
sc = np.load(path_sc + 'adj_noLog.npy')
sc[np.eye(nnodes).astype(bool)] = 0  # Make diagonal elements zero

# Load other networks
gc = np.load(path_networks + 'gene_coexpression.npy')
rs = np.load(path_networks + 'receptor_similarity.npy')
ls = np.load(path_networks + 'laminar_similarity.npy')
mc = np.load(path_networks + 'metabolic_connectivity.npy')
hc = np.load(path_networks + 'haemodynamic_connectivity.npy')

# Dictionary of networks
networks = {
    "gene_coexpression": gc,
    "receptor_similarity": rs,
    "laminar_similarity": ls,
    "metabolic_connectivity": mc,
    "haemodynamic_connectivity": hc
}

#------------------------------------------------------------------------------
# Node-neighbour - based on sc
#------------------------------------------------------------------------------

for network in networks.keys():
    net_temp = networks[network]
    net_temp = np.arctanh(networks[network]) # Normalization step
    net_temp[np.eye(nnodes).astype(bool)] = 0 # Make diagonal zero
    net_temp[net_temp < 0] = 0 # Remove neg values
    sc = np.load(path_sc + 'adj_nolog.npy')

    if include_sc == 1:
        net_temp[sc <= 0] = 0

    networks[network] = net_temp

neighbour_abnormality_sc = np.zeros((nnodes,))
for i in range(nnodes):
     neighbour_abnormality_sc[i] = np.nansum(disease_profile * sc[i, :])/(np.sum(sc[i, :]))

print('node-neighbour correlation based on sc alone:')
print(pearsonr(disease_profile.flatten(),
               neighbour_abnormality_sc))

#------------------------------------------------------------------------------
# Node-neighbour - based on each inter-regional similarity network
#------------------------------------------------------------------------------

neighbour_abnormality = np.zeros((len(networks.keys()), nnodes))
print('node-neighbour correlation based on other networks:')
for n, network in enumerate(networks.keys()):
    temp_disease_profile = disease_profile # copy disease pattern into this loop

    net = networks[network].copy()
    plt.figure()
    sns.heatmap(net)
    plt.show()
    for i in range(nnodes):
         neighbour_abnormality[n, i] = np.nansum(temp_disease_profile * net[i, :])/(np.sum(net[i, :]))

    mask = ~np.isnan(neighbour_abnormality[n, :])
    neighbour_abnormality[n, :][~mask] = 0
    temp_disease_profile[~mask] = 0

    print(pearsonr(temp_disease_profile,
                   neighbour_abnormality[n, :]))
    plt.figure()
    plt.scatter(temp_disease_profile,
                           neighbour_abnormality[n, :])
    plt.show()

#------------------------------------------------------------------------------
# Perform Linear Regression using SC only, networks + SC, and networks alone
#------------------------------------------------------------------------------

y = disease_profile
x1 = neighbour_abnormality_sc
x1 = x1.reshape(1, -1).T
model = LinearRegression()
model.fit(x1, y)
yhat = model.predict(x1)
SS_Residual = sum((y-yhat)**2)
SS_Total = sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
rsq_sc = 1 - (1-r_squared)*(len(y)-1)/(len(y)-x1.shape[1]-1)

print(rsq_sc)

rsq_sc_net = np.zeros((len(networks.keys()), 1))
for n, network in enumerate(networks.keys()):
    y = disease_profile
    x1 = np.column_stack((neighbour_abnormality_sc, neighbour_abnormality[n]))
    model = LinearRegression()
    model.fit(x1, y)
    yhat = model.predict(x1)
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    rsq_sc_net[n] = 1 - (1-r_squared)*(len(y)-1)/(len(y)-x1.shape[1]-1)

print(rsq_sc_net)

rsq_net = np.zeros((len(networks.keys()), 1))
for n, network in enumerate(networks.keys()):
    y = disease_profile
    x1 = neighbour_abnormality[n]
    x1 = x1.reshape(1, -1).T
    model = LinearRegression()
    model.fit(x1, y)
    yhat = model.predict(x1)
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    rsq_net[n] = 1 - (1-r_squared)*(len(y)-1)/(len(y)-x1.shape[1]-1)

print(rsq_net)

#------------------------------------------------------------------------------
# Create the nulls - another method
#------------------------------------------------------------------------------

spins = vasa_null_Schaefer(nspins)

map_spin = np.zeros((nnodes, nspins, len(networks.keys())))
for n, network in enumerate(networks.keys()):
    for s in range(nspins):
        map_spin[:, s, n] = neighbour_abnormality[n, spins[:, s]]

rsq_spins = np.zeros((nspins, len(networks.keys())))
for n, network in enumerate(networks.keys()):
    for s in range(nspins):
        y = disease_profile
        x1 = np.column_stack((neighbour_abnormality_sc, map_spin[:, s, n]))
        mask = ~np.isnan(x1[:, 1])
        x1 = x1[mask]
        y = y[mask]
        model = LinearRegression()
        model.fit(x1, y)
        yhat = model.predict(x1)
        SS_Residual = sum((y-yhat)**2)
        SS_Total = sum((y-np.mean(y))**2)
        r_squared = 1 - (float(SS_Residual))/SS_Total
        rsq_spins[s, n] = 1 - (1-r_squared)*(len(y)-1)/(len(y)-x1.shape[1]-1)

p_spin = np.zeros(len(networks.keys()))
for n, network in enumerate(networks.keys()):
    plt.figure(figsize = (5, 5))
    plt.hist(rsq_spins[:, n],
             bins = 30,
             alpha = 0.7,
             color = 'silver')
    plt.axvline(rsq_sc_net[n],
                color = 'red',
                linestyle = '--')
    print(np.sum(rsq_spins[:,n] >rsq_sc_net[n]))
    plt.show()
    p_spin[n] = pval_cal(rsq_sc_net[n], rsq_spins[:, n], nspins)
p_spin_corr = [multipletests(p_spin, method = 'fdr_bh')][0][1]

#------------------------------------------------------------------------------
# Create the plot for visualization
#------------------------------------------------------------------------------

# Set up the figure with subplots
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 10)) # 5 networks mean 5 subplots
axes = axes.flatten()  # Flatten the array of axes for easy iteration

# Plot each network in its own subplot
for n, (network, ax) in enumerate(zip(networks.keys(), axes)):
    ax.hist(rsq_spins[:, n], bins = 30, alpha = 0.7, color = 'skyblue')
    ax.axvline(rsq_sc_net[n], color = 'red', linestyle = '--')
    ax.set_title(f'{network}')
    ax.set_xlabel('R-squared values')
    ax.set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Remove any unused subplots
for ax in axes[len(networks):]:
    fig.delaxes(ax)

plt.savefig(path_fig + '/r2_histogram_' + str(include_sc) + '.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# END