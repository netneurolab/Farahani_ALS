"""
*******************************************************************************

Script purpose:

    SIR modeling to find epicenters


Script output:

    -----------------------------------------------------------------------
    Epicenter likelihood maps:

        GIFTI:
        rh.epicenters_' + subtype + '_SIR.func.gii'
        lh.epicenters_' + subtype + '_SIR.func.gii'

        CIFTI:
        'epicenters_' + subtype + '_SIR.dscalar.nii'

        Numpy array:
        'epicenters_' + subtype + '_SIR.npy'

    -----------------------------------------------------------------------

    Max pearson correlation between simulated and real atrophy for the two top parcels:

    All:
        area 49 --> 0.32769573252403916
        area 289 --> 0.29743832721449964

    Spinal:
        area 49 --> 0.2811815538935223
        area 332 --> 0.2507930290826092

    Bulbar:
        area 244 --> 0.3303722300376202
        area 247 -->  0.34368817393018025

    -----------------------------------------------------------------------

NOTE:

    The results coming from this script are shown in Fig. 2c and also in Fig. 6a.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import simulated_atrophy as sim
from scipy.stats import pearsonr
from netneurotools.datasets import fetch_schaefer2018
from neuromaps.images import load_data, dlabel_to_gifti
from functions import (save_gifti,
                       parcel2fsLR,
                       save_parcellated_data_in_Schaefer_forVis)
from globals import nnodes, path_fig, path_results, path_sc

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

load_atrophy = 1 # = 1 if simulate_atrophy is just to be loaded and is already calculated
subtype = 'all' # Options: 'all', 'spinal', 'bulbar'
timesteps = 10000 # Simulation timesteps

#------------------------------------------------------------------------------
# Needed Functions
#------------------------------------------------------------------------------

def count_unique_values(arr):
    """Counts unique values in an array."""
    counts = {}
    for value in arr:
        counts[value] = counts.get(value, 0) + 1
    return counts

#------------------------------------------------------------------------------

schaefer = fetch_schaefer2018('fslr32k')[f'{nnodes}Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

counts = count_unique_values(atlas.ravel())
counts_list = list(counts.items())
sorted_counts = sorted(counts_list, key=lambda x: x[0])
ROI_size = np.array(sorted_counts)[1:, 1]

#------------------------------------------------------------------------------
# Load structural connectivity data
#------------------------------------------------------------------------------

Structural_connectivity_Den = np.load(path_sc + 'adj_noLog.npy')
Structural_connectivity_Len = np.load(path_sc + 'len.npy')

data = {'SC_den'  : Structural_connectivity_Den,
        'SC_len'  : Structural_connectivity_Len,
        'roi_size': ROI_size}

#------------------------------------------------------------------------------
# Simulate atrophy while seeding different parcels
#------------------------------------------------------------------------------

if load_atrophy == 0:
    atrophy_seed = np.zeros((nnodes, nnodes, timesteps))
    for seed_id in range (nnodes):
        atrophy_seed[seed_id,:,:] = sim.simulate_atrophy(data['SC_den'],
                                           data['SC_len'],
                                           seed_id,
                                           data['roi_size'],
                                           dt = 0.02,
                                           p_stay = 0.99,
                                           k1 = 0.5,
                                           v = 1,
                                           T_total = timesteps)
        print(seed_id)
    np.save(path_results + 'atrophy_SIR.npz', atrophy_seed)
else: # if already calculated, just load it
    atrophy_seed = np.load(path_results + 'atrophy_SIR.npz.npy')

#------------------------------------------------------------------------------
# Load real atrophy
#------------------------------------------------------------------------------

disease_profile = np.load(path_results + 'mean_w_score_' + subtype + '_schaefer.npy')
# Compute correlations between real atrophy and simulated atrophy

corr_array = np.zeros((nnodes, timesteps))
for seed_id in range(nnodes):
    for n in range(timesteps):
        corr_array[seed_id,n], _ = pearsonr(atrophy_seed[seed_id,:,n].flatten(),
                                            disease_profile.flatten())
    print(seed_id) # To show the process

#------------------------------------------------------------------------------
# Visualize the similarity between the actual and the stimulated atrophy pattern
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
plt.rcParams['font.family'] = 'Arial'
positive_color = [0.87, 0.47, 0.45]
gray_color     = [0.19, 0.19, 0.19]

# Determine the top two parcels
max_values = [max(corr_array[seed_id, :]) for seed_id in range(nnodes)]

top_two_indices = sorted(range(len(max_values)),
                         key=lambda i: max_values[i],
                         reverse = True)[:2]

for seed_id in range(nnodes):
    color = 'red' if seed_id == top_two_indices[0] else ([1, 0.4, 0.1]
                                                         if seed_id == top_two_indices[1] else gray_color)
    alpha = 1.0 if seed_id in top_two_indices else 0.2
    plt.plot(corr_array[seed_id, :],
             color = color,
             alpha = alpha)

# Highlight maximum values with vertical lines
for seed_id in range(nnodes):
    color = 'red' if seed_id == top_two_indices[0] else ([1, 0.4, 0.1]
                                                         if seed_id == top_two_indices[1] else gray_color)
    if seed_id in top_two_indices:
        # Find the index of the maximum value for this seed_id
        max_index = np.argmax(corr_array[seed_id, :])
        print(np.max(corr_array[seed_id, :]))
        print(seed_id)
        plt.axvline(x = max_index,
                    color = color,
                    alpha = 0.8,
                    linestyle = '--')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(gray_color)
ax.spines['left'].set_color(gray_color)
[ax.get_xticklabels()[i].set_color(gray_color) for i in range(len(ax.get_xticklabels()))]
[ax.get_yticklabels()[i].set_color(gray_color) for i in range(len(ax.get_yticklabels()))]
eps_filename = 'SIR_model' + subtype + '.svg'
plt.savefig(os.path.join(path_fig, eps_filename), format = 'svg', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Find the maximum correlation value for each parcel and sort them accordingly
# This process will lead to an epicenter cortical map
#------------------------------------------------------------------------------

max_values = np.max(corr_array, axis = 1)

# Pair region IDs with their max values
region_ids = range(nnodes)
paired = list(zip(region_ids, max_values))

# Sort based on the max values in descending order
sorted_regions = sorted(paired,
                        key = lambda x: x[1],
                        reverse = True)

# Extract sorted region IDs
sorted_region_ids = [region[0] for region in sorted_regions]
c = 1
array_to_save = np.zeros((nnodes, 1))
for i in range(nnodes):
    j = sorted_region_ids[i]
    array_to_save[j, 0] = c
    c = c + 1

# Save as numpy array
np.save(path_results + 'epicenters_' + subtype + '_SIR.npy', array_to_save)

# Save as gifti
save_gifti(parcel2fsLR(atlas,
                       array_to_save[:int(nnodes/2)].reshape(int(nnodes/2), 1),
                       'L'),
              path_results + 'lh.epicenters_' + subtype +  '_SIR')
save_gifti(parcel2fsLR(atlas,
                       array_to_save[int(nnodes/2):].reshape(int(nnodes/2), 1),
                       'R'),
              path_results + 'rh.epicenters_' + subtype + '_SIR')

# Save as cifti
save_parcellated_data_in_Schaefer_forVis(array_to_save,
                                         path_results,
                                        'epicenters_' + subtype + '_SIR')
#------------------------------------------------------------------------------
# END