"""
*******************************************************************************

Script purpose:

    Generate Epicenter maps for different disease subtypes using the ranking approach

Script output:

    -----------------------------------------------------------------------

    Epicenter likelihood maps will be saved as:

        Gifti:
        'rh.epicenters_' + subtype + '_rank.func.gii'
        'lh.epicenters_' + subtype + '_rank.func.gii'

        Cifti:
        'epicenter_neighbor_rankings_' + subtype + '.dscalar.nii'

        Numpy array:
        'epicenter_' + subtype + '_rank.npy'

    -----------------------------------------------------------------------

    Pearson correlation values:

        noLog:0.5183006859750392
        Log: 0.47186638267664144

        For spinal subset of ALS subjects:
        noLog: 0.48010689068376156

        For bulbar subset of ALS subjects:
        noLog: 0.5354744135606359

    -----------------------------------------------------------------------

    'Node' atrophy ranking map:

        Cifti:
        'epicemter_regional_rankings_' + subtype + '.dscalar.nii'

    'Neighbour' atrophy ranking map:

        Cifti:
        'epicenter_neighbor_rankings_' + subtype + '.dscalar.nii'

    -----------------------------------------------------------------------

Note:

    The results coming from this script are shown in Fig.2b and Fig.6a.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from netneurotools.datasets import fetch_schaefer2018
from functions import parcel2fsLR, save_gifti
from globals import path_results, path_sc, path_fig, nnodes
from functions import save_parcellated_data_in_Schaefer_forVis

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all' # Options: 'all', 'spinal', 'bulbar'

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

# Load disease atropy map (average across ALS participants)
disease_profile = np.load(path_results + 'mean_w_score_' + subtype + '_Schaefer.npy')
disease_profile = np.reshape(disease_profile, nnodes)

# Load structural connectivity
SC_matrix = np.load(path_sc + 'adj_noLog.npy')

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def calculate_average_neighbor_atrophy(connectivity_matrix, disease_profile):
     connectivity_matrix[np.eye(nnodes).astype(bool)] = 0
     connectivity_matrix[connectivity_matrix < 0] = 0 # Remove neg values
     num_regions = len(disease_profile)
     average_neighbor_atrophy = np.zeros(num_regions)
     for i in range(nnodes):
         average_neighbor_atrophy[i] = np.nansum(disease_profile * connectivity_matrix[i, :])/(np.nansum(connectivity_matrix[i, :]))
     return average_neighbor_atrophy


def plot_atrophy_vs_average_neighbor(disease_profile, average_neighbor_atrophy):
    plt.figure() 
    plt.scatter(disease_profile,
                average_neighbor_atrophy,
                color = 'blue',
                linewidth = 1,
                alpha = 0.6)
    plt.xlabel('regional atrophy')
    plt.ylabel('average neighbor atrophy')
    # Remove NaN values from both vectors
    valid_indices = ~np.isnan(disease_profile) & ~np.isnan(average_neighbor_atrophy)
    cleaned_regional_atrophy = disease_profile[valid_indices]
    cleaned_average_neighbor_atrophy = average_neighbor_atrophy[valid_indices]
    r, p = stats.pearsonr(cleaned_regional_atrophy,
                           cleaned_average_neighbor_atrophy)
    print(r)
    plt.annotate(f'r = {r:.2f}',
                 xy = (0.7, 0.9),
                 xycoords = 'axes fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(path_fig,'plot_atrophy_vs_average_neighbor_' + subtype + '.svg'),
                dpi = 300)

#------------------------------------------------------------------------------
# Calculate atrophy of structurally connected nodes
#------------------------------------------------------------------------------

average_neighbor_atrophy_SC = calculate_average_neighbor_atrophy(SC_matrix,
                                                                 disease_profile)

# Plot the scatter plot of node and neighbour atrophy
plot_atrophy_vs_average_neighbor(disease_profile,
                                 average_neighbor_atrophy_SC)

#------------------------------------------------------------------------------
# Epicentre analysis  - SC
#------------------------------------------------------------------------------

# Calculate the rankings of nodes based on regional atrophy values
regional_rankings = stats.rankdata(disease_profile)

# Save the file - cifti - regional ranking
save_parcellated_data_in_Schaefer_forVis(regional_rankings,
                                         path_results,
                                        'epicemter_regional_rankings_' + subtype)
neighbor_rankings = stats.rankdata(average_neighbor_atrophy_SC)

# Save the file - cifti - neighbour ranking
save_parcellated_data_in_Schaefer_forVis(neighbor_rankings,
                                         path_results,
                                        'epicenter_neighbor_rankings_' + subtype)


# Calculate the average ranking of each node in the two lists --> epicenter map
average_rankings = (regional_rankings + neighbor_rankings) / 2
average_rankings[np.isnan(disease_profile)] = 0

# Calculate correlation of neighbour ranking, regional_ranking and average ranking
r_spearman_avg_neighbor = stats.spearmanr(average_rankings, neighbor_rankings)[0]
r_spearman_avg_regional = stats.spearmanr(average_rankings, regional_rankings)[0]
r_spearman_regional_neighbor = stats.spearmanr(regional_rankings, neighbor_rankings)[0]

r_pearson_avg_neighbor = stats.pearsonr(average_rankings, neighbor_rankings)[0]
r_pearson_avg_regional = stats.pearsonr(average_rankings, regional_rankings)[0]
r_pearson_regional_neighbor = stats.pearsonr(regional_rankings, neighbor_rankings)[0]


# Save the epicenter map as a numpy array
np.save(path_results + 'epicenter_' + subtype + '_rank.npy',
        average_rankings)

schaefer = fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

# Save the epicenter map as gifti file
save_gifti(parcel2fsLR(atlas,
                       average_rankings[:int(nnodes/2)].reshape(int(nnodes/2), 1),
                       'L'),
           path_results + 'lh.epicenters_' + subtype + '_rank')
save_gifti(parcel2fsLR(atlas,
                       average_rankings[int(nnodes/2):].reshape(int(nnodes/2), 1),
                       'R'), 
           path_results + 'rh.epicenters_' + subtype + '_rank')

# Save the epicenter map as cifti file
save_parcellated_data_in_Schaefer_forVis(average_rankings,
                                         path_results,
                                        'epicenters_' + subtype + '_rank')

#------------------------------------------------------------------------------
# END