"""
*******************************************************************************

Purpose:

    Generate Epicenter maps for different disease subtypes using the ranking approach

    Epicenter likelihood maps will be saved as gifti files:
        rh.epicenters_' + subtype + '_rank'
        lh.epicenters_' + subtype + '_rank'

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import scipy.stats as stats
from IPython import get_ipython
from neuromaps.images import load_data
import matplotlib.pyplot as plt
from neuromaps.images import dlabel_to_gifti
from netneurotools.datasets import fetch_schaefer2018
from functions import (show_on_surface_and_save,
                       parcel2fsLR,
                       save_gifti)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path    = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_results = os.path.join(base_path,'results/')
path_sc      = os.path.join(base_path, 'data/SC/')
script_name  = os.path.basename(__file__).split('.')[0]
path_fig     = os.path.join(os.getcwd(), 'generated_figures', script_name)
os.makedirs(path_fig, exist_ok = True)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all' #'all' #'spinal' #'bulbar'
nnodes  = 400 

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

disease_profile = np.load(path_results + 'mean_w_score_' + subtype + '_schaefer.npy')

disease_profile = np.reshape(disease_profile, nnodes)

# Load structural connectivity
SC_matrix = np.load(path_sc + 'adj.npy')

#------------------------------------------------------------------------------
# Network atrophy
#------------------------------------------------------------------------------

def calculate_average_neighbor_atrophy(connectivity_matrix, disease_profile):
     connectivity_matrix[np.eye(nnodes).astype(bool)] = 0
     connectivity_matrix[connectivity_matrix < 0] = 0 # Remove neg values
     num_regions = len(disease_profile)
     average_neighbor_atrophy = np.zeros(num_regions)
     for i in range(nnodes):
         average_neighbor_atrophy[i] = np.nansum(disease_profile * connectivity_matrix[i, :])/(np.count_nonzero(connectivity_matrix[i, :]))
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
    plt.savefig(os.path.join(path_fig,'plot_atrophy_vs_average_neighbor_' + subtype + '.svg'), dpi = 300)
    #plt.close()

#------------------------------------------------------------------------------
# Structurally-connected networks
#------------------------------------------------------------------------------

average_neighbor_atrophy_SC = calculate_average_neighbor_atrophy(SC_matrix,
                                                                 disease_profile)
plot_atrophy_vs_average_neighbor(disease_profile,
                                 average_neighbor_atrophy_SC)

#------------------------------------------------------------------------------
# Epicentre analysis  - SC
#------------------------------------------------------------------------------

# Calculate the rankings of nodes based on regional atrophy values
regional_rankings = stats.rankdata(disease_profile)

neighbor_rankings = stats.rankdata(average_neighbor_atrophy_SC)


# Calculate the average ranking of each node in the two lists
average_rankings = (regional_rankings + neighbor_rankings) / 2
average_rankings[np.isnan(disease_profile)] = 0

np.save(path_results + 'epicenter_' + subtype + '_rank.npy',
        average_rankings)

# visualzie the potential epicenters
schaefer = fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

save_gifti(parcel2fsLR(atlas,
                       average_rankings[:int(nnodes/2)].reshape(int(nnodes/2), 1),
                       'L'),
           path_results + 'lh.epicenters_' + subtype + '_rank')
save_gifti(parcel2fsLR(atlas,
                       average_rankings[int(nnodes/2):].reshape(int(nnodes/2), 1),
                       'R'), 
           path_results + 'rh.epicenters_' + subtype + '_rank')

show_on_surface_and_save(average_rankings.reshape(nnodes, 1), nnodes, 0 , 400,
                         path_fig, 'epicenters_' + subtype + '_rank.png')

#------------------------------------------------------------------------------
# END