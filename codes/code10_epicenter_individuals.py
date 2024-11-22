"""
*******************************************************************************

Script purpose:

    Generate Epicenter maps for individuals with ALS
    Create man and std of epicenter maps across them
    Create histogram to show node-neighbor values per ALS participant.

Script output:

    -----------------------------------------------------------------------

    In the obtained histogram:
        np.mean(subjectwise_node_neighbor) --> 0.44489506008302
        np.median(subjectwise_node_neighbor) --> 0.445385791923118

    -----------------------------------------------------------------------

    Mean Epicenter maps:

        Cifti:
        'mean_epicenteres_across_ALS_subjects'.dscalar.nii'

    -----------------------------------------------------------------------

    std Epicenter maps:

        Cifti:
        'std_epicenteres_across_ALS_subjects'.dscalar.nii'

    -----------------------------------------------------------------------

    Mean/std Epicenter maps:

        Cifti:
        'mean_by_std_epicenteres_across_ALS_subjects'.dscalar.nii'

    -----------------------------------------------------------------------

Note:

    The results coming from this script are shown as a supplementary figure.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from functions import load_nifti
from functions import save_parcellated_data_in_Schaefer_forVis
from globals import path_results, path_atlas, path_sc, nnodes, path_fig

#------------------------------------------------------------------------------
# Load dataframe including subjects' information and w-score maps
#------------------------------------------------------------------------------

df = pd.read_csv(path_results + 'data_demographic_clean.csv')
df = df[df['Visit Label'].str.contains('Visit 1') &
        (df['Diagnosis'].str.contains('ALS'))].reset_index(drop = True)
num_subjects = len(df)

w_score_individuals = -1 * np.load(path_results + 'w_score_all.npy')

#------------------------------------------------------------------------------
# Load schaefer atlas and structural connectome (noLog)
#------------------------------------------------------------------------------

schafer_atlas = load_nifti(os.path.join(path_atlas, 'schaefer400.nii.gz'))
SC_matrix = np.load(path_sc + 'adj_noLog.npy')

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def calculate_average_neighbor_atrophy(connectivity_matrix, disease_profile):
     connectivity_matrix[np.eye(nnodes).astype(bool)] = 0
     num_regions = len(disease_profile)
     average_neighbor_atrophy = np.zeros(num_regions)
     for i in range(nnodes):
         average_neighbor_atrophy[i] = np.nansum(disease_profile * connectivity_matrix[i, :])/(np.nansum(connectivity_matrix[i, :]))
     return average_neighbor_atrophy

def calculate_mean_w_score(w_score, atlas, n_parcels):
    mean_w_score = np.zeros((int(n_parcels), 1))
    for i in range(1, int(n_parcels) + 1):
        mean_w_score[i - 1, :] = np.mean(w_score[atlas == i])
    return mean_w_score

#------------------------------------------------------------------------------
# Parcellate the ALS individuals' atrophy maps
#------------------------------------------------------------------------------

w_score_schaefer = np.zeros((nnodes, num_subjects))
for i in range(num_subjects):
    w_score_schaefer[:, i]  = calculate_mean_w_score(w_score_individuals[i, :, :, :],
                                                    schafer_atlas,
                                                    nnodes).reshape(nnodes,)
    
#------------------------------------------------------------------------------ 
# Calculate epicetner maps and node-neighbour values per ALS participant
#------------------------------------------------------------------------------

subjectwise_epi = np.zeros((nnodes, num_subjects))
subjectwise_node_neighbor = np.zeros((num_subjects, 1))

for sub_ind in range(num_subjects):
    average_neighbor_atrophy_SC = calculate_average_neighbor_atrophy(SC_matrix,
                                                                     w_score_schaefer[:, sub_ind])
    subjectwise_node_neighbor[sub_ind], _ = stats.pearsonr(average_neighbor_atrophy_SC,
                                                           w_score_schaefer[:, sub_ind])
    # Calculate the rankings of nodes based on regional atrophy values
    regional_rankings = stats.rankdata(w_score_schaefer[:, sub_ind])
    # Calculate the rankings of nodes based on average neighbor atrophy values
    neighbor_rankings = stats.rankdata(average_neighbor_atrophy_SC)
    # Calculate the average ranking of each node in the two lists
    average_rankings = (regional_rankings + neighbor_rankings) / 2
    subjectwise_epi[:, sub_ind] = average_rankings
    print(sub_ind)

#------------------------------------------------------------------------------
# Show the histogram of individuals' node-neighbour values (ALS participants)
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize = (5, 5))
ax = plt.gca()
plt.hist(subjectwise_node_neighbor,
         bins = 20,
         color = 'silver')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(path_fig, 'histogram_individual_node_neighbour_values.svg'),
            bbox_inches = 'tight',
            dpi = 300,
            transparent = True)
plt.show()

#------------------------------------------------------------------------------
# Calculate the mean and standard deviation and mean/std maps for epicenters across ALS participants
#------------------------------------------------------------------------------

subjectwise_epi_mean = np.mean(subjectwise_epi,
                               axis = 1)
subjectwise_epi_std = np.std(subjectwise_epi,
                             axis = 1)
subjectwise_epi_mean_by_std = subjectwise_epi_mean / subjectwise_epi_std

# Save as cifti files
save_parcellated_data_in_Schaefer_forVis(subjectwise_epi_mean,
                                         path_results,
                                        'mean_epicenteres_across_ALS_subjects')

save_parcellated_data_in_Schaefer_forVis(subjectwise_epi_std,
                                         path_results,
                                        'std_epicenteres_across_ALS_subjects')

save_parcellated_data_in_Schaefer_forVis(subjectwise_epi_mean_by_std,
                                         path_results,
                                        'mean_by_std_epicenteres_across_ALS_subjects')

#------------------------------------------------------------------------------
# END