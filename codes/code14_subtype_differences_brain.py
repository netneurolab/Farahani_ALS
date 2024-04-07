"""
*******************************************************************************

Purpose:
    Assess differences between bulbar and spinal ALS (t-test)

Difference map (x_weights):
    rh.t_values_bulbar_spinal" + str(lv)
    rh.t_values_bulbar_spinal" + str(lv)

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython import get_ipython
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from neuromaps.images import dlabel_to_gifti
from netneurotools import datasets as nntdata
from neuromaps.images import load_data
from functions import(save_gifti,
                      parcel2fsLR,
                      load_nifti)
#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path    = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_sc      = os.path.join(base_path, 'data/sc/')
path_atlas   = os.path.join(base_path, 'data/TransformedAtlases/')
path_results = os.path.join(base_path, 'results/')

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

nnodes = 400

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def calculate_average_neighbor_atrophy(connectivity_matrix, disease_profile):
     connectivity_matrix[np.eye(nnodes).astype(bool)] = 0
     connectivity_matrix[connectivity_matrix < 0] = 0 # Remove neg values
     num_regions = len(disease_profile)
     average_neighbor_atrophy = np.zeros(num_regions)
     for i in range(nnodes):
         average_neighbor_atrophy[i] = np.nansum(disease_profile * connectivity_matrix[i, :])/(np.count_nonzero(connectivity_matrix[i, :]))
     return average_neighbor_atrophy

#------------------------------------------------------------------------------
# Load atlas
#------------------------------------------------------------------------------

schafer_atlas = load_nifti(os.path.join(path_atlas, f'schaefer{nnodes}.nii.gz'))

#------------------------------------------------------------------------------
# Load SC
#------------------------------------------------------------------------------

SC_matrix = np.load(path_sc + 'adj.npy')

#------------------------------------------------------------------------------
# Load individualized w-score data
#------------------------------------------------------------------------------

w_score_bulbar  = np.load(path_results + 'w_score_bulbar.npy')
w_score_bulbar = -1 * w_score_bulbar

indices = np.arange(1, int(nnodes) + 1)
parcelwise_bulbar_w_score = np.array([np.mean(w_score_bulbar[:, schafer_atlas == i], axis=1) for i in indices]).T

w_score_spinal = np.load(path_results + 'w_score_spinal.npy')
w_score_spinal = -1 * w_score_spinal

parcelwise_spinal_w_score = np.array([np.mean(w_score_spinal[:, schafer_atlas == i], axis=1) for i in indices]).T


num_subjects_spinal = len(w_score_spinal)
num_subjects_bulbar = len(w_score_bulbar)
num_subjects = num_subjects_spinal + num_subjects_bulbar

parcelwise_w_score = np.concatenate((parcelwise_bulbar_w_score, parcelwise_spinal_w_score))

#------------------------------------------------------------------------------
# Convert individualized w-score data into epicenter likelihood maps
#------------------------------------------------------------------------------

subjectwise_epi = np.zeros((nnodes, num_subjects))

spearman_epi = np.zeros((num_subjects, 1))

for sub_ind in range(num_subjects):
    average_neighbor_atrophy_SC = calculate_average_neighbor_atrophy(SC_matrix, parcelwise_w_score[sub_ind,:])

    # Calculate the rankings of nodes based on regional atrophy values
    regional_rankings = stats.rankdata(parcelwise_w_score[sub_ind,:])

    # Calculate the rankings of nodes based on average neighbor atrophy values
    neighbor_rankings = stats.rankdata(average_neighbor_atrophy_SC)

    # Calculate the average ranking of each node in the two lists
    average_rankings = (regional_rankings + neighbor_rankings) / 2
    average_rankings[np.isnan(parcelwise_w_score[sub_ind,:])] = 0
    subjectwise_epi[:,sub_ind] = average_rankings

brain_data = pd.DataFrame()
for i in range(int(nnodes)):
    column_name = f'parcel_{i}'
    brain_data[column_name] = subjectwise_epi[i, :]

bulbar_mean = np.mean(subjectwise_epi[:,:num_subjects_bulbar], axis = 1)
spinal_mean = np.mean(subjectwise_epi[:,num_subjects_bulbar:], axis = 1)
#bulbar_mean = (bulbar_mean - np.min(bulbar_mean) ) / (np.max(bulbar_mean) - np.min(bulbar_mean))
#spinal_mean = (spinal_mean - np.min(spinal_mean) ) / (np.max(spinal_mean) - np.min(spinal_mean))
#------------------------------------------------------------------------------
p_vals = np.zeros((nnodes,))
t_vals = np.zeros((nnodes,))

for n in range(num_subjects):
    subjectwise_epi[:,n] = (subjectwise_epi[:,n] - np.min(subjectwise_epi[:,n]) ) / (np.max(subjectwise_epi[:,n]) - np.min(subjectwise_epi[:,n]))

for n in range(nnodes):
    array1 = subjectwise_epi[n,:num_subjects_bulbar]
    array2 = subjectwise_epi[n,num_subjects_bulbar:]

    # Perform the t-test
    t_stat, p_value = ttest_ind(array1, array2,equal_var=False)
    p_vals[n,] = p_value
    t_vals[n,] = t_stat

p_values_corr = multipletests(p_vals, method = 'fdr_bh')[1]
#------------------------------------------------------------------------------
# save results as gifti files

schaefer=nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

left_fslr = parcel2fsLR(atlas,
                        p_values_corr[0:int(nnodes/2),].reshape(int(nnodes/2), 1),
                        'L')
save_gifti(left_fslr, (''.join([path_results+"lh.p_values_bulbar_spinal"])))

right_fslr = parcel2fsLR(atlas
                         ,p_values_corr[int(nnodes/2):,].reshape(int(nnodes/2), 1),
                          'R')
save_gifti(right_fslr, (''.join([path_results+"rh.p_values_bulbar_spinal"])))

left_fslr = parcel2fsLR(atlas,

                        t_vals[0:int(nnodes/2),].reshape(int(nnodes/2), 1),
                        'L')
save_gifti(left_fslr, (''.join([path_results+"lh.t_values_bulbar_spinal"])))

right_fslr = parcel2fsLR(atlas
                         ,t_vals[int(nnodes/2):,].reshape(int(nnodes/2), 1),
                          'R')
save_gifti(right_fslr, (''.join([path_results+"rh.t_values_bulbar_spinal"])))

#------------------------------------------------------------------------------
# create border files for visualization purposes

p_values_corr_filtered = np.squeeze(np.int64(p_values_corr<0.05)).reshape(nnodes,1)

save_gifti(parcel2fsLR(atlas,
                       p_values_corr_filtered[:int(nnodes/2)],
                       'L'), 
           path_results + 'lh.p_values_corr_filtered')
save_gifti(parcel2fsLR(atlas,
                       p_values_corr_filtered[int(nnodes/2):],
                       'R'), 
           path_results + 'rh.p_values_corr_filtered')

path_wb_command = '/Users/asaborzabadifarahani/Downloads/workbench/bin_macosx64/wb_command'
command = f'{path_wb_command} -metric-label-import ' + \
          f'{os.path.join(path_results, "lh.p_values_corr_filtered.func.gii")} ' + \
          f'{os.path.join(path_results, "p_sig.txt")} ' + \
          f'{os.path.join(path_results, "lh.p_values_corr_filtered.label.gii")}'
# Execute the command
os.system(command)

path_wb_command = '/Users/asaborzabadifarahani/Downloads/workbench/bin_macosx64/wb_command'
command = f'{path_wb_command} -metric-label-import ' + \
          f'{os.path.join(path_results, "rh.p_values_corr_filtered.func.gii")} ' + \
          f'{os.path.join(path_results, "p_sig.txt")} ' + \
          f'{os.path.join(path_results, "rh.p_values_corr_filtered.label.gii")}'
# Execute the command
os.system(command)

path_surface = '/Users/asaborzabadifarahani/Desktop/GA/HumanCorticalParcellations_wN_6V6gD/'
# generate border for primary motor cortex
command = f'{path_wb_command} -label-to-border ' + \
          f'{os.path.join(path_surface, "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "lh.p_values_corr_filtered.label.gii")} ' +\
          f'{os.path.join(path_results, "lh.p_values_corr_filtered.border")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' + \
          f'{os.path.join(path_surface, "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "rh.p_values_corr_filtered.label.gii")} ' +\
          f'{os.path.join(path_results, "rh.p_values_corr_filtered.border")}'
# Execute the command
os.system(command)

#------------------------------------------------------------------------------
# END


