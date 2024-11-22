"""
*******************************************************************************

Script purpose:

    Load and analyze w-score maps to identify significant deviations in brain parcels.

Methodology:

    Load w-score values and perform statistical analysis to identify significant deviations.

Script output:

    -----------------------------------------------------------------------

    case: "All"
        significant WM tracts are:
         ['Anterior thalamic radiation L' 'Anterior thalamic radiation R'
         'Corticospinal tract L' 'Corticospinal tract R'
         'Superior longitudinal fasciculus L' 'Superior longitudinal fasciculus R']

         pval of significant WM tracts are:
         [6.52873499e-08 2.04241615e-05
          8.83237542e-16 4.12979459e-16
          7.38599382e-04 1.67657382e-04]

         t-statistics of significant WM tracts are:
         [6.44238896 5.25742749
          9.61612418 9.83729082
          4.34222926 4.74972628]

    -----------------------------------------------------------------------

    case: "Spinal"
         significant WM tracts are:
         ['Anterior thalamic radiation L' 'Anterior thalamic radiation R'
         'Corticospinal tract L' 'Corticospinal tract R']

         pval of significant WM tracts are:
        [4.31294680e-05 9.61605193e-04 
         1.45236555e-09 7.72180899e-10]

         t-statistics of significant WM tracts are:
        [5.34324051 4.55292948 
         7.49837679 7.74089937]

    -----------------------------------------------------------------------

    case: "Bulbar"
        significant WM tracts are:
         ['Corticospinal tract L' 'Corticospinal tract R'
         'Superior longitudinal fasciculus L' 'Superior longitudinal fasciculus R']

        pval of significant WM tracts are:
        [0.00065282 0.00055218
         0.00172495 0.00534311]

        t-statistics of significant WM tracts are:
        [5.2643309  5.41064443
         4.87827547 4.40075757]

    -----------------------------------------------------------------------

    At the cortical level, the number of parcels significant after FDR is equal to 33.
    [ np.sum(p_values_corr<0.05) = 33 ]

    Before FDR the number is equal to 105 parcels.

Note:

    The loaded w-score maps are raw and not yet multiplied by -1.
    All t-maps and significance maps (both for Schaefer-400 parcels and JHU parcels)
    are saved.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import nibabel as nib
from scipy.stats import t
import matplotlib.pyplot as plt
from neuromaps.images import load_data
from nilearn.plotting import plot_glass_brain
from neuromaps.images import dlabel_to_gifti
from statsmodels.stats.multitest import multipletests
from netneurotools.datasets import fetch_schaefer2018
from functions import (parcel2fsLR,
                       save_gifti,
                       show_on_surface_and_save,
                       save_nifti,
                       save_parcellated_data_in_Schaefer_forVis)
from globals import path_results, nnodes, path_atlas, path_fig, path_mask

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all' # Options: 'all', 'spinal', 'bulbar'
n_subjects = None # This will be set after loading w_score

positive_color = [1.00, 0.73, 0.62]
negative_color = [0.74, 0.85, 1.00]
gray_color     = [0.19, 0.19, 0.19]

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def calculate_t_statistic(w_scores, parcel_indices):
    """
    Calculates t-statistics for given w-scores.
    """
    w_score_parcelwise = np.array([np.mean(w_scores[:, indices], axis = 1)
                                   for indices in parcel_indices]).T
    mean_w_score = np.mean(w_score_parcelwise, axis = 0)
    std_w_score = np.std(w_score_parcelwise, axis = 0, ddof = 1)
    standard_error = std_w_score / np.sqrt(n_subjects)
    return mean_w_score / standard_error

def plot_bar_chart(categories, values, color_map, x_label, y_label, name_png):
    """
    Generates a bar chart for the given data.
    """
    fig, ax = plt.subplots(figsize = (20, 10))
    plt.rcParams['font.family'] = 'Arial'
    colors = [color_map[val > 0] for val in values]
    ax.bar(categories, values, color = colors, alpha = 0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(gray_color)
    ax.spines['left'].set_color(gray_color)
    ax.set_xticklabels(categories, rotation = 90, color = gray_color)
    ax.set_yticklabels(ax.get_yticks(), color = gray_color)
    ax.set_xlabel(x_label, fontsize = 12, fontname = 'Arial')
    ax.set_ylabel(y_label, fontsize = 12, fontname = 'Arial')
    plt.ylim(-10, 10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(12)
    plt.tight_layout()
    plt.savefig(os.path.join(path_fig, name_png), dpi = 300)
    plt.show()

#------------------------------------------------------------------------------
# Load the w-score maps per subject
#------------------------------------------------------------------------------

w_score = -1 * np.load(path_results + 'w_score_' + subtype + '.npy')
n_subjects = w_score.shape[0]

#------------------------------------------------------------------------------
# Load atlas information
#------------------------------------------------------------------------------

schaefer_atlas = nib.load(path_atlas + 'schaefer' + str(nnodes) + '.nii.gz').get_fdata()
parcel_indices_schaefer = [schaefer_atlas == i for i in range(1, int(nnodes) + 1)]

JHU_atlas = nib.load(path_atlas + 'JHU_thr25.nii.gz').get_fdata()
nJHU = np.unique(JHU_atlas).shape[0] - 1
parcel_indices_JHU = [JHU_atlas == i for i in range(1, nJHU + 1)]

#------------------------------------------------------------------------------
# Compute t-statistics and save results
#------------------------------------------------------------------------------

#t_statistic_cortex = calculate_t_statistic(w_score, parcel_indices_schaefer)

w_score_parcelwise = np.array([np.mean(w_score[:, indices], axis = 1)
                               for indices in parcel_indices_schaefer]).T
mean_w_score = np.mean(w_score_parcelwise, axis = 0)
std_w_score = np.std(w_score_parcelwise, axis = 0, ddof = 1)
standard_error = std_w_score / np.sqrt(n_subjects)
t_statistic_cortex =  mean_w_score / standard_error

np.save(os.path.join(path_results, f't_statistics_cortex_{subtype}_schaefer_{nnodes}.npy'),
        t_statistic_cortex)

t_statistic_subcortex = calculate_t_statistic(w_score, parcel_indices_JHU)
np.save(os.path.join(path_results, f't_statistics_subcortex_{subtype}.npy'),
        t_statistic_subcortex)

#------------------------------------------------------------------------------
# P-value adjustment and significant indices calculation
#------------------------------------------------------------------------------

degrees_of_freedom = n_subjects - 1

p_values = [t.sf(np.abs(t_val), degrees_of_freedom) * 2
            for t_val in np.concatenate((t_statistic_subcortex, t_statistic_cortex))]

p_values_corr = multipletests(p_values, method = 'fdr_bh')[1]
significant_indices = np.where(p_values_corr <= 0.05)[0]

#------------------------------------------------------------------------------
# Mark significant values and visualization
#------------------------------------------------------------------------------

sig_values_cortex = np.zeros((nnodes, 1))
sig_values_subcortex = np.zeros((nJHU, 1))
p_corr_cortex = np.zeros((nnodes, 1))
p_corr_subcortex = np.zeros((nJHU, 1))

for index in significant_indices:
    if index <= (nJHU - 1):
        sig_values_subcortex[index, 0] = 1
        p_corr_subcortex[index, 0] = p_values_corr[index]
    if index > (nJHU - 1):
        index1 = index - (nJHU)
        sig_values_cortex[index1, 0] = 1
        p_corr_cortex[index1, 0] = p_values_corr[index1]

#------------------------------------------------------------------------------
#                     Save results, show on surface, etc.
#------------------------------------------------------------------------------

#------------------------------ t statistics ----------------------------------

schaefer = fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

# gifti format
save_gifti(parcel2fsLR(atlas,
                       t_statistic_cortex[:int(nnodes/2)].reshape(int(nnodes/2), 1),
                       'L'),
           path_results + 'lh.t_statistic_' + subtype + '_schaefer_' + str(nnodes))
save_gifti(parcel2fsLR(atlas,
           t_statistic_cortex[int(nnodes/2):].reshape(int(nnodes/2), 1),
           'R'),
           path_results + 'rh.t_statistic_' + subtype +'_schaefer_' + str(nnodes))

# cifti format
save_parcellated_data_in_Schaefer_forVis(t_statistic_cortex,
                                         path_results,
                                        't_statistic_cortex_' + subtype + '_schaefer')

# visualization here
show_on_surface_and_save(t_statistic_cortex.reshape(nnodes,1), nnodes, -5, 5,
                         path_fig, 't_statistic_' + subtype + '_schaefer_' + str(nnodes) + '.png')

# Save results for WM tracts as a nifti file
JHU_atlas_copy = JHU_atlas.copy()
for n in range(1, nJHU + 1):
    JHU_atlas_copy[JHU_atlas_copy == n] = t_statistic_subcortex[n - 1]
save_nifti(JHU_atlas_copy,
           't_statistic_subcortex_' + subtype + '.nii.gz',
           path_mask,
           path_results)

# Visualize the results as a glass brain
template = nib.load(path_results + 't_statistic_subcortex_' + subtype + '.nii.gz')
plt.figure()
plot_glass_brain(template,
                 colorbar       = True,
                 plot_abs       = False,
                 vmax           = 5,
                 vmin           = -5,
                 alpha          = 0.4,
                 symmetric_cbar = True,
                 cmap           = "coolwarm")
plt.savefig(os.path.join(path_fig,'t_statistic_subcortex_' + subtype + '.png'), dpi = 300)
plt.close()

#------------------------------------------------------------------------------
#-------------------------------- significance --------------------------------
#------------------------------------------------------------------------------

# gifti format
save_gifti(parcel2fsLR(atlas,
                       sig_values_cortex[:int(nnodes/2)].reshape(int(nnodes/2), 1),
                       'L'),
           path_results + 'lh.sig_values_cortex_' + subtype + '_schaefer_' + str(nnodes))
save_gifti(parcel2fsLR(atlas,
                       sig_values_cortex[int(nnodes/2):].reshape(int(nnodes/2), 1),
                       'R'),
           path_results + 'rh.sig_values_cortex_' + subtype + '_schaefer_' + str(nnodes))

# cifti format
save_parcellated_data_in_Schaefer_forVis(sig_values_cortex,
                                         path_results,
                                        'sig_values_cortex_' + subtype + '_schaefer')

# visualization here
show_on_surface_and_save(sig_values_cortex.reshape(nnodes, 1), nnodes, 1 , 1,
                         path_fig, 'sig_values_cortex_' + subtype + '_schaefer_' + str(nnodes) + '.png')

# Save results for WM tracts as a nifti file
JHU_atlas_copy = JHU_atlas.copy()
for n in range(1, nJHU + 1):
    JHU_atlas_copy[JHU_atlas_copy == n] = sig_values_subcortex[n - 1, 0]
save_nifti(JHU_atlas_copy,
           'sig_values_subcortex_' + subtype + '.nii.gz',
           path_mask,
           path_results)

# Visualize the results as a glass brain
template = nib.load(path_results + 'sig_values_subcortex_' + subtype + '.nii.gz')
plt.figure()
plot_glass_brain(template,
                 colorbar       = True,
                 plot_abs       = False,
                 vmax           = 5,
                 vmin           = -5,
                 alpha          = 0.4,
                 symmetric_cbar = True,
                 cmap           = "coolwarm")
plt.savefig(os.path.join(path_fig, 'sig_values_subcortex_' + subtype + '.png'),
            dpi = 300)
plt.close()

#------------------------------------------------------------------------------
# Name of the tracts
#------------------------------------------------------------------------------

name_columns = ['Anterior thalamic radiation L',
                'Anterior thalamic radiation R',
                'Corticospinal tract L',
                'Corticospinal tract R',
                'Cingulum (cingulate gyrus) L',
                'Cingulum (cingulate gyrus) R',
                'Cingulum (hippocampus) L',
                'Cingulum (hippocampus) R',
                'Forceps major',
                'Forceps minor',
                'Inferior fronto-occipital fasciculus L',
                'Inferior fronto-occipital fasciculus R',
                'Inferior longitudinal fasciculus L',
                'Inferior longitudinal fasciculus R',
                'Superior longitudinal fasciculus L',
                'Superior longitudinal fasciculus R',
                'Uncinate fasciculus L',
                'Uncinate fasciculus R',
                'Superior longitudinal fasciculus (temporal part) L',
                'Superior longitudinal fasciculus (temporal part) R']

#------------------------------------------------------------------------------
# Sorting data (WM atrophy t-statistics) to create a barplot
#------------------------------------------------------------------------------

t_statistic_subcortex = np.reshape(t_statistic_subcortex, nJHU)
sorted_indices = np.argsort(t_statistic_subcortex)
sorted_values = t_statistic_subcortex[sorted_indices]
sorted_categories = np.array(name_columns)[sorted_indices]

print('\n\nsignificant WM tracts are:\n\n',
      str(np.array(name_columns)[(sig_values_subcortex == 1).reshape(nJHU,)]))
print('\n\n pval of significant WM tracts are:\n\n',
      str(p_corr_subcortex[sig_values_subcortex == 1]))
print('\n\n t-statistics of significant WM tracts are:\n\n',
      str(t_statistic_subcortex[sig_values_subcortex.flatten() == 1]))

plot_bar_chart(sorted_categories,
               sorted_values,
               {True: positive_color, False: negative_color},
               'white matter tracts',
               'atrophy (t-statistics)',
               'barplot_t_statistic_subcortex_' + subtype + '.png')

#------------------------------------------------------------------------------
# END