"""
*******************************************************************************

Script purpose:

    Relate individualized atrophy maps and the clinical/behavioral manifestations of ALS

Script output:

    ----------------------------------------------------------------------

    Singular values for latent variable 0: 0.2022
    x-score and y-score Spearman correlation for latent variable 0:           0.6739
    x-score and y-score Pearson correlation for latent variable 0:           0.6505

    Singular values for latent variable 1: 0.1242
    x-score and y-score Spearman correlation for latent variable 1:           0.6237
    x-score and y-score Pearson correlation for latent variable 1:           0.6356

    ----------------------------------------------------------------------

    # Dimension behavior: 62

    ----------------------------------------------------------------------

    Variance:
       2.02214161e-01, 1.24225133e-01, 7.93832533e-02, 7.64448902e-02,
       5.52551856e-02, 5.21029436e-02, 4.08178016e-02, 3.64467384e-02,
       3.05320808e-02, 2.51178068e-02

    ----------------------------------------------------------------------

    Pvals:
       1.99800200e-03, 1.89810190e-02, 9.99000999e-04, 2.99700300e-03,
       1.39860140e-02, 9.29070929e-02, 3.60639361e-01, 9.79020979e-02,
       2.79720280e-02, 1.29870130e-02

    ----------------------------------------------------------------------

    Cortical maps corresponding to each latent varible are saved as gifti and cifti:

        GIFTI
        'lh.atropy_based_weights_cortex_lv_' + str(lv) + '.func.gii'
        'rh.atropy_based_weights_cortex_lv_' + str(lv) + '.func.gii'

        CIFTI
        'atropy_based_weights_cortex_lv_' + str(lv) + '.dscalar.gii'

    ----------------------------------------------------------------------

NOTE:

    The results coming from this script are shown in a supplementary figure.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import random
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from pyls import behavioral_pls
from neuromaps.images import dlabel_to_gifti
from netneurotools import datasets as nntdata
from neuromaps.images import load_data
from functions import (show_on_surface,
                       parcel2fsLR,
                       save_gifti,
                       load_nifti,
                       save_parcellated_data_in_Schaefer_forVis)
from pls_func import (plot_loading_bar,
                      plot_scores_and_correlations_unicolor)
from globals import path_fig, path_results, path_networks, path_atlas, path_mask, nnodes

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
random.seed(4)

#------------------------------------------------------------------------------
# Needed Functions
#------------------------------------------------------------------------------

def calculate_mean_w_score(w_score, atlas, n_parcels):
    mean_w_score = np.zeros((int(n_parcels), 1))
    for i in range(1, int(n_parcels) + 1):
        mean_w_score[i - 1, :] = np.mean(w_score[atlas == i])
    return mean_w_score

#------------------------------------------------------------------------------
# Brain data handeling
#------------------------------------------------------------------------------

# Load atlas
schafer_atlas = load_nifti(os.path.join(path_atlas, 'schaefer' + str(nnodes) + '.nii.gz'))

# Load individual-level atrophy maps
w_score = -1 * np.load(path_results + 'w_score_all.npy')

# Parcellate the individuals' atrophy maps
w_score_schaefer = np.zeros((nnodes, (len(w_score))))
for i in range(len(w_score)):
    w_score_schaefer[:, i]  = calculate_mean_w_score(w_score[i,:,:,:],
                                                    schafer_atlas, nnodes).reshape(nnodes,)
w_score_schaefer = w_score_schaefer

#------------------------------------------------------------------------------
# Clinical measure handling
#------------------------------------------------------------------------------

# Load demographic information
df =  pd.read_csv(path_results + 'df_ALS_all.csv')

handedness_mapping = {'right': 0,
                      'left': 1,
                      'ambidextrous': 2}

df['Handedness'] = df['Handedness'].map(handedness_mapping)
med = {'yes': 1,
       'no': 0
       }
df['MedicalExamination_Riluzole'] = df['MedicalExamination_Riluzole'].map(med)
df = df[(df['Diagnosis'] == 'ALS') | 
        (df['Diagnosis'].str.contains('Control'))]

df = df[((df['Visit Label'].str.contains('Visit ' + str(1))) &
             (df['Diagnosis'] == 'ALS'))].reset_index(drop = True)

filenames_to_remove = df[df['Inclusion for Cognitive Analysis'].notna()]
indices_with_caution = filenames_to_remove.index
df = df.drop(indices_with_caution).reset_index(drop = True)
w_score_schaefer = np.delete(w_score_schaefer, indices_with_caution, axis = 1)

num_subjects = len(df)
print('number of remaining patients is: ' + str(num_subjects))

#------------------------------------------------------------------------------

for i in range(int(nnodes)):
    column_name = f'parcel_{i}'
    df[column_name] = w_score_schaefer[i,:]

#------------------------------------------------------------------------------

# Get a list of columns with missing values
columns_with_missing = df.columns[df.isnull().any()]

# Filter out categorical columns from the list
numerical_columns_with_missing = columns_with_missing.intersection(df.select_dtypes(include = ['number']).columns)

# Create a dictionary to store the counts of unique subjects with missing values for each column
missing_counts = {}
# Iterate through columns with missing values
for column in numerical_columns_with_missing:
    missing_subjects = df[df[column].isnull()]['Filename'].unique()
    missing_counts[column] = len(missing_subjects)

# Add columns with zero missing values to the dictionary
numerical_columns_without_missing = df.select_dtypes(include = ['number']).columns.difference(numerical_columns_with_missing)
for column in numerical_columns_without_missing:
    missing_counts[column] = 0

# Alternatively, you can create a DataFrame from the dictionary for easier analysis
missing_counts_df = pd.DataFrame.from_dict(missing_counts,
                                           orient = 'index',
                                           columns = ['Missing Subjects'])

# Optimize the filtering conditions
limit_num_sub_with_nan = 21 # remove the feature if it was rarely recorded across subjects
conditions = ((missing_counts_df['Missing Subjects'] <= limit_num_sub_with_nan) &
      (~missing_counts_df.index.str.contains("date|Date|index|Unnamed|Visit_details|DBM|Site|Status|TimePoint")))
missing_counts_df = missing_counts_df[conditions]

# Define column groups
column_groups = {"TAP"         : ["TAP"],
                 "ECAS"        : ["ECAS"],
                 "ALSFRS"      : ["ALSFRS"],
                 "Tone"        : ["Tone"],
                 "Reflexes"    : ["Reflexes"],
                 "BIO"         : ["BIO"]}

# Create a dictionary to map groups to colors
group_colors = {'TAP'        : 'darksalmon',
                'ECAS'       : 'salmon',
                'ALSFRS'     : 'rosybrown',
                'Tone'       : 'sandybrown',
                'Reflexes'   : 'goldenrod',
                'BIO'        : 'darkgray'}

# Define multiple options for BIO group
column_groups["BIO"] = ["Age",
                        "Sex",
                        "YearsEd",
                        "Handedness",
                        "Symptom_Duration",
                        "MedicalExamination_Riluzole"]

missing_counts_df_temp = missing_counts_df
for group, substrings in column_groups.items():
    columns = missing_counts_df_temp.index[missing_counts_df_temp.index.str.contains('|'.join(substrings))]
    column_groups[group] = columns
    missing_counts_df_temp = missing_counts_df_temp[~missing_counts_df_temp.index.isin(columns)]

# Print the optimized column groups
for group, columns in column_groups.items():
    print(f"columns_{group}:", columns)

# Combine column names into a single list which is sorted now
combined_columns = [column
                    for columns in column_groups.values()
                    for column in columns]

#------------------------------------------------------------------------------
#                              X and Y for PLS
#------------------------------------------------------------------------------

# Brain data
columns_to_keep  = [col 
                    for col in df.columns
                    if 'parcel_' in col]
brain_data = df[columns_to_keep]
brain_data_array = np.array(brain_data)

# Behavioral data
behaviour_data_array = df[combined_columns]

# Behavioral data imputation
for i in enumerate(behaviour_data_array.columns):
    behaviour_data_array[i[1]].fillna(df[i[1]].median(), inplace = True)

num_behavioral_features = np.size(behaviour_data_array, 1)

X = brain_data_array
Y = np.array(behaviour_data_array)

X = stats.zscore(X, axis = 0) # Brain
Y = stats.zscore(Y, axis = 0) # Behavior

#------------------------------------------------------------------------------
#                            PLS analysis - main
#------------------------------------------------------------------------------

nspins = 1000
num_subjects = len(df)
spins = np.zeros((num_subjects, 1000))
np.random.seed(1234)
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))
spins = spins.astype(int)

pls_result = behavioral_pls(X,
                            Y,
                            n_boot = nspins,
                            n_perm = nspins,
                            permsamples = spins,
                            test_split = 0,
                            seed = 3456)

#------------------------------------------------------------------------------
# SCORES and VarExp
#------------------------------------------------------------------------------

for behavior_ind in range(np.size(Y, axis = 1)):
    for lv in range(2): # Plot Scores and variance explained figures
        title = f'Latent Variable {lv + 1}'
        column_name = (combined_columns[behavior_ind])
        colors = (behaviour_data_array[column_name] - min(behaviour_data_array[column_name])) / (max(behaviour_data_array[column_name]) - min(behaviour_data_array[column_name]))
        plot_scores_and_correlations_unicolor(lv,
                                              pls_result,
                                              title,
                                              colors,
                                              path_fig,
                                              'atrophy_based_' + column_name)

#------------------------------------------------------------------------------
# Loading plots
#------------------------------------------------------------------------------

# Generate plots for different latent variables
for lv in range(7):
    plot_loading_bar(lv,
                     pls_result,
                     combined_columns,
                     column_groups,
                     group_colors,
                     -0.5,
                     0.5,
                     path_fig,
                     'atrophy_based')

# Flip x and y in pls to get x CI
xload = behavioral_pls(Y,
                       X,
                       n_boot = 1000,
                       n_perm = 0,
                       test_split = 0,
                       seed = 0)

for lv in range(7):
    weights_cortex = xload.y_loadings

    schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
    atlas = load_data(dlabel_to_gifti(schaefer))

    # Save as gifti
    save_gifti(parcel2fsLR(atlas,
                           weights_cortex[:200,:].reshape(int(nnodes/2),np.size(weights_cortex,1)),
                           'L'),
                  path_results + 'lh.atropy_based_weights_cortex_lv_' + str(lv))

    save_gifti(parcel2fsLR(atlas,
                           weights_cortex[200:,:].reshape(int(nnodes/2),np.size(weights_cortex,1)),
                           'R'),
                  path_results + 'rh.atropy_based_weights_cortex_lv_' + str(lv))

    # Visualize here
    show_on_surface(weights_cortex[:, lv].reshape(nnodes, 1),
                    nnodes,
                    -0.1,
                    0.1)

    # Save as cifti
    save_parcellated_data_in_Schaefer_forVis(weights_cortex[:, lv],
                                             path_results,
                                            'atropy_based_weights_cortex_lv_' + str(lv))

#------------------------------------------------------------------------------
# END