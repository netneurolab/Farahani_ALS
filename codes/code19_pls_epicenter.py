"""
*******************************************************************************

Script purpose:

    Relate individualized epicenter maps and the clinical/behavioral manifestations of ALS

Script output:

    ----------------------------------------------------------------------

    Singular values for latent variable 0: 0.2489
    x-score and y-score Spearman correlation for latent variable 0:           0.5383
    x-score and y-score Pearson correlation for latent variable 0:           0.5391

    Singular values for latent variable 1: 0.1221
    x-score and y-score Spearman correlation for latent variable 1:           0.5003
    x-score and y-score Pearson correlation for latent variable 1:           0.5663

    ----------------------------------------------------------------------

    # Dimension behavior: 62

    ----------------------------------------------------------------------

    Variance:
        2.48865619e-01, 1.22113034e-01, 8.25124337e-02, 7.34686033e-02,
        5.78083723e-02, 4.44483317e-02, 3.97440802e-02 ...

    ----------------------------------------------------------------------

    Pvals:
        9.99000999e-04, 4.89510490e-02, 9.99000999e-04, 2.99700300e-03,
        5.99400599e-03, 2.69730270e-02, 1.39860140e-02, ...

    ----------------------------------------------------------------------

    Cross validation:

    pval is (for lv = 0): 0.039603960396039604

    np.mean(flat_test)
    0.19861396708598433
    
    np.mean(flat_test_per)
    -0.01474278876466783

    ----------------------------------------------------------------------

    Cortical maps corresponding to each latent varible are saved as gifti and cifti:

        GIFTI
        'lh.weights_cortex_lv_' + str(lv) + '.func.gii'
        'rh.weights_cortex_lv_' + str(lv) + '.func.gii'

        CIFTI
        'weights_cortex_lv_' + str(lv) + '.dscalar.nii'

    ----------------------------------------------------------------------

Note:

    The results coming from this script are shown in Fig. 5 and also in a supplementary figure.

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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
from globals import path_results, path_fig, path_atlas, path_sc, nnodes

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
random.seed(4)

#------------------------------------------------------------------------------
# Needed Functions
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
# Brain data handeling
#------------------------------------------------------------------------------

# Load atlas
schafer_atlas = load_nifti(os.path.join(path_atlas, 'schaefer' + str(nnodes) + '.nii.gz'))

# Load SC
SC_matrix = np.load(path_sc + 'adj_noLog.npy')

# Load individual-level atrophy maps
w_score = -1 * np.load(path_results + 'w_score_all.npy')

# Parcellate the individuals' atrophy maps
w_score_schaefer = np.zeros((nnodes, (len(w_score))))
for i in range(len(w_score)):
    w_score_schaefer[:, i]  = calculate_mean_w_score(w_score[i,:,:,:],
                                                    schafer_atlas, nnodes).reshape(nnodes,)

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

subjectwise_epi = np.zeros((nnodes, num_subjects))

spearman_epi = np.zeros((num_subjects, 1))

for sub_ind in range(num_subjects):
    average_neighbor_atrophy_SC = calculate_average_neighbor_atrophy(SC_matrix,
                                                                     w_score_schaefer[:, sub_ind])
    # Calculate the rankings of nodes based on regional atrophy values
    regional_rankings = stats.rankdata(w_score_schaefer[:, sub_ind])
    # Calculate the rankings of nodes based on average neighbor atrophy values
    neighbor_rankings = stats.rankdata(average_neighbor_atrophy_SC)
    # Calculate the average ranking of each node in the two lists
    average_rankings = (regional_rankings + neighbor_rankings) / 2
    subjectwise_epi[:, sub_ind] = average_rankings

for i in range(int(nnodes)):
    column_name = f'parcel_{i}'
    df[column_name] = subjectwise_epi[i, :]

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
                                              column_name)

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
                     'epicenter_based')


# flip x and y in pls to get x CI
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
                           weights_cortex[:200, lv].reshape(int(nnodes/2),np.size(weights_cortex,1)),
                           'L'),
                  path_results + 'lh.weights_cortex_lv_' + str(lv))

    save_gifti(parcel2fsLR(atlas,
                           weights_cortex[200:, lv].reshape(int(nnodes/2),np.size(weights_cortex,1)),
                           'R'),
                  path_results + 'rh.weights_cortex_lv_' + str(lv))

    # Visualize here
    show_on_surface(weights_cortex[:, lv].reshape(nnodes, 1),
                    nnodes,
                    -0.1,
                    0.1)

    # Save as cifti
    save_parcellated_data_in_Schaefer_forVis(weights_cortex[:, lv],
                                             path_results,
                                            'weights_cortex_lv_' + str(lv))

#------------------------------------------------------------------------------
# Cross-validation for the PLS analysis
#------------------------------------------------------------------------------

n_splits = 2 # 2-fold
np.random.seed(1234)
lv = 0
nperm  = 100
def cv_cal(X, Y):
    corr_test = np.zeros((n_splits, nperm))
    corr_train = np.zeros((n_splits, nperm))

    for iter_ind in range(nperm):
        kf = KFold(n_splits = n_splits, shuffle = True, random_state=iter_ind)
        c = 0
        for train_index, test_index in kf.split(X):

            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]

            train_result = behavioral_pls(Xtrain,
                                          Ytrain,
                                          n_boot = 0,
                                          n_perm = 0,
                                          test_split = 0,
                                          seed = 10)
            corr_train[c, iter_ind], _ = stats.pearsonr(train_result['x_scores'][:, lv],
                                            train_result['y_scores'][:, lv])

            # project weights, correlate predicted scores in the test set
            corr_test[c, iter_ind], _ = stats.pearsonr(Xtest @ train_result['x_weights'][:, lv],
                                   Ytest @ train_result['y_weights'][:, lv])
            c = c + 1
    return(corr_train, corr_test)

corr_train, corr_test = cv_cal(X,Y) 

stats.pearsonr(X @ pls_result['x_weights'][:, 0], Y @ pls_result['y_weights'][:, 0])
stats.pearsonr(pls_result['y_scores'][:, 0], pls_result['y_scores'][:, 0])

# VISUALIZATION ---------------------------------------------------------------

flat_train = corr_train[0, :].flatten()
flat_test  = corr_test[0, :].flatten()

combined_train_test = [flat_train, flat_test]
plt.boxplot(combined_train_test)
plt.xticks([1, 2], ['train', 'test'])

# Remove the box around the figure
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax = plt.gca()
y_ticks = np.linspace(-0.5, 1, num = 5)
ax.set_yticks(y_ticks)
plt.savefig(path_fig + 'train_test_CV_' + str(lv) +'.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# Permutation
#------------------------------------------------------------------------------

def single_cv_cal(X, Y, val_random):
    corr_test = np.zeros((n_splits, 1))
    corr_train = np.zeros((n_splits, 1))
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = val_random)
    c = 0
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        train_result = behavioral_pls(Xtrain,
                                      Ytrain,
                                      n_boot = 0,
                                      n_perm = 0,
                                      test_split = 0,
                                      seed = 10)
        corr_train[c, 0], _ = stats.pearsonr(train_result['x_scores'][:, lv],
                                        train_result['y_scores'][:, lv])

        # project weights, correlate predicted scores in the test set
        corr_test[c, 0], _ = stats.pearsonr(Xtest @ train_result['x_weights'][:, lv],
                               Ytest @ train_result['y_weights'][:, lv])
        c = c + 1
    return(corr_train.flatten(), corr_test.flatten())

per_train_corr = np.zeros((n_splits, nperm))
per_test_corr = np.zeros((n_splits, nperm))

num_subjects = len(df)
perms_y = np.zeros((num_subjects, nperm))

for perm_ind in range(nperm):
    perms_y[:, perm_ind] = np.random.permutation(range(0, num_subjects))

for perm_ind in range(nperm):
    tempy = perms_y[:, perm_ind].astype(int)
    Y_permuted = Y[tempy]

    per_train_corr[:, perm_ind], per_test_corr[:, perm_ind] = single_cv_cal(X, Y_permuted, perm_ind)
    print(perm_ind)

# VISUALIZATION ---------------------------------------------------------------

flat_train = corr_train[0, :].flatten()
flat_test  = corr_test[0, :].flatten()

flat_train_per = per_train_corr[0, :].flatten()
flat_test_per  = per_test_corr[0, :].flatten()


p_val = (1 + np.count_nonzero(flat_test_per  > np.mean(flat_test))) / (nperm + 1)

print('pval is (for lv = ' + str(lv) + '): ' + str(p_val))

combined_train_test = [flat_train, flat_train_per, flat_test, flat_test_per]

plt.boxplot(combined_train_test)
plt.xticks([1, 2, 3, 4], ['train', 'train permute', 'test', 'test permute'])
# Remove the box around the figure
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax = plt.gca()
y_ticks = np.linspace(-0.5, 1, num = 5)
ax.set_yticks(y_ticks)
plt.savefig(path_fig + 'train_test_per_CV_' + str(lv) +'.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# END