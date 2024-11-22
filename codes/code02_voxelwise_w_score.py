"""
*******************************************************************************

Script purpose:

    To create a w-score map per subject for various groups like 'all', 'bulbar', 'spinal'.
    Output includes individual subjects' w-score maps and a mean w-score map per group.

    spinal onset = 140 subjects
    bulbar onset = 38 subjects
    The remaining subjects cannot be classified as a unique phenotype.

Script output:

    -----------------------------------------------------------------------

    # individuaized w-score maps (volume-based):
        'w_score_' + str(filename) + '.npy'

    -----------------------------------------------------------------------

    # group-averaged map (volume-based):
        'mean_w_score_' + str(filename) + '.nii.gz'
        'mean_w_score_all.nii.gz' is shown in Fig. 1a.

    -----------------------------------------------------------------------

Note:

    The w-score saved is raw and has ''not'' yet been multiplied by -1.
    The results coming from this script are shown in Fig. 1a, and are later used
    by 'code03_parcellate_w_score.py' script.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
from functions import (load_nifti, save_nifti)
from globals import path_mask, path_results

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

# For which subtype do you plan to perform the analysis?
subtype = 'bulbar' # Options: 'all', 'spinal', 'bulbar'

# Dimentions (d) of NIFTI file
d1      = 193
d2      = 229
d3      = 193

#------------------------------------------------------------------------------
# Load the MNI mask to only perform analysis for the regions inside the brain
#------------------------------------------------------------------------------

mni_mask = load_nifti(path_mask + 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii')

#------------------------------------------------------------------------------
# Load cleaned dataframe including subjects' information
#------------------------------------------------------------------------------

df = pd.read_csv(path_results + 'data_demographic_clean.csv')

#------------------------------------------------------------------------------
# Filter dataframe based on the disease 'subtype'
#------------------------------------------------------------------------------

if subtype =='spinal':
    df = df[(df['Region_of_Onset'] == 'upper_extremity') |
            (df['Region_of_Onset'] == 'lower_extremity') |
            (df['Region_of_Onset'] == 'upper_extremity{@}lower_extremity') |
            (df['Diagnosis'].str.contains('Control'))].reset_index(drop = True)
if subtype == 'bulbar':
    df = df[(df['Region_of_Onset'] == 'bulbar') |
            (df['Region_of_Onset'] == 'bulbar_speech') |
            (df['Region_of_Onset'] == 'bulbar_speech{@}bulbar_swallowing') |
            (df['Diagnosis'].str.contains('Control'))].reset_index(drop = True)
if subtype == 'all':
    df = df[(df['Diagnosis'] == 'ALS') |
            (df['Diagnosis'].str.contains('Control'))].reset_index(drop = True)

#------------------------------------------------------------------------------
# Specify the DBM data for healthy controls (consider only Visit 1)
#------------------------------------------------------------------------------

df_hc = df[((df['Visit Label'].str.contains('Visit 1')) &
            (df['Diagnosis'] == 'Control'))].reset_index(drop = True)

num_hc = len(df_hc)

print('----------------------------------------------------------------------')
print('number controls first visit subjects: ' + str(num_hc))
print('----------------------------------------------------------------------')

# Load data of healthy control based on the order they have in df_hc
hc_dbm = np.zeros((num_hc, d1, d2, d3))
for subject in range(num_hc):
    hc_dbm[subject,:,:,:] = nib.load(df_hc['Path'][subject]).get_fdata()

df_hc.to_csv(path_results + 'df_HC_' + str(subtype) + '.csv', index = False)

#------------------------------------------------------------------------------
# Specify the DBM data for ALS subjects (consider only Visit 1)
#------------------------------------------------------------------------------

df_als = df[((df['Visit Label'].str.contains('Visit 1')) &
             (df['Diagnosis'] == 'ALS'))].reset_index(drop = True)

num_als = len(df_als)

print('----------------------------------------------------------------------')
print('number ALS first visit ' + str(subtype) + 'subjects: ' + str(num_als))
print('----------------------------------------------------------------------')

# Load data of ALS subjects based on the order they have in df_als
als_dbm = np.zeros((num_als, d1, d2, d3))
for subject in range(num_als):
    als_dbm[subject,:,:,:] = nib.load(df_als['Path'][subject]).get_fdata()

df_als.to_csv(path_results + 'df_ALS_' + str(subtype) + '.csv', index = False)

#------------------------------------------------------------------------------
# Calculate the w-score maps - nifti verions
#------------------------------------------------------------------------------

# Create a big array to save the w-score result
w_score = np.zeros((num_als, d1, d2, d3))

# Concatenate 'Site' columns from both HC and ALS subjects
all_sites = pd.concat([df_hc['Site_x'], df_als['Site_x']])
all_sites = all_sites + 1
# Create dummy variables for all sites
all_site_dummies = pd.get_dummies(all_sites,
                                  prefix = 'Site',
                                  drop_first = False)

# Split the dummy variables back into separate ones for HC and ALS
site_dummies_hc  = all_site_dummies.iloc[:len(df_hc)]
site_dummies_als = all_site_dummies.iloc[len(df_hc):]

# Include age, sex, and site dummies in your predictors
X_hc = pd.concat([df_hc[['Age', 'Sex']],
               site_dummies_hc],
               axis = 1)
X_hc = sm.add_constant(X_hc)

X_patient = pd.concat([df_als[['Age', 'Sex']],
                       site_dummies_als],
                       axis = 1)
X_patient = sm.add_constant(X_patient)

# W‐score = (actual – expected)/SD
for voxel_x in range(d1):
    for voxel_y in range(d2):
        for voxel_z in range(d3):
            # If within brain mask, continue:
            if mni_mask[voxel_x, voxel_y, voxel_z] != 0:
                # Build a model based on data from HCs
                Y_hc = hc_dbm[:, voxel_x, voxel_y, voxel_z]
                model = sm.OLS(Y_hc, X_hc * 1).fit()
                dbm_pred = model.predict(X_hc)
                # Calculate residuals for the HC group
                residuals_hc = dbm_pred - Y_hc
                # Calculate the standard deviation of HC residuals
                sd_residuals_hc = np.std(residuals_hc)
                # Predict DBM for the patients using a model trained on HC
                dbm_pred_als = model.predict(X_patient*1)
                # Calculate w-score map
                w_score[:, voxel_x, voxel_y, voxel_z] =  \
                    (als_dbm[:, voxel_x, voxel_y, voxel_z] - dbm_pred_als) / sd_residuals_hc
    print(voxel_x)

# Save the w-score map for individuals with ALS
np.save(path_results + 'w_score_' + str(subtype) + '.npy', w_score)

# Calculate the mean w-score map across all ALS subjects
w_score_mean = np.squeeze(np.mean(w_score, axis = 0))

# Save the results as a NIFTI file - shown in Fig. 1a. fror subtype = 'all'
save_nifti(w_score_mean, 
           'mean_w_score_' + str(subtype) + '.nii.gz',
           path_mask,
           path_results)

#------------------------------------------------------------------------------
# END