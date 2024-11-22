"""
*******************************************************************************

Script purpose:

    This script converts HCP group average functional maps into 
    borders presented in Schaefer-400 parcellation.
    Specifically, it creates border files based on a threshold on Cohen'd maps
    if more than 50% of the voxels exceed a specified threshold, it should be 
    included within the border.

Script output:

    - Border files for visualization:
        'lh.HCP_GA_functional.border'
        'rh.HCP_GA_functional.border'

    - Also save them as a numpy array
        'HCP_GA_functional.npy'

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
from neuromaps import images
from brainspace import datasets
import fsl.data.gifti as loadgifti
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from netneurotools import datasets as nntdata
from functions import parcel2fsLR, save_gifti
from globals import path_results, path_HCP_GA, path_wb_command, nnodes, path_surface

#------------------------------------------------------------------------------
# Thresholding parameter
#------------------------------------------------------------------------------

THRESHOLD = 1.0

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def cifti_to_gifti(wb_command_path, cifti_in, cifti_in_direction, convert_or_load):
    """
    Converts or loads CIFTI files into GIFTI format for left and right hemispheres.
    
    Parameters:
    - wb_command_path (str): Path to workbench command.
    - cifti_in (str): Base name of the CIFTI file.
    - cifti_in_direction (str): Path to input CIFTI file.
    - convert_or_load (int): If 1, convert and load the data. If 0, load only.
    
    Returns:
    - gifti_out_right, gifti_out_left: Loaded GIFTI files for right and left hemispheres.
    """
    if convert_or_load == 1:
        command_cifti_to_gifti = (wb_command_path + "wb_command -cifti-separate " \
                                  + cifti_in_direction+cifti_in + ".dscalar.nii" \
                                  + " COLUMN -metric CORTEX_LEFT " \
                                  + cifti_in_direction+cifti_in + "_LEFT.func.gii"\
                                  + " -metric CORTEX_RIGHT " + cifti_in_direction \
                                  + cifti_in + "_RIGHT.func.gii")
        os.system(''.join([command_cifti_to_gifti]))
    right_gifti_add = (''.join([cifti_in + "_RIGHT.func.gii"]))
    left_gifti_add = (''.join([cifti_in + "_LEFT.func.gii"]))
    gifti_out_right = images.load_data(''.join([cifti_in_direction + right_gifti_add]))
    gifti_out_left = images.load_data(''.join([cifti_in_direction + left_gifti_add]))
    return gifti_out_right,gifti_out_left

def atlas_masking(atlas_hem, data, thr):

    """
    Applies thresholding to the data
    
    Parameters:
    - atlas_hem: Schaefer atlas (gifti - 32k)
    - data: Functional data for the hemisphere (gifti - 32k)
    - threshold: Threshold value for voxel-wise activation.

    Returns:
    - results: Binary array where parcels are set to 1 if more than 50% of
               their voxels exceed the threshold, otherwise 0.
    """

    size_data = np.zeros_like(data)
    unique_labels_hem = np.sort(np.unique(atlas_hem))
    number_of_labels_hem = np.size(unique_labels_hem)

    results = np.zeros((size_data.shape[1],
                       number_of_labels_hem))
    for count, x in enumerate(unique_labels_hem, start = 0):
        for map_n in range(int(size_data.shape[1])):
            size_parcel_in_vertices = np.sum(atlas_hem == x)
            num_voxels_greater_thr_in_vertices = np.sum((data[atlas_hem == x, map_n] > thr))
            ratio_parcel = num_voxels_greater_thr_in_vertices/size_parcel_in_vertices
            results[map_n, count] =  1 if ratio_parcel > 0.5  else 0
    return results

#------------------------------------------------------------------------------
# Load HCP Group averaged fMRI activation maps (Cohen's d maps)
#------------------------------------------------------------------------------

# Convert the functional activation map to gifti (originally it is in cifti format)
cifti_to_gifti(path_wb_command,
               'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMSulc',
               path_HCP_GA,
               0)

data_HCP_L = loadgifti.loadGiftiVertexData(path_HCP_GA +
                                           'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMSulc_LEFT.func.gii')[1]
data_HCP_R = loadgifti.loadGiftiVertexData(path_HCP_GA +
                                           'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMSulc_RIGHT.func.gii')[1]

#------------------------------------------------------------------------------
# Load Schaefer parcellation
#------------------------------------------------------------------------------

schaeferL = datasets.load_parcellation('schaefer',
                                       scale = nnodes,
                                       join = False)[0]
schaeferR = datasets.load_parcellation('schaefer',
                                       scale = nnodes,
                                       join = False)[1]

#------------------------------------------------------------------------------
# Parcel out HCP activation maps based on Schaefer parcellation and thresholding
#------------------------------------------------------------------------------

masked_parcels_left = atlas_masking(schaeferL, data_HCP_L, THRESHOLD)
masked_parcels_right = atlas_masking(schaeferR, data_HCP_R, THRESHOLD)

masked_parcels = np.concatenate((masked_parcels_left[:, 1:].T,
                               masked_parcels_right[:, 1:].T))
# Save the files as a numpy array
np.save(path_results + 'HCP_GA_functional.npy', masked_parcels)

# Save the files as a gifti file
schaefer = nntdata.fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))
save_gifti(parcel2fsLR(atlas,
                       masked_parcels[:int(nnodes/2), :],
                       'L'),
              path_results + 'lh.HCP_GA_functional')

save_gifti(parcel2fsLR(atlas,
                       masked_parcels[int(nnodes/2):, :],
                       'R'),
              path_results + 'rh.HCP_GA_functional')

#------------------------------------------------------------------------------
# Create border files - for visualization
#------------------------------------------------------------------------------

command = f'{path_wb_command} -metric-label-import ' + \
          f'{os.path.join(path_results, "lh.HCP_GA_functional.func.gii")} ' + \
          f'{os.path.join(path_results, "black.txt")} ' + \
          f'{os.path.join(path_results, "lh.HCP_GA_functional.label.gii")}'
os.system(command)


command = f'{path_wb_command} -metric-label-import ' + \
          f'{os.path.join(path_results, "rh.HCP_GA_functional.func.gii")} ' + \
          f'{os.path.join(path_results, "black.txt")} ' + \
          f'{os.path.join(path_results, "rh.HCP_GA_functional.label.gii")}'
os.system(command)


# Generate border for primary motor cortex
command = f'{path_wb_command} -label-to-border ' + \
          f'{os.path.join(path_surface, "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "lh.HCP_GA_functional.label.gii")} ' +\
          f'{os.path.join(path_results, "lh.HCP_GA_functional.border")}'
os.system(command)

command = f'{path_wb_command} -label-to-border ' + \
          f'{os.path.join(path_surface, "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "rh.HCP_GA_functional.label.gii")} ' +\
          f'{os.path.join(path_results, "rh.HCP_GA_functional.border")}'
os.system(command)

#------------------------------------------------------------------------------
# END