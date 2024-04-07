"""
*******************************************************************************

Purpose:
    This code will convert the group average maps of HCP into Schaefer parcellation.

    Numpy array:
        'HCP_schaefer_' + str(nnodes) + '.npy'

    Gifti file:
        lh.HCP_schaefer_' + str(nnodes)
        rh.HCP_schaefer_' + str(nnodes)

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
from neuromaps import images
from brainspace import datasets
from IPython import get_ipython
import fsl.data.gifti as loadgifti
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from netneurotools import datasets as nntdata
from functions import (parcel2fsLR,
                       save_gifti)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path      = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_results   = os.path.join(base_path, 'results/')
path_workbench = '/Users/asaborzabadifarahani/Downloads/workbench/bin_macosx64/'
path_HCP_GA    = '/Users/asaborzabadifarahani/Desktop/GA/HCP_S1200_GroupAvg_v1/'

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

nnodes = 400

#------------------------------------------------------------------------------
def cifti_to_gifti(wb_command_path, cifti_in, cifti_in_direction, convert_or_load):
    """
    convert_or_load --> if you need to convert and load the data put it 1
    if you only need to load the data put 0 in here
    """
    if convert_or_load == 1:
        command_cifti_to_gifti=(wb_command_path+"wb_command -cifti-separate "\
                                  +cifti_in_direction+cifti_in+".dscalar.nii"+" COLUMN -metric CORTEX_LEFT "\
                                  +cifti_in_direction+cifti_in+"_LEFT.func.gii"\
                                  +" -metric CORTEX_RIGHT "+cifti_in_direction+cifti_in+"_RIGHT.func.gii")
        os.system(''.join([command_cifti_to_gifti]))
    right_gifti_add= (''.join([cifti_in+"_RIGHT.func.gii"]))
    left_gifti_add= (''.join([cifti_in+"_LEFT.func.gii"]))
    gifti_out_right=images.load_data(''.join([cifti_in_direction+right_gifti_add]))
    gifti_out_left=images.load_data(''.join([cifti_in_direction+left_gifti_add]))
    return gifti_out_right,gifti_out_left

cifti_to_gifti(path_workbench,
               'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMSulc',
               path_HCP_GA, 
               0)

def atlas_masking(atlas_hem,data):
    """
    atlas = gifti - 32k
    data = gifti 32k
    """
    size_data = np.zeros_like(data)
    unique_labels_hem = np.sort(np.unique(atlas_hem))
    number_of_labels_hem = np.size(unique_labels_hem)
    mean_parcels = np.zeros((size_data.shape[1],
                           number_of_labels_hem))
    for count, x in enumerate(unique_labels_hem, start = 0):
        for map_n in range(int(size_data.shape[1])):
            mean_parcels[map_n, count] = np.mean(data[atlas_hem == x, map_n])
    
    return mean_parcels

#------------------------------------------------------------------------------
# Load HCP Group averaged fMRI activation maps (Cohen's d maps)
#------------------------------------------------------------------------------

data_HCP_L = loadgifti.loadGiftiVertexData(path_HCP_GA + 'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMSulc_LEFT.func.gii')[1]
data_HCP_R = loadgifti.loadGiftiVertexData(path_HCP_GA + 'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMSulc_RIGHT.func.gii')[1]

#------------------------------------------------------------------------------
# Load Schaefer parcellation
#------------------------------------------------------------------------------

schaeferL = datasets.load_parcellation('schaefer', scale = 400, join = False)[0]
schaeferR = datasets.load_parcellation('schaefer', scale = 400, join = False)[1]

#------------------------------------------------------------------------------
# Parcel out HCP activation maps based on the Schaefer parcellation
#------------------------------------------------------------------------------

# Save the files as a python array

mean_parcels_left = atlas_masking(schaeferL, data_HCP_L)
mean_parcels_right = atlas_masking(schaeferR, data_HCP_R)

mean_parcels = np.concatenate((mean_parcels_left[:, 1:].T ,mean_parcels_right[:,1:].T))
np.save(path_results + 'HCP_schaefer_' + str(nnodes) + '.npy', mean_parcels)


# Save the files as a gifti file

schaefer = nntdata.fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

save_gifti(parcel2fsLR(atlas,
                       mean_parcels[:int(nnodes/2),:],
                       'L'),
              path_results + 'lh.HCP_schaefer_' + str(nnodes))

save_gifti(parcel2fsLR(atlas,
                       mean_parcels[int(nnodes/2):,:],
                       'R'),
              path_results + 'rh.HCP_schaefer_' + str(nnodes))

#------------------------------------------------------------------------------
# Create border files - for visualization
#------------------------------------------------------------------------------

schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

filtered_HCP_maps = np.squeeze(np.int64([mean_parcels>1]))

save_gifti(parcel2fsLR(atlas,
                       filtered_HCP_maps[:int(nnodes/2),:],
                       'L'), 
           path_results + 'lh.filtered_HCP_maps_HCP')
save_gifti(parcel2fsLR(atlas,
                       filtered_HCP_maps[int(nnodes/2):,:],
                       'R'), 
           path_results + 'rh.filtered_HCP_maps_HCP')

path_wb_command = '/Users/asaborzabadifarahani/Downloads/workbench/bin_macosx64/wb_command'
command = f'{path_wb_command} -metric-label-import ' + \
          f'{os.path.join(path_results, "lh.filtered_HCP_maps_HCP.func.gii")} ' + \
          f'{os.path.join(path_results, "HCP_black_border.txt")} ' + \
          f'{os.path.join(path_results, "lh.filtered_HCP_maps_HCP.label.gii")}'
# Execute the command
os.system(command)

path_wb_command = '/Users/asaborzabadifarahani/Downloads/workbench/bin_macosx64/wb_command'
command = f'{path_wb_command} -metric-label-import ' + \
          f'{os.path.join(path_results, "rh.filtered_HCP_maps_HCP.func.gii")} ' + \
          f'{os.path.join(path_results, "HCP_black_border.txt")} ' + \
          f'{os.path.join(path_results, "rh.filtered_HCP_maps_HCP.label.gii")}'
# Execute the command
os.system(command)

path_surface = '/Users/asaborzabadifarahani/Desktop/GA/HumanCorticalParcellations_wN_6V6gD/'
# generate border for primary motor cortex
command = f'{path_wb_command} -label-to-border ' + \
          f'{os.path.join(path_surface, "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "lh.filtered_HCP_maps_HCP.label.gii")} ' +\
          f'{os.path.join(path_results, "lh.filtered_HCP_maps_HCP.border")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' + \
          f'{os.path.join(path_surface, "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "rh.filtered_HCP_maps_HCP.label.gii")} ' +\
          f'{os.path.join(path_results, "rh.filtered_HCP_maps_HCP.border")}'
# Execute the command
os.system(command)

#------------------------------------------------------------------------------
# END