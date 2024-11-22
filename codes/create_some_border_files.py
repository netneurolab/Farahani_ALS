"""
*******************************************************************************

Script purpose:

    This code is used to generate the border files
    border files are used later on to create figures using wb_view

    To create these border files, ``wb_command'' is used!

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
from functions import save_gifti
import fsl.data.gifti as loadgifti
from netneurotools.datasets import fetchers
from globals import path_results, path_surface, path_wb_command

#------------------------------------------------------------------------------
# Create a border for Broca's area
#------------------------------------------------------------------------------

# glasser parcellation
atlas_mml = fetchers.fetch_mmpall(version='fslr32k')
data_L = loadgifti.loadGiftiVertexData(atlas_mml[0])[1]

# area 44 and area 45
broca = np.int32(1*((data_L==74+180))) + np.int32(1*((data_L == 75+180)))
save_gifti(broca, path_results + 'lh.broca')

command = f'{path_wb_command} -metric-label-import ' +\
          f'{os.path.join(path_results, "lh.broca.func.gii")} ' +\
          f'{os.path.join(path_results, "broca.txt")} ' +\
          f'{os.path.join(path_results, "lh.broca.label.gii")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' +\
          f'{os.path.join(path_surface, "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "lh.broca.label.gii")} ' +\
          f'{os.path.join(path_results, "lh.broca.border")}'
# Execute the command
os.system(command)

#------------------------------------------------------------------------------
# Create top epicenter areas - ranking
#------------------------------------------------------------------------------

epi_rank_L = path_results + 'lh.epicenters_all_rank.func.gii'
data_epi_rank_L = loadgifti.loadGiftiVertexData(epi_rank_L)[1]

top_epi_rank_L = np.int32(1*((data_epi_rank_L==np.max(data_epi_rank_L.T))))
save_gifti(top_epi_rank_L, path_results + 'lh.top_epi_rank_L')

command = f'{path_wb_command} -metric-label-import ' +\
          f'{os.path.join(path_results, "lh.top_epi_rank_L.func.gii")} ' +\
          f'{os.path.join(path_results, "black.txt")} ' +\
          f'{os.path.join(path_results, "lh.top_epi_rank_L.label.gii")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' +\
          f'{os.path.join(path_surface, "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "lh.top_epi_rank_L.label.gii")} ' +\
          f'{os.path.join(path_results, "lh.top_epi_rank_L.border")}'
# Execute the command
os.system(command)

epi_rank_R = path_results + 'rh.epicenters_all_rank.func.gii'
data_epi_rank_R = loadgifti.loadGiftiVertexData(epi_rank_R)[1]

top_epi_rank_R = np.int32(1*((data_epi_rank_R == np.max(data_epi_rank_R.T))))
save_gifti(top_epi_rank_R, path_results + 'rh.top_epi_rank_R')

command = f'{path_wb_command} -metric-label-import ' +\
          f'{os.path.join(path_results, "rh.top_epi_rank_R.func.gii")} ' +\
          f'{os.path.join(path_results, "black.txt")} ' +\
          f'{os.path.join(path_results, "rh.top_epi_rank_R.label.gii")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' +\
          f'{os.path.join(path_surface, "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "rh.top_epi_rank_R.label.gii")} ' +\
          f'{os.path.join(path_results, "rh.top_epi_rank_R.border")}'
# Execute the command
os.system(command)

#------------------------------------------------------------------------------
# Create top epicenter areas - SIR
#------------------------------------------------------------------------------

epi_SIR_L = path_results + 'lh.epicenters_all_SIR.func.gii'
data_epi_SIR_L = loadgifti.loadGiftiVertexData(epi_SIR_L)[1]
top_epi_SIR_L = np.int32(1*((data_epi_SIR_L == 1)))
save_gifti(top_epi_SIR_L, path_results + 'lh.top_epi_SIR_L')

command = f'{path_wb_command} -metric-label-import ' +\
          f'{os.path.join(path_results, "lh.top_epi_SIR_L.func.gii")} ' +\
          f'{os.path.join(path_results, "blue.txt")} ' +\
          f'{os.path.join(path_results, "lh.top_epi_SIR_L.label.gii")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' +\
          f'{os.path.join(path_surface, "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "lh.top_epi_SIR_L.label.gii")} ' +\
          f'{os.path.join(path_results, "lh.top_epi_SIR_L.border")}'
# Execute the command
os.system(command)

epi_SIR_R = path_results + 'rh.epicenters_all_SIR.func.gii'
data_epi_SIR_R = loadgifti.loadGiftiVertexData(epi_SIR_R)[1]

top_epi_SIR_R = np.int32(1*((data_epi_SIR_R==2)))
save_gifti(top_epi_SIR_R, path_results + 'rh.top_epi_SIR_R')

command = f'{path_wb_command} -metric-label-import ' +\
          f'{os.path.join(path_results, "rh.top_epi_SIR_R.func.gii")} ' +\
          f'{os.path.join(path_results, "yellow.txt")} ' +\
          f'{os.path.join(path_results, "rh.top_epi_SIR_R.label.gii")}'
# Execute the command
os.system(command)

command = f'{path_wb_command} -label-to-border ' +\
          f'{os.path.join(path_surface, "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii")} ' +\
          f'{os.path.join(path_results, "rh.top_epi_SIR_R.label.gii")} ' +\
          f'{os.path.join(path_results, "rh.top_epi_SIR_R.border")}'
# Execute the command
os.system(command)

#------------------------------------------------------------------------------
# END