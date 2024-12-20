"""
*******************************************************************************

Function to define paths and constants for the whole project!

The user should set the paths in this script to be able to run the project.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Needed libraries
#------------------------------------------------------------------------------

import os

#------------------------------------------------------------------------------
# Set Paths - Modify according to your environment
#------------------------------------------------------------------------------

# Base directory where subfolders 'data', 'results', 'figures', and 'gene_enrichment' are located.
base_path         = '/Users/asaborzabadifarahani/Desktop/ALS/New_code/'

# Path to group-averaged surface files from the HCP-YA dataset.
path_surface      = '/Users/asaborzabadifarahani/Desktop/GA/HumanCorticalParcellations_wN_6V6gD/'

# Path to the wb_command binary, provided by the HCP team.
path_wb_command   = '/Users/asaborzabadifarahani/Downloads/workbench/bin_macosxub/wb_command'

# Path to functional group-average maps from HCP-YA.
path_HCP_GA       = '/Users/asaborzabadifarahani/Desktop/GA/HCP_S1200_GroupAvg_v1/'

# Path of folder where results, generated by the scripts, will be saved.
path_results      = os.path.join(base_path, 'results/')

# Path to folder where figures generated by scripts will be saved.
path_fig          = os.path.join(base_path, 'figures/')

# Paths to various data directories required by the scripts.
path_data         = os.path.join(base_path, 'data/mri_data/')
path_demographic  = os.path.join(base_path, 'data/demographic_info/')
path_networks     = os.path.join(base_path, 'data/networks/')
path_sc           = os.path.join(base_path, 'data/sc_networks/')
path_atlas        = os.path.join(base_path, 'data/parcellations/')
path_mask         = os.path.join(base_path, 'data/mni_icbm152_nlin_sym_09c/')
path_templates    = os.path.join(base_path, 'data/templates/')
path_medialwall   = os.path.join(base_path, 'data/medialwall/')
path_gene         = os.path.join(base_path, 'data/gene_data/')

# Path to the results of gene-enrichment analysis.
path_gene_results = os.path.join(base_path, 'gene_enrichment/Results/')

#------------------------------------------------------------------------------
# Define constants
#------------------------------------------------------------------------------

# Number of parcels in the Schaefer-400 parcellation.
nnodes = 400

# Number of cortical vertices in a cifti file, excluding the medial wall.
num_cort_vertices_noMW = 59412

# Number of cortical vertices in two gifti files, including the medial wall.
num_cort_vertices_withMW = 64984

# Number of vertices in a single hemisphere (gifti file), including the medial wall.
num_vertices_gifti = 32492

#------------------------------------------------------------------------------
# END