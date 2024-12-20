"""
*******************************************************************************

Script purpose:

    Data needed to perform gene expression analysis
    All genes in Allen Human Brain
    This data is used later on in the gene enrichment analysis

Script output:
    The name of the genes are saved as:
    'names_genes_filtered.mat'

    Gene maps are saved as:
    'gene_coexpression_filtered.mat'

Note:
    These files are inputs to the ABAnnotate software for gene enrichment analysis.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import abagen
import warnings
import numpy as np
import pandas as pd
from scipy.io import savemat
from nilearn.datasets import fetch_atlas_schaefer_2018
from globals import path_results, nnodes

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

no_filter = 0
parc      = 'Schaefer400'

#------------------------------------------------------------------------------
# Start using abagen to get gene information
#------------------------------------------------------------------------------

parc_file_mni = fetch_atlas_schaefer_2018(n_rois = nnodes)['maps']
cortex = np.arange(nnodes)

if no_filter == 1:
    expression = abagen.get_expression_data(parc_file_mni,
                                            lr_mirror     = 'bidirectional',
                                            missing       = 'interpolate',
                                            return_donors = False)
    expression.to_csv(path_results + 'gene_coexpression_nofilter')
    data_to_save = {'expression': expression}
    savemat(path_results + 'gene_coexpression_nofilter.mat', data_to_save)
    columns_name= np.array(expression.columns)
    data_to_save = {'names': columns_name}
    savemat(path_results + 'names_genes_nofilter.mat', data_to_save)
else:
    expression = abagen.get_expression_data(parc_file_mni,
                                            lr_mirror = 'bidirectional',
                                            missing = 'interpolate',
                                            return_donors = True)

    expression_st, ds = abagen.correct.keep_stable_genes(list(expression.values()),
                                                      threshold        = 0.1,
                                                      percentile       = False,
                                                      return_stability = True)

    expression_st = pd.concat(expression_st).groupby('label').mean()

    columns_name = np.array(expression_st.columns)
    data_to_save = {'names': columns_name}
    savemat(path_results + 'names_genes_filtered.mat', data_to_save)

    data_to_save = {'gene_coexpression': np.array(expression_st)}
    savemat(path_results + 'gene_coexpression_filtered.mat', data_to_save)

#------------------------------------------------------------------------------
# END