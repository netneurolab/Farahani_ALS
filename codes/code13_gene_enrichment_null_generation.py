"""
*******************************************************************************

Script purpose:

    Generate atrophy null models as input to the Matlab package.
    Needed for the gene enrichment step

Script output:

    Save the epicenter map as a .mat file:
        'map_matlab_epi_rank' + subtype + '.mat'

    Save the epicenter null map as a .mat file:
        'spins_stat_matlab_epi_rank_' + subtype + '.mat'

Note:

    The generated files are needed for ABAnnotate (related to Fig. 4).

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
from scipy.io import savemat
from netneurotools.stats import gen_spinsamples
from globals import path_results, path_atlas, nnodes

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all' # Options: 'all', 'spinal', 'bulbar'
nspins  = 30000

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

disease_profile = np.load(path_results + 'epicenter_' + subtype + '_rank.npy').reshape(nnodes, 1)
data_to_save = {'disease_profile': disease_profile}
savemat(path_results + 'map_matlab_epi_rank' + subtype + '.mat', data_to_save)

#-----------------------------------------------------------------------------
# Spin tests
#------------------------------------------------------------------------------

coords = np.genfromtxt(path_atlas + 'Schaefer_' + str(nnodes) + '.txt')
coords = coords[:, -3:]
nnodes = len(coords)
hemiid = np.zeros((nnodes,))
hemiid[:int(nnodes/2)] = 1

spins = gen_spinsamples(coords,
                        hemiid,
                        n_rotate = nspins,
                        seed = 2468,
                        method = 'vasa')
spin_res = []
for i in range(nspins):
    spin_res.append(disease_profile[spins[:, i], 0])

#------------------------------------------------------------------------------
# Save as .mat to load in Matlab later on to do gene enrichment
#------------------------------------------------------------------------------

spin_res = np.array(spin_res)
data_to_save = {'spin_res' : spin_res}
savemat(path_results + 'spins_stat_matlab_epi_rank_' + subtype + '.mat', data_to_save)

#------------------------------------------------------------------------------
# END