"""
*******************************************************************************

Purpose:

    Generate atrophy null models as input to the Matlab package
    Gene Enrichment Step

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
from scipy.io import savemat
from netneurotools.stats import gen_spinsamples

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path    = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_results = os.path.join(base_path, 'results/')

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all'
nnodes  = 400
nspins  = 30000

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

disease_profile = np.load(path_results + 'epicenter_' + subtype + '_rank.npy').reshape(400,1)
data_to_save = {'disease_profile': disease_profile}
savemat(path_results + 'map_matlab_epi_rank' + subtype + '.mat', data_to_save)

#-----------------------------------------------------------------------------
# Spin tests
#------------------------------------------------------------------------------

coords = np.genfromtxt(base_path + 'data/Schaefer_400.txt')
coords = coords[:, -3:]
nnodes = len(coords)
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1

spins = gen_spinsamples(coords,
                        hemiid,
                        n_rotate = nspins,
                        seed = 2468,
                        method = 'vasa') # check this later
spin_res = []
for i in range(nspins):
    spin_res.append(disease_profile[spins[:,i], 0])

#------------------------------------------------------------------------------
# Save as .mat to load in Matlab later on to do gene enrichment
#------------------------------------------------------------------------------

spin_res = np.array(spin_res)
data_to_save = {'spin_res' : spin_res}
savemat(path_results + 'spins_stat_matlab_epi_rank_' + subtype + '.mat', data_to_save)

#------------------------------------------------------------------------------
# END