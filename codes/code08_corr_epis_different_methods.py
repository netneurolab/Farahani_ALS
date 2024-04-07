"""
*******************************************************************************

Purpose:
    calculate the correlation value of epicenters obtained with different methods

Results: n = 1,000 - method: vasa


    Spinal:
        PearsonRResult(statistic=0.8042495649392122, pvalue=5.592599784995778e-92)
        p value is - spin test:0.000999000999000999

    Bulbar:
        PearsonRResult(statistic=0.8001017112691977, pvalue=2.307227588723794e-90)
        p value is - spin test:0.000999000999000999

    All:
        PearsonRResult(statistic=0.8026198094086654, pvalue=2.4375368329046723e-91)
        p value is - spin test:0.000999000999000999

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import warnings
import scipy.stats as stats
from IPython import get_ipython
from scipy.stats import pearsonr
from netneurotools.stats import gen_spinsamples
from functions import pval_cal

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path    = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_results = os.path.join(base_path,'results/')

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all'   #'all' #'spinal' #'bulbar'
nnodes  = 400     # as we are using schaefer-400 parcellation
nspins  = 1000    # number of nulls

#------------------------------------------------------------------------------
# Load epicenter data (both ranking and SIR modeling results)
#------------------------------------------------------------------------------

epi_rank = np.load(path_results + 'epicenter_' + subtype + '_rank.npy').reshape(nnodes,1)
epi_sir  = 400 - np.load(path_results + 'epicenters_' + subtype + '_SIR.npy').reshape(nnodes,1)

print(stats.pearsonr(epi_rank.flatten(), epi_sir.flatten()))

#------------------------------------------------------------------------------
# Spin test to evaluate the significance of the correlation
#------------------------------------------------------------------------------

def corr_spin(x, y, spins, nspins):
    """
    Spin test - account for spatial autocorrelation
    """
    rho, _ = pearsonr(x.flatten(), y.flatten())
    null = np.zeros((nspins,)) # null correlation
    for i in range(nspins):
         null[i], _ = pearsonr((x[spins[:,i]]).flatten(), y.flatten())
    return rho, null

coords = np.genfromtxt(base_path + 'data/schaefer_' + str(nnodes) + '.txt')
coords = coords[:, -3:]
nnodes = len(coords)
hemiid = np.zeros((nnodes,))
hemiid[:int(nnodes/2)] = 1
spins = gen_spinsamples(coords,
                        hemiid,
                        n_rotate = nspins,
                        seed = 1234,
                        method = 'vasa')

nn_corrs, generated_null = corr_spin(epi_rank,
                                     epi_sir,
                                     spins,
                                     nspins)
res = pval_cal(nn_corrs, generated_null, nspins)

# print the obtained p-spin
print('p value is - spin test:' +  str(res))

#------------------------------------------------------------------------------
# END