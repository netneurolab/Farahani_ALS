"""
*******************************************************************************

Script purpose:

    Calculate the correlation value of epicenters obtained with different methods

Script output:

    -----------------------------------------------------------------------

    n = 1,000 - method: vasa

    -----------------------------------------------------------------------

    All:
        PearsonRResult(statistic=0.7673206435104709, pvalue=8.252042786870684e-79)
        p value is - spin test:0.000999000999000999

    Spinal:
        PearsonRResult(statistic=0.7637509204962694, pvalue=1.147281854851163e-77)
        p value is - spin test:0.000999000999000999

    Bulbar:
        PearsonRResult(statistic=0.7762654769510311, pvalue=9.118784282522179e-82)
        p value is - spin test:0.000999000999000999

    -----------------------------------------------------------------------

NOTE:

    The results coming from this script used in Fig. 2b,c and also in Fig. 6a.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr
from functions import pval_cal
from functions import vasa_null_Schaefer
from globals import path_results, nnodes

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

subtype = 'all'   # Options: 'all', 'spinal', 'bulbar'
nspins  = 1000    # Number of spins

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
    null = np.zeros((nspins,))
    for i in range(nspins):
         null[i], _ = pearsonr((x[spins[:,i]]).flatten(), y.flatten())
    return rho, null

spins = vasa_null_Schaefer(nspins)

nn_corrs, generated_null = corr_spin(epi_rank,
                                     epi_sir,
                                     spins,
                                     nspins)
res = pval_cal(nn_corrs, generated_null, nspins)

# Print the obtained p-spin
print('p value is - spin test:' +  str(res))

#------------------------------------------------------------------------------
# END