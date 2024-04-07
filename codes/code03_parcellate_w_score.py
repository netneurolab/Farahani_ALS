"""
*******************************************************************************

Purpose:
    Create parcel-wise results out of voxelwise results using Schaefer parcellation
    for cortical regions and the JHU atlas for white matter regions.

Output:
    ------------------------------- python file -------------------------------

    'mean_w_score_' + str(subtype) + '_Schaefer.npy'
    'mean_w_score_' + str(subtype) + '_WM_tracts.npy'

    ------------------  parcellated schaefer mean of atrophy ------------------

    lh.mean_wscore_' + str(subtype) + '_Schaefer'
    rh.mean_wscore_' + str(subtype) + '_Schaefer'

    -----------------------  JHU labels mean of atrophy -----------------------

    'mean_w_score_' + str(subtype) + '_WM_tracts.nii.gz

*******************************************************************************
NOTE:
    in the saved files w-scores have already been multiplied by -1

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython import get_ipython
from neuromaps.images import load_data
from nilearn.plotting import plot_glass_brain
from neuromaps.images import dlabel_to_gifti
from netneurotools import datasets as nntdata
from functions import (show_on_surface_and_save, 
                       parcel2fsLR,
                       save_gifti,
                       load_nifti,
                       save_nifti)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path       = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_mask       = os.path.join(base_path,'data/MNITemplates/mni_icbm152_nlin_sym_09c/')
path_atlas      = os.path.join(base_path,'data/TransformedAtlases/')
path_results    = os.path.join(base_path,'results/')
script_name     = os.path.basename(__file__).split('.')[0]
path_fig        = os.path.join(os.getcwd(), 'generated_figures/', script_name)
os.makedirs(path_fig, exist_ok = True)

#------------------------------------------------------------------------------
# Constants 
#------------------------------------------------------------------------------

subtype         = 'all' #'all' #'spinal' #'bulbar'
nnodes          =  400     # Run the code with nnodes = 400

#------------------------------------------------------------------------------
# Needed Functions
#------------------------------------------------------------------------------

def calculate_mean_w_score(w_score, atlas, n_parcels):
    """
    Calculate mean w-score across parcels.
    """
    mean_w_score = np.zeros((int(n_parcels), 1))
    for i in range(1, int(n_parcels) + 1):
        mean_w_score[i - 1, :] = np.mean(w_score[atlas == i])
    return mean_w_score

#------------------------------------------------------------------------------
# Load atlases ( Schaefer-400 and White-matter tractography - JHU )

schafer_atlas = load_nifti(os.path.join(path_atlas, f'schaefer{nnodes}.nii.gz'))
WM_atlas      = load_nifti(os.path.join(path_atlas, 'JHU_thr25.nii.gz'))

# -1 is done to exclude the 0 label
nWM = np.unique(WM_atlas).shape[0] - 1

#------------------------------------------------------------------------------
# Load group-averaged w-score map and multiply it by -1

w_score_mean = -1 * load_nifti(path_results + 'mean_w_score_' + subtype + '.nii.gz')

#------------------------------------------------------------------------------
# Calculate mean w-scores for Schaefer and JHU parcels - parcellation step

w_score_schaefer  = calculate_mean_w_score(w_score_mean, schafer_atlas, nnodes)
w_score_wm_tracts = calculate_mean_w_score(w_score_mean, WM_atlas, nWM)

#------------------------------------------------------------------------------
# Save the parcellated results to .npy format

''' NOTE: group-averaged maps are parcellated'''

np.save(path_results + 'mean_w_score_' + subtype + '_Schaefer.npy',
        w_score_schaefer)
np.save(path_results + 'mean_w_score_' + subtype + '_WM_tracts.npy',
        w_score_wm_tracts)

#------------------------------------------------------------------------------
# Visualize and save cortical mean w-score maps

schaefer = nntdata.fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

save_gifti(parcel2fsLR(atlas,
                       w_score_schaefer[:int(nnodes/2)].reshape(int(nnodes/2), 1),
                       'L'),
           path_results + 'lh.mean_wscore_' + subtype + '_Schaefer')
save_gifti(parcel2fsLR(atlas,
                       w_score_schaefer[int(nnodes/2):].reshape(int(nnodes/2), 1),
                       'R'),
           path_results + 'rh.mean_wscore_' + subtype + '_Schaefer')

show_on_surface_and_save(w_score_schaefer.reshape(nnodes,1), nnodes, -0.3, 0.3,
                         path_fig, 'mean_wscore_' + subtype + '_Schaefer.png')

#------------------------------------------------------------------------------
# Labels of the white matter tracks defined by the atlas

name_columns = ['Anterior thalamic radiation L',
                'Anterior thalamic radiation R',
                'Corticospinal tract L',
                'Corticospinal tract R',
                'Cingulum (cingulate gyrus) L',
                'Cingulum (cingulate gyrus) R',
                'Cingulum (hippocampus) L',
                'Cingulum (hippocampus) R',
                'Forceps major',
                'Forceps minor',
                'Inferior fronto-occipital fasciculus L',
                'Inferior fronto-occipital fasciculus R',
                'Inferior longitudinal fasciculus L',
                'Inferior longitudinal fasciculus R',
                'Superior longitudinal fasciculus L',
                'Superior longitudinal fasciculus R',
                'Uncinate fasciculus L',
                'Uncinate fasciculus R',
                'Superior longitudinal fasciculus (temporal part) L',
                'Superior longitudinal fasciculus (temporal part) R']

# Sorting white matter data to create a barplot
w_score_wm_tracts = np.reshape(w_score_wm_tracts, nWM)
sorted_indices = np.argsort(w_score_wm_tracts)
sorted_values = w_score_wm_tracts[sorted_indices]
sorted_categories = np.array(name_columns)[sorted_indices]

fig, ax = plt.subplots(figsize = (20, 10))
plt.rcParams['font.family'] = 'Arial'

# Defined colors
positive_color  = [1.00, 0.73, 0.62]
negative_color  = [0.74, 0.85, 1.00]
gray_color      = [0.19, 0.19, 0.19]

# Assign colors based on the sign of the value
colors = [positive_color if val > 0 else negative_color for val in sorted_values]
alphas = [0.8 for val in sorted_values]

for category, value, color, alpha in zip(sorted_categories, sorted_values, colors, alphas):
    ax.bar(category, value, color = color, alpha = alpha)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(gray_color)
ax.spines['left'].set_color(gray_color)

for label in ax.get_xticklabels():
    label.set_color(gray_color)

for label in ax.get_yticklabels():
    label.set_color(gray_color)

ax.xaxis.label.set_color(gray_color)
ax.yaxis.label.set_color(gray_color)

plt.xticks(rotation = 90)
plt.ylim(-0.1, 0.4)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(12)

plt.tight_layout()
plt.savefig(os.path.join(path_fig, 'barplot_wscore_' + subtype + '_WMtracts'), dpi = 300)
plt.show()

#------------------------------------------------------------------------------

for n in range(1, nWM + 1):
    WM_atlas[WM_atlas == n] = w_score_wm_tracts[n - 1]

save_nifti(WM_atlas,
          'w_score_' + subtype + '_WM_tracts.nii.gz',
           path_mask,
           path_results)

template = nib.load(path_results + 'w_score_' +  subtype + '_WM_tracts.nii.gz')

plt.figure()
# Whole brain sagittal cuts and map is thresholded at 3
plot_glass_brain(template,
                 colorbar = False,
                 plot_abs = False,
                 vmax = 0.3,
                 vmin = -0.3, 
                 alpha = 0.4,
                 symmetric_cbar = True,
                 cmap = "coolwarm")

plt.savefig(os.path.join(path_fig, 'w_score_' + subtype + '_WM_tracts.png'), dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END