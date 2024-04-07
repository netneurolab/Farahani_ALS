"""
*******************************************************************************

General Function of the ALS Project      

*******************************************************************************               
"""

#------------------------------------------------------------------------------
import random
import numpy as np
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation
import nibabel as nib
from sklearn.utils.validation import check_random_state
import os

random.seed(1)
#------------------------------------------------------------------------------

def pval_cal(rho_actual, null_dis, num_spins):
    """
    Calculate p-value - non parametric
    """
    p_value = (1 + np.count_nonzero(abs((null_dis - np.mean(null_dis))) > abs((rho_actual - np.mean(null_dis))))) / (num_spins + 1)
    return(p_value)

#------------------------------------------------------------------------------

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    return nib.load(atlas_path).get_fdata()

#------------------------------------------------------------------------------

def save_nifti(IN, filename, path_mask, path_results):
    """
    Save data as nifti file using template.
    """
    template = nib.load(path_mask + 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii')
    new_header = template.header.copy()
    new_data = IN
    new_img = nib.nifti1.Nifti1Image(new_data,
                                     None,
                                     header = new_header)
    nib.save(new_img, path_results + filename)

#------------------------------------------------------------------------------
def parcel2fsLR(atlas, data_parcelwise, hem):
    """
    Generate 32492 valuse for vertices from a parcelwise data
    """
    if (np.size(data_parcelwise.shape) != 1):
        results = np.zeros((32492,
                            int(np.size(data_parcelwise, axis = 1))))
    else:
        results = np.zeros((32492,))
    if hem in ['l', 'L']:
        atlas_hem = atlas[0: 32492]
    if hem in ['r', 'R']:
        atlas_hem = atlas[32492:]
        
    unique_labels_hem = np.sort(np.unique(atlas_hem))
    
    for count, x in enumerate(unique_labels_hem[1:]):
        results[atlas_hem == x,:] = data_parcelwise[count, :]
    return results

#------------------------------------------------------------------------------

def save_gifti(file, file_name):
    """
    Generate as a func.gii gile - for visualization in workbench
    """
    da = nib.gifti.GiftiDataArray(file, datatype = 'NIFTI_TYPE_FLOAT32')
    img = nib.GiftiImage(darrays = [da])
    nib.save(img, (file_name +'.func.gii')) 

#------------------------------------------------------------------------------

def show_on_surface(in_data, nnodes, rangeLow, rangehigh):
    """
    Show the data on the surface
    nnodes : the Schaefer atlas number of nodes
    rangelow and rangehigh : limits of the surface visualization
    """
    color_range = (rangeLow,  rangehigh)

    surfaces = fetch_fslr()
    lh, rh = surfaces['veryinflated']

    lh_parc, rh_parc = load_parcellation('schaefer', scale = nnodes)
    regions_lh = np.zeros_like(lh_parc)
    regions_rh = np.zeros_like(rh_parc)

    for i in range(1, nnodes + 1):
        regions_lh = np.where(np.isin(lh_parc, i),
                              in_data[i-1, 0], 0) + regions_lh
        regions_rh = np.where(np.isin(rh_parc, i),
                              in_data[i-1, 0], 0) + regions_rh

    p = Plot(lh,
             size = (nnodes, int(nnodes/2)),
             zoom = 1.2,
             layout = 'row') # left hemishphere
    p.add_layer(regions_lh, cmap = 'coolwarm', color_range = color_range)
    p.add_layer(regions_lh, cmap = 'gray', 
                as_outline = True,
                cbar = False
                )
    fig = p.build()
    fig.show()

    p = Plot(rh,
             size = (nnodes, int(nnodes/2)),
             zoom = 1.2, 
             layout = 'row') # right hemisphere

    p.add_layer(regions_rh, cmap = 'coolwarm', color_range = color_range)
    p.add_layer(regions_rh,
                cmap = 'gray',
                as_outline = True,
                cbar = False
                )
    
    fig = p.build()
    fig.show()

#------------------------------------------------------------------------------

def show_on_surface_and_save(in_data, nnodes, rangeLow, rangehigh, fig_path, fig_name):
    """
    Show the data on the surface and also save the created figures as a png file
    """

    color_range = (rangeLow,  rangehigh)

    surfaces = fetch_fslr()
    lh, rh = surfaces['veryinflated']

    lh_parc, rh_parc = load_parcellation('schaefer', scale = nnodes)
    regions_lh = np.zeros_like(lh_parc)
    regions_rh = np.zeros_like(rh_parc)

    for i in range(1, nnodes + 1):
        regions_lh = np.where(np.isin(lh_parc, i),
                              in_data[i-1, 0], 0) + regions_lh
        regions_rh = np.where(np.isin(rh_parc, i),
                              in_data[i-1, 0], 0) + regions_rh
    p = Plot(lh, size=(nnodes, int(nnodes/2)), zoom = 1.2, layout = 'row')  # left hemisphere
    p.add_layer(regions_lh, cmap = 'coolwarm', color_range = color_range)
    p.add_layer(regions_lh, cmap = 'gray', as_outline = True, cbar = False)
    fig = p.build()
    fig.show()
    fig.savefig(os.path.join(fig_path,'lh.' + fig_name), dpi = 300)  # Save left hemisphere figure

    p = Plot(rh, size=(nnodes, int(nnodes/2)), zoom = 1.2, layout = 'row')  # right hemisphere
    p.add_layer(regions_rh, cmap = 'coolwarm', color_range = color_range)
    p.add_layer(regions_rh, cmap = 'gray', as_outline = True, cbar = False)
    fig = p.build()
    fig.show()
    fig.savefig(os.path.join(fig_path, 'rh.' + fig_name), dpi = 300) # Save right hemisphere figure

#------------------------------------------------------------------------------

def match_length_degree_distribution(W, D, nbins = 10, nswap = 1000,
                                     replacement = False, weighted = True,
                                     seed=None):
    """
    Generate degree- and edge length-preserving surrogate connectomes.

    Parameters
    ----------
    W : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    D : (N, N) array-like
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping
        connections in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20
        Default = 1000.
    replacement : bool, optional
        if True all the edges are available for swapping. Default= False
    weighted : bool, optional
        Whether to return weighted rewired connectivity matrix. Default = True
    seed : float, optional
        Random seed. Default = None

    Returns
    -------
    newB : (N, N) array-like
        binary rewired matrix
    newW: (N, N) array-like
        weighted rewired matrix. Returns matrix of zeros if weighted=False.
    nr : int
        number of successful rewires

    Notes
    -----
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship

    """
    rs = check_random_state(seed)
    N = len(W)
    # divide the distances by lengths
    bins = np.linspace(D[D.nonzero()].min(), D[D.nonzero()].max(), nbins + 1)
    bins[-1] += 1
    L = np.zeros((N, N))
    for n in range(nbins):
        i, j = np.where(np.logical_and(bins[n] <= D, D < bins[n + 1]))
        L[i, j] = n + 1

    # binarized connectivity
    B = (W != 0).astype(np.int_)

    # existing edges (only upper triangular cause it's symmetric)
    cn_x, cn_y = np.where(np.triu((B != 0) * B, k=1))

    tries = 0
    nr = 0
    newB = np.copy(B)

    while ((len(cn_x) >= 2) & (nr < nswap)):
        # choose randomly the edge to be rewired
        r = rs.randint(len(cn_x))
        n_x, n_y = cn_x[r], cn_y[r]
        tries += 1

        # options to rewire with
        # connected nodes that doesn't involve (n_x, n_y)
        index = (cn_x != n_x) & (cn_y != n_y) & (cn_y != n_x) & (cn_x != n_y)
        if len(np.where(index)[0]) == 0:
            cn_x = np.delete(cn_x, r)
            cn_y = np.delete(cn_y, r)

        else:
            ops1_x, ops1_y = cn_x[index], cn_y[index]
            # options that will preserve the distances
            # (ops1_x, ops1_y) such that
            # L(n_x,n_y) = L(n_x, ops1_x) & L(ops1_x,ops1_y) = L(n_y, ops1_y)
            index = (L[n_x, n_y] == L[n_x, ops1_x]) & (
                L[ops1_x, ops1_y] == L[n_y, ops1_y])
            if len(np.where(index)[0]) == 0:
                cn_x = np.delete(cn_x, r)
                cn_y = np.delete(cn_y, r)

            else:
                ops2_x, ops2_y = ops1_x[index], ops1_y[index]
                # options of edges that didn't exist before
                index = [(newB[min(n_x, ops2_x[i])][max(n_x, ops2_x[i])] == 0) & (newB[min(n_y, ops2_y[i])][max(n_y, ops2_y[i])] == 0)
                         for i in range(len(ops2_x))]
                if (len(np.where(index)[0]) == 0):
                    cn_x = np.delete(cn_x, r)
                    cn_y = np.delete(cn_y, r)

                else:
                    ops3_x, ops3_y = ops2_x[index], ops2_y[index]

                    # choose randomly one edge from the final options
                    r1 = rs.randint(len(ops3_x))
                    nn_x, nn_y = ops3_x[r1], ops3_y[r1]

                    # Disconnect the existing edges
                    newB[n_x, n_y] = 0
                    newB[nn_x, nn_y] = 0
                    # Connect the new edges
                    newB[min(n_x, nn_x), max(n_x, nn_x)] = 1
                    newB[min(n_y, nn_y), max(n_y, nn_y)] = 1
                    # one successfull rewire!
                    nr += 1

                    # rewire with replacement
                    if replacement:
                        cn_x[r], cn_y[r] = min(n_x, nn_x), max(n_x, nn_x)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x[index], cn_y[index] = min(n_y, nn_y), max(n_y, nn_y)
                    # rewire without replacement
                    else:
                        cn_x = np.delete(cn_x, r)
                        cn_y = np.delete(cn_y, r)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)

    if nr < nswap:
        print(f"I didn't finish, out of rewirable edges: {len(cn_x)}")

    i, j = np.triu_indices(N, k=1)
    # Make the connectivity matrix symmetric
    newB[j, i] = newB[i, j]

    # check the number of edges is preserved
    if len(np.where(B != 0)[0]) != len(np.where(newB != 0)[0]):
        print(
            f"ERROR --- number of edges changed, \
            B:{len(np.where(B!=0)[0])}, newB:{len(np.where(newB!=0)[0])}")
    # check that the degree of the nodes it's the same
    for i in range(N):
        if np.sum(B[i]) != np.sum(newB[i]):
            print(
                f"ERROR --- node {i} changed k by: \
                {np.sum(B[i]) - np.sum(newB[i])}")

    newW = np.zeros((N, N))
    if weighted:
        # Reassign the weights
        mask = np.triu(B != 0, k=1)
        inids = D[mask]
        iniws = W[mask]
        inids_index = np.argsort(inids)
        # Weights from the shortest to largest edges
        iniws = iniws[inids_index]
        mask = np.triu(newB != 0, k=1)
        finds = D[mask]
        i, j = np.where(mask)
        # Sort the new edges from the shortest to the largest
        finds_index = np.argsort(finds)
        i_sort = i[finds_index]
        j_sort = j[finds_index]
        # Assign the initial sorted weights
        newW[i_sort, j_sort] = iniws
        # Make it symmetrical
        newW[j_sort, i_sort] = iniws

    return newB, newW, nr

#------------------------------------------------------------------------------

def randmio_und(W, itr):
    """
    Optimized version of randmio_und.

    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    This function is significantly faster if numba is enabled, because
    the main overhead is `np.random.randint`, see `here <https://stackoverflow.com/questions/58124646/why-in-python-is-random-randint-so-much-slower-than-random-random>`_

    Parameters
    ----------
    W : (N, N) array-like
        Undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    W : (N, N) array-like
        Randomized network
    eff : int
        number of actual rewirings carried out
    """  # noqa: E501
    W = W.copy()
    n = len(W)
    i, j = np.where(np.triu(W > 0, 1))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for _ in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k), np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a, b = i[e1], j[e1]
                c, d = i[e2], j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # flip edge c-d with 50% probability
            # to explore all potential rewirings
            if np.random.random() > .5:
                i[e2], j[e2] = d, c
                c, d = d, c

            if not (W[a, d] or W[c, b]):
                W[a, d] = W[a, b]
                W[a, b] = 0
                W[d, a] = W[b, a]
                W[b, a] = 0
                W[c, b] = W[c, d]
                W[c, d] = 0
                W[b, c] = W[d, c]
                W[d, c] = 0

                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return W, eff

#------------------------------------------------------------------------------

def load_HCP_names():
    """
    This containes the ordered name of contrast in the HCP GA map (86 maps)
    """
    name_contrasts = [
                    'tfMRI_WM_2BK_BODY',
                    'tfMRI_WM_2BK_FACE',
                    'tfMRI_WM_2BK_PLACE',
                    'tfMRI_WM_2BK_TOOL',
                    'tfMRI_WM_0BK_BODY',
                    'tfMRI_WM_0BK_FACE',
                    'tfMRI_WM_0BK_PLACE',
                    'tfMRI_WM_0BK_TOOL',
                    'tfMRI_WM_2BK',
                    'tfMRI_WM_0BK',
                    'tfMRI_WM_2BK-0BK',
                    'tfMRI_WM_neg_2BK',
                    'tfMRI_WM_neg_0BK',
                    'tfMRI_WM_0BK-2BK',
                    'tfMRI_WM_BODY',
                    'tfMRI_WM_FACE',
                    'tfMRI_WM_PLACE',
                    'tfMRI_WM_TOOL',
                    'tfMRI_WM_BODY-AVG',
                    'tfMRI_WM_FACE-AVG',
                    'tfMRI_WM_PLACE-AVG',
                    'tfMRI_WM_TOOL-AVG',
                    'tfMRI_WM_neg_BODY',
                    'tfMRI_WM_neg_FACE',
                    'tfMRI_WM_neg_PLACE',
                    'tfMRI_WM_neg_TOOL',
                    'tfMRI_WM_AVG-BODY',
                    'tfMRI_WM_AVG-FACE',
                    'tfMRI_WM_AVG-PLACE',
                    'tfMRI_WM_AVG-TOOL',
                    'tfMRI_GAMBLING_PUNISH',
                    'tfMRI_GAMBLING_REWARD',
                    'tfMRI_GAMBLING_PUNISH-REWARD',
                    'tfMRI_GAMBLING_neg_PUNISH',
                    'tfMRI_GAMBLING_neg_REWARD',
                    'tfMRI_GAMBLING_REWARD-PUNISH',
                    'tfMRI_MOTOR_CUE',
                    'tfMRI_MOTOR_LF',
                    'tfMRI_MOTOR_LH',
                    'tfMRI_MOTOR_RF',
                    'tfMRI_MOTOR_RH',
                    'tfMRI_MOTOR_T',
                    'tfMRI_MOTOR_AVG',
                    'tfMRI_MOTOR_CUE-AVG',
                    'tfMRI_MOTOR_LF-AVG',
                    'tfMRI_MOTOR_LH-AVG',
                    'tfMRI_MOTOR_RF-AVG',
                    'tfMRI_MOTOR_RH-AVG',
                    'tfMRI_MOTOR_T-AVG',
                    'tfMRI_MOTOR_neg_CUE',
                    'tfMRI_MOTOR_neg_LF',
                    'tfMRI_MOTOR_neg_LH',
                    'tfMRI_MOTOR_neg_RF',
                    'tfMRI_MOTOR_neg_RH',
                    'tfMRI_MOTOR_neg_T',
                    'tfMRI_MOTOR_neg_AVG',
                    'tfMRI_MOTOR_AVG-CUE',
                    'tfMRI_MOTOR_AVG-LF',
                    'tfMRI_MOTOR_AVG-LH',
                    'tfMRI_MOTOR_AVG-RF',
                    'tfMRI_MOTOR_AVG-RH',
                    'tfMRI_MOTOR_AVG-T',
                    'tfMRI_LANGUAGE_MATH',
                    'tfMRI_LANGUAGE_STORY',
                    'tfMRI_LANGUAGE_MATH-STORY',
                    'tfMRI_LANGUAGE_STORY-MATH',
                    'tfMRI_LANGUAGE_neg_MATH',
                    'tfMRI_LANGUAGE_neg_STORY',
                    'tfMRI_SOCIAL_RANDOM',
                    'tfMRI_SOCIAL_TOM',
                    'tfMRI_SOCIAL_RANDOM-TOM',
                    'tfMRI_SOCIAL_neg_RANDOM',
                    'tfMRI_SOCIAL_neg_TOM',
                    'tfMRI_SOCIAL_TOM-RANDOM',
                    'tfMRI_RELATIONAL_MATCH',
                    'tfMRI_RELATIONAL_REL',
                    'tfMRI_RELATIONAL_MATCH-REL',
                    'tfMRI_RELATIONAL_REL-MATCH',
                    'tfMRI_RELATIONAL_neg_MATCH',
                    'tfMRI_RELATIONAL_neg_REL',
                    'tfMRI_EMOTION_FACES',
                    'tfMRI_EMOTION_SHAPES',
                    'tfMRI_EMOTION_FACES-SHAPES',
                    'tfMRI_EMOTION_neg_FACES',
                    'tfMRI_EMOTION_neg_SHAPES',
                    'tfMRI_EMOTION_SHAPES-FACES']

    return(name_contrasts)

#------------------------------------------------------------------------------
# END