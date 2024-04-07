# Network Spreading and Local Biological Vulnerability in Amyotrophic Lateral Sclerosis
Authors: Asa Farahani, Justine Y. Hansen, Vincent Bazinet, Golia Shafiei, Louis D. Collins, Mahsa Dadar, Sanjay Kalra, Alain Dagher, Bratislav Misic.

This repository contains the code used to generate the results in the study "Network Spreading and Local Biological Vulnerability in Amyotrophic Lateral Sclerosis". The study investigates the intricate relationship between network spreading mechanisms and local biological vulnerability for shaping the atrophy in Amyotrophic Lateral Sclerosis (ALS).

## Data Confidentiality Notice
The ALS dataset used in this study is provided by the Canadian ALS Neuroimaging Consortium (CALSNIC), and contains confidential information and cannot be publicly released. For more details and to request access to the dataset, please visit the [CALSNIC](https://calsnic.org/) website.

## Repository Structure
### Codes
This section outlines the functionality of each script within the repository:

- `code01_demographic_cleaning.py` - Cleans the data directory (etc. by removing subjects with incomplete structural data).
- `code02_voxelwise_w_score.py` - Builds the model to calculate voxel-wise w-score maps.
- `code03_parcellate_w_score.py` - Utilizes 400 nodes Schaefer cortical parcellation and the Johns Hopkins University (JHU) atlas of white matter tracts to generate parcellated atrophy maps.
- `code04_voneconomo_atrophy.py` - Assesses atrophy values within each cytoarchitectonic class defined by the von-Economo parcellation.
- `code05_sc_sulls.py` - Computes node-neighbor correlation values to assess the role of structural connectome in shaping the ALS-related atrophy.
- `code06_epicenter_ranking.py` - Identifies cortical epicenter likelihood maps through node-neighbor assessment.
- `code07_epicenter_model.py` - Identifies cortical epicenter likelihood maps using the SIR model.
- `code08_corr_epis_different_methods.py` - Compares epicenter maps derived from different methodologies.
- `code09_many_networks.py` - Assesses node-neighbor correlations using various biologically meaningful brain networks.
- `code10_gene_enrichment_null_generation.py` - Creates a null model to be used later by [ABAnnotate](https://github.com/LeonDLotter/ABAnnotate) (MATLAB) when performing the gene enrichment analysis.
- `code11_gene_results_cells.py & code11_gene_results_GO.py` - Visualize the results obtained from gene enrichment (results of ABAnnotate).
- `code12_hcp_group_maps.py` - Parcellates the [HCP](https://www.humanconnectome.org/study/hcp-young-adult/article/s1200-group-average-data-release) group average task-activation maps (400 nodes Schaefer parcellation).
- `code13_pls.py` - Applies a Behavioral Partial Least Squares ([PLS](https://github.com/netneurolab/pypyls)) model to correlate behavioral data with cortical epicenter likelihood maps in the ALS cohort.
- `code14_subtype_differences_brain.py & code15_subtype_differences_behavior.py` - Investigates differences between bulbar and spinal onset ALS patients in terms of behavior and epicenter likelihood maps.

#### Utility Scripts

- `create_some_border_files.py` - Generates border files necessary for figure creation when using the [wb_view](https://www.humanconnectome.org/software/connectome-workbench) software.
- `download_gene_info_abagen.py` - Retrieves gene data using the [abagen](https://github.com/rmarkello/abagen) software.
- `functions.py` - Contains functions utilized across various scripts in the project.
- `pls_func.py` - Contains functions used to illustrate the results of the PLS analysis.
- `simulated_atrophy.py` - Includes functions to run the SIR model. The code for this section is developed by [Vincent Bazinet](https://github.com/VinceBaz).

### Data
- The structural connectome, and other biologically defined connectomes (including metabolic similarity, hemodynamic similarity, transcriptomic similarity, receptor similarity and laminar similarity) are included in 'SC' and 'Network' folders, respectively. The biological connectomes are developed by [Justine Hansen](https://github.com/netneurolab/hansen_many_networks).
- The 400 nodes Schaefer parcellation and JHU atlas at [MNI152-2009c](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009) space are also provided in 'TransformedAtlases' folder.
- The von-Economo cytoarchitectonic classes in 400 nodes Shaefer parcellation are included in the 'economo_Schaefer400.mat' file.

## Contact Information
For questions, please email: [asa.borzabadifarahani@mail.mcgill.ca](mailto:asa.borzabadifarahani@mail.mcgill.ca).
