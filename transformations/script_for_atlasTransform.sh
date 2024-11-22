#!/bin/bash

#############################################################################
In this script the paths are not included - adjust the paths if need to rerun
#############################################################################


# Define the paths to your source label file and target template image

target_template= "mni_icbm152_t1_tal_nlin_sym_09c.nii" # the space I want to have data in
source_label_space= "MNI152NLin6_res-1x1x1_T1w.nii.gz" # the current space of the data (unwanted)

# Perform Registration and find the transformation matrix
antsRegistrationSyN.sh -d 3 -f "$target_template" -m "$source_label_space" -o output_transform

# Apply Transformation with Nearest-Neighbor Interpolation - Schaefer 400
label_in_source="Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz" # in source_label_space space

antsApplyTransforms -d 3 -i "$label_in_source" -r "$target_template" -o Schaefer400.nii.gz -t output_transform1Warp.nii.gz -t output_transform0GenericAffine.mat --interpolation NearestNeighbor

# Apply Transformation with Nearest-Neighbor Interpolation - Schaefer 800
label_in_source="Schaefer2018_800Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"

antsApplyTransforms -d 3 -i "$label_in_source" -r "$target_template" -o Schaefer800.nii.gz -t output_transform1Warp.nii.gz -t output_transform0GenericAffine.mat --interpolation NearestNeighbor

# Apply Transformation with Nearest-Neighbor Interpolation - JHU atlas
label_in_source="JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz"
antsApplyTransforms -d 3 -i "$label_in_source" -r "$target_template" -o JHU_thr25.nii.gz -t output_transform1Warp.nii.gz -t output_transform0GenericAffine.mat --interpolation NearestNeighbor


