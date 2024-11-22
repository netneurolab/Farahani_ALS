"""
*******************************************************************************

Script purpose:

    Compare behavioral measures across two ALS subtypes, namely spinal and bulbar-ALS

Script output:

    Significant columns: ['ALSFRS_1_Speech',
                          'ALSFRS_2_Salivation',
                          'ALSFRS_3_Swallowing',
                          'ALSFRS_4_Handwriting',
                          'ALSFRS_8_Walking',
                          'ALSFRS_9_Climbingstairs',

                          'Reflexes_Jaw',
                          'Reflexes_RightArm',
                          'Reflexes_LeftArm',

                          'Symptom_Duration']

    P-values: [4.213732858770071e-09,
               4.907393979427152e-05,
               4.282010697553192e-06,
               0.012375765586002289,
               0.02839695151936666,
               0.020184871471165846,

               0.0024630381869709074,
               1.0903416562789722e-05,
               0.020184871471165846,

               0.023104266546472414]

Note:

    Results are presented in Fig.6b.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from globals import path_fig, path_results
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

# Load data
b_df =  pd.read_csv(path_results + 'df_ALS_bulbar.csv')
s_df =  pd.read_csv(path_results + 'df_ALS_spinal.csv')

df = pd.concat([b_df, s_df], ignore_index=True)

#------------------------------------------------------------------------------
# Some cleaning
#------------------------------------------------------------------------------

handedness_mapping = {'right': 0,
                      'left': 1,
                      'ambidextrous': 2}

df['Handedness'] = df['Handedness'].map(handedness_mapping)
med = {'yes': 1,
       'no': 0}
df['MedicalExamination_Riluzole'] = df['MedicalExamination_Riluzole'].map(med)

df = pd.get_dummies(df,
                    columns=['Side_1st_MotorSymptomOnset'],
                    prefix=['side'])
columns_with_missing = df.columns[df.isnull().any()]
numerical_columns_with_missing = columns_with_missing.intersection(df.select_dtypes(include=['number']).columns)

missing_counts = {}
for column in numerical_columns_with_missing:
    missing_subjects = df[df[column].isnull()]['Filename'].unique()
    missing_counts[column] = len(missing_subjects)

numerical_columns_without_missing = df.select_dtypes(include = ['number']).columns.difference(numerical_columns_with_missing)
for column in numerical_columns_without_missing:
    missing_counts[column] = 0

missing_counts_df = pd.DataFrame.from_dict(missing_counts,
                                           orient = 'index',
                                           columns = ['Missing Subjects'])
conditions = ((missing_counts_df['Missing Subjects'] <= 21) &
    (~missing_counts_df.index.str.contains("date|Date|index|Unnamed|Visit_details|DBM|Site|Status|TimePoint")))
missing_counts_df = missing_counts_df[conditions]

column_groups = {
    "TAP"           : ["TAP"],
    "ECAS"          : ["ECAS"],
    "ALSFRS"        : ["ALSFRS"],
    "Tone"          : ["Tone"],
    "Reflexes"      : ["Reflexes"],
    "BIO"           : ["BIO"]}

# Define column groups
column_groups = {"TAP"         : ["TAP"],
                 "ECAS"        : ["ECAS"],
                 "ALSFRS"      : ["ALSFRS"],
                 "Tone"        : ["Tone"],
                 "Reflexes"    : ["Reflexes"],
                 "BIO"         : ["BIO"]}

# Define multiple options for BIO group
column_groups["BIO"] = ["Age",
                        "Sex",
                        "YearsEd",
                        "Handedness",
                        "Symptom_Duration",
                        "MedicalExamination_Riluzole"]


missing_counts_df_temp = missing_counts_df
for group, substrings in column_groups.items():
    columns = missing_counts_df_temp.index[missing_counts_df_temp.index.str.contains('|'.join(substrings))]
    column_groups[group] = columns
    missing_counts_df_temp = missing_counts_df_temp[~missing_counts_df_temp.index.isin(columns)]

df['Region_of_Onset'].fillna('unknown', inplace = True)

df['bulbar'] = np.array(1*[(df['Region_of_Onset'] == 'bulbar') |
                           (df['Region_of_Onset'] == 'bulbar_speech') |
                           (df['Region_of_Onset'] == 'bulbar_speech{@}bulbar_swallowing')]).T
df['spinal'] = np.array(1*[(df['Region_of_Onset'] == 'lower_extremity') |
                            (df['Region_of_Onset'] == 'upper_extremity') |
                            (df['Region_of_Onset'] == 'upper_extremity{@}lower_extremity')]).T
for group, columns in column_groups.items():
    print(f"columns_{group}:", columns)

combined_columns = [column 
                    for columns in column_groups.values() 
                    for column in columns]
# Behavioral data
behaviour_data_array = df[combined_columns]

df_spinal = df[(df['spinal'] == 1)]
df_bulbar = df[(df['bulbar'] == 1)]

df_spinal = df_spinal[combined_columns]
df_bulbar = df_bulbar[combined_columns]

#------------------------------------------------------------------------------
# Missing Value Count for Bulbar and Spinal Groups
#------------------------------------------------------------------------------

bulbar_missing_counts = df_bulbar.isna().sum().reset_index()
bulbar_missing_counts.columns = ['Measure', 'Bulbar_Missing_Count']

spinal_missing_counts = df_spinal.isna().sum().reset_index()
spinal_missing_counts.columns = ['Measure', 'Spinal_Missing_Count']

# Merge both for easy comparison
missing_counts_df = pd.merge(bulbar_missing_counts, spinal_missing_counts, on = 'Measure')

# Filter only measures with any missing values
missing_counts_df = missing_counts_df[(missing_counts_df['Bulbar_Missing_Count'] > -1) |
                                      (missing_counts_df['Spinal_Missing_Count'] > -1)]

#------------------------------------------------------------------------------
# Plot a barplot to show significant terms
#------------------------------------------------------------------------------

significant_columns = []

def add_star_between_bars(bar1, bar2, height, ax):
    """Add a star between two bars to indicate significance."""
    ax.annotate("", xy = (bar1.get_x() + bar1.get_width() / 2, height),
                xycoords = 'data',
                xytext = (bar2.get_x() + bar2.get_width() / 2, height),
                textcoords = 'data',
                arrowprops = dict(arrowstyle = "-",
                                ec = '#aaaaaa',
                                lw = 1.5),
                annotation_clip = False)
    ax.text((bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2,
            height,
            "*",
            ha = 'center',
            va = 'bottom',
            fontsize = 20)

p_vals_array = np.zeros((len(combined_columns),1))
for idx, column in enumerate(combined_columns):

    group_spinal = df_spinal[column].dropna()
    group_bulbar = df_bulbar[column].dropna()
    print(column)
    print(len(group_spinal))
    print(len(group_bulbar))
    print('-------------')
    p_vals = [stats.ttest_ind(group_spinal,
                group_bulbar,
                nan_policy = 'omit',
                equal_var = False).pvalue]
    p_vals_array[idx,:] = p_vals[0]

p_values_corr = multipletests(p_vals_array.flatten(), method = 'fdr_bh')[1]
significant_columns = []
significant_p = []
# Filter for significant columns based on corrected p-values
significant_columns = [col for idx, col
                       in enumerate(combined_columns)
                       if p_values_corr[idx] < 0.05]
significant_p = [col for idx, col
                       in enumerate(p_values_corr)
                       if p_values_corr[idx] < 0.05]

print("\nSignificant columns:", significant_columns)
print("\nSignificant columns:", significant_p)

n_rows = int(np.ceil(len(significant_columns) / 3.0))
n_cols = min(3, len(significant_columns))

fig, axs = plt.subplots(n_rows,
                        n_cols,
                        figsize = (15, 15),
                        squeeze = False)

for idx, column in enumerate(significant_columns):
    ax = axs[idx // n_cols, idx % n_cols]
    group_spinal = df_spinal[column].dropna()
    group_bulbar = df_bulbar[column].dropna()
    means = [group_spinal.mean(), group_bulbar.mean()]
    errors = [group_spinal.std(),
              group_bulbar.std(),]
    bars = ax.bar(['Spinal', 'Bulbar'],
                  means,
                  yerr = errors,
                  capsize = 5,
                  color = ['blue', 'red'])
    height = max(means) + 0.2 * max(means)
    add_star_between_bars(bars[0], bars[1], height, ax)
    ax.set_title(column)
    ax.set_ylabel('Value')

for idx in range(len(significant_columns), n_rows * n_cols):
    axs[idx // n_cols, idx % n_cols].axis('off')

plt.tight_layout()
plt.savefig(path_fig + '/Behavioral_differences_spinal_bulbar.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

significant_alsfrs_columns = [col for col in significant_columns if col.startswith('ALSFRS')]
n_alsfrs = len(significant_alsfrs_columns)
fig, ax = plt.subplots(figsize = (10, 5))
bar_width = 0.35
index = np.arange(n_alsfrs)

for idx, column in enumerate(significant_alsfrs_columns):
    group_spinal = df_spinal[column].dropna()
    group_bulbar = df_bulbar[column].dropna()
    mean_spinal = group_spinal.mean()
    mean_bulbar = group_bulbar.mean()
    error_spinal = group_spinal.std(),
    error_bulbar = group_bulbar.std(),
    plt.bar(index[idx] - bar_width/2,
            mean_spinal,
            yerr = error_spinal,
            capsize = 5,
            width = bar_width,
            label = 'Spinal' if idx == 0 else "",
            color = [162/255, 199/255, 255/255])
    plt.bar(index[idx] + bar_width/2,
            mean_bulbar,
            yerr = error_bulbar,
            capsize = 5,
            width = bar_width,
            label = 'Bulbar' if idx == 0 else "",
            color = [255/255, 164/255, 138/255])

ax.set_ylim(0, 5)
ax.set_xlabel('ALSFRS Columns')
ax.set_ylabel('Values')
ax.set_title('ALSFRS Scores by Phenotype')
ax.set_xticks(index)
ax.set_xticklabels(significant_alsfrs_columns, rotation = 45, ha = "right")
plt.tight_layout()
plt.savefig(path_fig + '/ALSFRS_differences_spinal_bulbar.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

significant_ref_columns = [col for col in significant_columns if col.startswith('Reflex')]
n_ref = len(significant_ref_columns)
fig, ax = plt.subplots(figsize = (10, 5))
bar_width = 0.35
index = np.arange(n_ref)
for idx, column in enumerate(significant_ref_columns):
    group_spinal = df_spinal[column].dropna()
    group_bulbar = df_bulbar[column].dropna()
    mean_spinal = group_spinal.mean()
    mean_bulbar = group_bulbar.mean()
    error_spinal = group_spinal.std(),
    error_bulbar = group_bulbar.std(),

    plt.bar(index[idx] - bar_width/2,
            mean_spinal,
            yerr = error_spinal,
            capsize = 5,
            width = bar_width,
            label = 'Spinal' if idx == 0 else "",
            color = [162/255, 199/255, 255/255])
    plt.bar(index[idx] + bar_width/2,
            mean_bulbar,
            yerr = error_bulbar,
            capsize = 5,
            width = bar_width,
            label = 'Bulbar' if idx == 0 else "",
            color = [255/255, 164/255, 138/255])

ax.set_ylim(-0.2, 1.4)
ax.set_xlabel('Reflexes Columns')
ax.set_ylabel('Values')
ax.set_title('Reflexes Scores by Phenotype')
ax.set_xticks(index)
ax.set_xticklabels(significant_ref_columns, rotation = 45, ha  ="right")
ax.legend()
plt.tight_layout()
plt.savefig(path_fig + '/Reflexes_differences_spinal_bulbar.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# END