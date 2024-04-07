"""
*******************************************************************************

Purpose:
    To clean and modify the demographic Excel file for a research study.
    This includes filtering out specific sites and subjects,
    filling in missing demographic info, adding DBM paths,
    and excluding certain patient categories (PLS, etc.).

Expected Output:
    A cleaned dataset with 792 entries saved as 'data_demographic_clean.csv'.

Results:
    -----------------------------------------------------------------------
    number_ALS_first_visit 192
    mean_age_subjects_ALSV1 59.619791666666664
    std_age_subjects_ALSV1 10.198356891672821
    Sex
    0.0    122
    1.0     70 - female
    Name: count, dtype: int64
    -----------------------------------------------------------------------
    number_Control_first_visit 175
    mean_age_subjects_controlV1 55.40571428571429
    std_age_subjects_controlV1 10.002340950487337
    Sex
    1.0    96 - female
    0.0    79
    Name: count, dtype: int64

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import glob
import datetime
import warnings
import numpy as np
import pandas as pd
from IPython import get_ipython

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def days_to_months(delta):
    """
    Converts time delta to months
    """
    months = delta.days / 30.44 # Average days in a month
    return months

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

base_path        = '/Users/asaborzabadifarahani/Desktop/ALS/ALS_git/'
path_data        = '/Users/asaborzabadifarahani/Desktop/ALS/Data_nifti/'
path_demographic = os.path.join(base_path, 'data/DemographicInfo/')
path_results     = os.path.join(base_path, 'results/')

#------------------------------------------------------------------------------
# Get available DBM file names
#------------------------------------------------------------------------------

folder_names = sorted(glob.glob(path_data + '/*'))

# Function to process file names and extract relevant information
column_names = ['Filename',
                'Study',
                'Visit Label',
                'Path',
                'Site',
                'Status']

def process_file_name(file_name, folder_index):
    """
    Extracts relevant information from file names.
    """
    return {
        'Filename'   : file_name[9:-7],
        'Study'      : file_name[9:-16],
        'Visit Label': f'Visit {file_name[-5]}',
        'Path'       : os.path.join(folder_name, file_name),
        'Site'       : folder_index,
        'Status'     : file_name[-11]
        }

# Get file names within a site folder and process the name of it to extract info
data_records = []
for i, folder_name in enumerate(folder_names):
    files = os.listdir(folder_name)
    for file_name in files:
        data_records.append(process_file_name(file_name, i))

# Create DataFrame based on the extracted information

df = pd.DataFrame(data_records, columns=column_names).astype({name: 'category'
                                                              for name in column_names})
df = df.reset_index(drop = True)

#------------------------------------------------------------------------------
# Load original demographic information
#------------------------------------------------------------------------------

demographic_info = pd.read_excel(path_demographic + "Data_Demographic.xlsx")

#------------------------------------------------------------------------------
# Cleaning of the dataset - nine different steps are included in the following
#------------------------------------------------------------------------------

# 1. Clean values by removing trailing spaces ( both datasets )

columns_to_strip = ['Visit Label',
                    'Filename',
                    'Diagnosis']
demographic_info[columns_to_strip] = demographic_info[columns_to_strip].apply(lambda x: x.str.strip())

columns_to_strip = ['Visit Label',
                    'Filename']
df[columns_to_strip] = df[columns_to_strip].apply(lambda x: x.str.strip())

#------------------------------------------------------------------------------
# 2. Group by 'Filename' and fill in missing values using forward fill within each subject

columns_to_fill = ['Sex',
                   'Age',
                   'Patient or Control',
                   'Diagnosis',
                   'Symptom_Duration',
                   'Region_of_Onset',
                   'Side_1st_MotorSymptomOnset',
                   'MedicalExamination_Riluzole']

demographic_info[columns_to_fill] = demographic_info.groupby('Filename')[columns_to_fill].ffill()

# Replace values in 'Diagnosis' and 'Patient or Control' columns based on conditions
demographic_info.loc[demographic_info['Patient or Control'] == 'Control', 'Diagnosis'] = 'Control'
demographic_info.loc[demographic_info['Diagnosis'] == 'Control', 'Patient or Control'] = 'Control'
demographic_info.loc[demographic_info['Diagnosis'] == 'ALS', 'Patient or Control'] = 'Patient'

# Replace 'Study' with CALSNIC1/2 based on Filename
demographic_info['Study'] = demographic_info['Filename'].str.extract('(CALSNIC1|CALSNIC2)')

# Replace 'Site'columns with Site information embedded in the Filename
site_mapping = {
    '_CAL_': 'Calgary',
    '_EDM_': 'Edmonton',
    '_LON_': 'London',
    '_MON_': 'Montreal',
    '_TOR_': 'Toronto',
    '_QUE_': 'Quebec',
    '_UTA_': 'Utah',
    '_MIA_': 'Miami',
    '_VAN_': 'Vancouver'}

for substring, site_value in site_mapping.items():
    demographic_info.loc[demographic_info['Filename'].str.contains(substring),
                         'Site'] = site_value

#------------------------------------------------------------------------------
# 3. Filter rows where the diagnosis is not 'Control' or 'ALS' (as we have cases of PLS, etc.)

diagnosis_to_keep = ['Control',
                     'ALS']
demographic_info = demographic_info[demographic_info['Diagnosis'].isin(diagnosis_to_keep)]
demographic_info = demographic_info.reset_index(drop = True)

#------------------------------------------------------------------------------
# 4. Complete information regarding the scan time

MRI_timing = demographic_info[['Filename', 'MRI_Date']]
filenames_uniq = np.array(MRI_timing['Filename'].unique())
list_of_dfs = []

for i, x in enumerate(filenames_uniq):
    MRI_timing_1 = demographic_info[demographic_info['Filename'] == x]
    MRI_timing_1 = MRI_timing_1.reset_index()
    
    sorted_MRI_timing_1 = MRI_timing_1.sort_values(by = 'Visit Label')
    time_onset = MRI_timing_1.SymptomOnset_Date[0]
    sorted_MRI_timing_1['Diff_time'] = np.nan

    if type(time_onset) != pd._libs.tslibs.timestamps.Timestamp:
        pass
    else:
        for j in range(len(sorted_MRI_timing_1)):
            if sorted_MRI_timing_1['Visit Label'][j] == 'Visit 1':
                time_1 = MRI_timing_1.MRI_Date[j]
                if type(time_1) != datetime.datetime:
                    pass 
                else:
                    sorted_MRI_timing_1['Diff_time'][j] = days_to_months(time_1 - time_onset)
            else:
                time_2 = MRI_timing_1.MRI_Date[j]
                if type(time_2) != datetime.datetime:
                    pass 
                else:
                    if time_2 == 'not_done':
                        sorted_MRI_timing_1['Diff_time'][j] == 'not valid'
                    else:
                        sorted_MRI_timing_1['Diff_time'][j] = days_to_months(time_2 - time_onset)
    list_of_dfs.append(sorted_MRI_timing_1)


# Combine the list of dataframes into a single dataframe
combined_df = pd.concat(list_of_dfs, ignore_index=True)

#------------------------------------------------------------------------------
# 5. Fill in the missing value in Diff_time according to Symptom_Duration and
# The approximate timing between scans provided by the paper (4 and 8)

visit_mapping = {
    'Visit 1' : 0,
    'Visit 2' : 4,
    'Visit 3' : 8}

for visit_label, time_offset in visit_mapping.items():
    mask = combined_df['Visit Label'] == visit_label
    combined_df.loc[mask, 'Diff_time'] = combined_df.loc[mask, 'Symptom_Duration'] + time_offset

#------------------------------------------------------------------------------
# 6. Merge DataFrames based on all available shared columns

demographic_info_include_diffTime = pd.merge(demographic_info,
                                             combined_df,
                                             on = demographic_info.columns.intersection(combined_df.columns).tolist(),
                                             how = 'left')

data_include_diffTime = pd.merge(df,
                                 demographic_info_include_diffTime,
                                 on = ['Filename',
                                       'Study',
                                       'Visit Label'],
                                 how = 'inner')
#------------------------------------------------------------------------------
# 7. Identify filenames with non-empty 'Inclusion for Cognitive Analysis

filenames_to_remove = data_include_diffTime[data_include_diffTime['Inclusion for Cognitive Analysis'].notna()]['Filename'].unique()

# From these, select filenames that should be kept because they have a specific caution note
filenames_to_keep = data_include_diffTime[
    (data_include_diffTime['Filename'].isin(filenames_to_remove)) &
    (data_include_diffTime['Cautionary notes on MRI and Cognitive Analysis          '] == 'Incorrect ECAS administration ')]['Filename'].unique()

# Now, filter out rows with filenames to remove, but keep the specific ones with the caution note
data_include_diffTime = data_include_diffTime[
    (~data_include_diffTime['Filename'].isin(filenames_to_remove)) |
    (data_include_diffTime['Filename'].isin(filenames_to_keep))]

#------------------------------------------------------------------------------
# 8. Remove sites that have a low number of subjects

data_include_diffTime['Site_x']
value_counts = data_include_diffTime['Site_x'].value_counts()
values_to_remove = value_counts[value_counts < 10].index
data_include_diffTime = data_include_diffTime[~data_include_diffTime['Site_x'].isin(values_to_remove)]

#------------------------------------------------------------------------------
# 9. Replace categorical columns with numbers - sex

sex_mapping = {'Male'  : 0.0,
               'Female': 1.0}

data_include_diffTime['Sex'] = data_include_diffTime['Sex'].map(sex_mapping)

#------------------------------------------------------------------------------
# Save the cleaned dataset
#------------------------------------------------------------------------------

data_include_diffTime.to_csv(path_results + 'data_demographic_clean.csv',
                             index = False)

#------------------------------------------------------------------------------
# Visualize some demographic information used in the paper
#------------------------------------------------------------------------------

print('-----------------------------------------------------------------------')

number_ALS_first_visit = sum((data_include_diffTime['Diagnosis'] == 'ALS') &
                             (data_include_diffTime['Visit Label'] == 'Visit 1'))
print('number_ALS_first_visit ' + str(number_ALS_first_visit))
mean_age_subjects_ALSV1 = np.mean(data_include_diffTime[(data_include_diffTime['Diagnosis'] == 'ALS') &
                                                        (data_include_diffTime['Visit Label'] == 'Visit 1')].Age)
print('mean_age_subjects_ALSV1 ' + str(mean_age_subjects_ALSV1))
std_age_subjects_ALSV1 = np.std(data_include_diffTime[(data_include_diffTime['Diagnosis'] == 'ALS') &
                                                      (data_include_diffTime['Visit Label'] == 'Visit 1')].Age)
print('std_age_subjects_ALSV1 ' + str(std_age_subjects_ALSV1))
unique_counts_ALS = data_include_diffTime[(data_include_diffTime['Diagnosis'] == 'ALS') &
                                          (data_include_diffTime['Visit Label'] == 'Visit 1')]['Sex'].value_counts()
print(unique_counts_ALS)

print('-----------------------------------------------------------------------')

number_Control_first_visit = sum((data_include_diffTime['Diagnosis'] == 'Control') &
                                 (data_include_diffTime['Visit Label'] == 'Visit 1'))
print('number_Control_first_visit ' + str(number_Control_first_visit))
mean_age_subjects_controlV1 = np.mean(data_include_diffTime[ (data_include_diffTime['Diagnosis'] == 'Control') &
                                                            (data_include_diffTime['Visit Label'] == 'Visit 1')].Age)
print('mean_age_subjects_controlV1 ' + str(mean_age_subjects_controlV1))
std_age_subjects_controlV1 = np.std(data_include_diffTime[ (data_include_diffTime['Diagnosis'] == 'Control') &
                                                          (data_include_diffTime['Visit Label'] == 'Visit 1')].Age)
print('std_age_subjects_controlV1 ' + str(std_age_subjects_controlV1))
unique_counts_Control = data_include_diffTime[(data_include_diffTime['Diagnosis'] == 'Control') &
                                              (data_include_diffTime['Visit Label'] == 'Visit 1')]['Sex'].value_counts()
print(unique_counts_Control)

print('-----------------------------------------------------------------------')

print('number of subjects with ALS within each imaging site:')
unique_counts_Control = data_include_diffTime[(data_include_diffTime['Diagnosis'] == 'ALS') &
                                              (data_include_diffTime['Visit Label'] == 'Visit 1')]['Site_x'].value_counts()
print(unique_counts_Control)

print('-----------------------------------------------------------------------')

#------------------------------------------------------------------------------
# END