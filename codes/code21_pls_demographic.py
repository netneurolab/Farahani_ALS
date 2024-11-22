"""
*******************************************************************************

Script purpose:

    Print the number of subjects with missing data for each behavioral data.
    PLS analysis-related

Script output:

    ----------------------------------------------------------------------

    Measure -->	Missing Subjects
    Handedness	6
    YearsEd	6
    Symptom_Duration	6
    ALSFRS_1_Speech	5
    ALSFRS_2_Salivation	5
    ALSFRS_3_Swallowing	5
    ALSFRS_4_Handwriting	5
    ALSFRS_5_Cuttingfood&handlingutensils	5
    ALSFRS_6_Dressing&hygiene	5
    ALSFRS_7_Turninginbed	5
    ALSFRS_8_Walking	5
    ALSFRS_9_Climbingstairs	5
    ALSFRS_10_Dyspnea	5
    ALSFRS_11_Orthopnea	5
    ALSFRS_12_RespiratoryInsufficiency	5
    ALSFRS_TotalScore	5
    TAP_Trial1RightFinger	11
    TAP_Trial1LeftFinger	11
    TAP_Trial2RightFinger	11
    TAP_Trial2leftFinger	11
    TAP_Trial1RightFoot	12
    TAP_Trial1LeftFoot	12
    TAP_Trial2RightFoot	12
    TAP_Trial2LeftFoot	12
    TAP_Fingertapping_Right_avg	11
    TAP_Fingertapping_Left_avg	11
    TAP_Foottapping_Right_avg	12
    TAP_Foottapping_Left_avg	12
    Tone_RightArm	16
    Tone_LeftArm	16
    Tone_RightLeg	17
    Tone_LeftLeg	17
    Reflexes_Jaw	19
    Reflexes_RightArm	16
    Reflexes_LeftArm	16
    Reflexes_RightLeg	15
    Reflexes_LeftLeg	16
    ECAS_Naming	12
    ECAS_Comprehension	12
    ECAS_Spelling	12
    ECAS_VerbFluencyS	12
    ECAS_VerbFluencyT	12
    ECAS_RevDigSpan	12
    ECAS_Alternation	12
    ECAS_SentenceCompletion	12
    ECAS_SocialCognition	12
    ECAS_ImmediateRecall	12
    ECAS_DelayedRecall	12
    ECAS_DelayedRecognition	12
    ECAS_DotCounting	12
    ECAS_CubeCounting	12
    ECAS_NumberLocation	12
    ECAS_LanguageTotal	12
    ECAS_VerbalFluencyTotal	12
    ECAS_ExecutiveTotal	12
    ECAS_ALSSpecific Total	12
    ECAS_MemoryTotal	12
    ECAS_VisuospatialTotal	12
    ECAS_ALSNonSpecific Total	12
    ECAS_TotalScore	12
    Diff_time	6
    Age	0
    Sex	0

    ----------------------------------------------------------------------

NOTE:

    The results coming from this script are shown as a supplementary table (Table. S7).

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import pandas as pd
from globals import path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Clinical measure handling
#------------------------------------------------------------------------------

# Load demographic information
df =  pd.read_csv(path_results + 'df_ALS_all.csv')

handedness_mapping = {'right': 0,
                      'left': 1,
                      'ambidextrous': 2}

df['Handedness'] = df['Handedness'].map(handedness_mapping)
med = {'yes': 1,
       'no': 0
       }
df['MedicalExamination_Riluzole'] = df['MedicalExamination_Riluzole'].map(med)
df = df[(df['Diagnosis'] == 'ALS') | 
        (df['Diagnosis'].str.contains('Control'))]

df = df[((df['Visit Label'].str.contains('Visit ' + str(1))) &
             (df['Diagnosis'] == 'ALS'))].reset_index(drop = True)

filenames_to_remove = df[df['Inclusion for Cognitive Analysis'].notna()]
indices_with_caution = filenames_to_remove.index
df = df.drop(indices_with_caution).reset_index(drop = True)
num_subjects = len(df)

# Get a list of columns with missing values
columns_with_missing = df.columns[df.isnull().any()]

# Filter out categorical columns from the list
numerical_columns_with_missing = columns_with_missing.intersection(df.select_dtypes(include = ['number']).columns)

# Create a dictionary to store the counts of unique subjects with missing values for each column
missing_counts = {}
# Iterate through columns with missing values
for column in numerical_columns_with_missing:
    missing_subjects = df[df[column].isnull()]['Filename'].unique()
    missing_counts[column] = len(missing_subjects)

# Add columns with zero missing values to the dictionary
numerical_columns_without_missing = df.select_dtypes(include = ['number']).columns.difference(numerical_columns_with_missing)
for column in numerical_columns_without_missing:
    missing_counts[column] = 0

# Alternatively, you can create a DataFrame from the dictionary for easier analysis
missing_counts_df = pd.DataFrame.from_dict(missing_counts,
                                           orient = 'index',
                                           columns = ['Missing Subjects'])

# Optimize the filtering conditions
limit_num_sub_with_nan = 21 # Remove the feature if it was rarely recorded across subjects
conditions = ((missing_counts_df['Missing Subjects'] <= limit_num_sub_with_nan) &
      (~missing_counts_df.index.str.contains("date|Date|index|Unnamed|Visit_details|DBM|Site|Status|TimePoint")))
missing_counts_df = missing_counts_df[conditions]
#------------------------------------------------------------------------------
# END