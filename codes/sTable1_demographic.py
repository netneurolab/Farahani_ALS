"""
*******************************************************************************

Script purpose:

    This script is developed to create two summary tables for supplementary material.

Summary of results:

    -----------------------------------------------------------------------
    Control demgraphic:
         Study       Site_y   mean_age        Female (%)      total_subjects
    0   CALSNIC1    Calgary  54.300000        60.000000              10
    1   CALSNIC1   Edmonton  58.350000        45.000000              20
    2   CALSNIC1   Montreal  53.857143        28.571429               7
    3   CALSNIC1    Toronto  46.200000        70.000000              10
    4   CALSNIC1  Vancouver  52.875000        75.000000               8
    5   CALSNIC2    Calgary  59.300000        50.000000              10
    6   CALSNIC2   Edmonton  55.688889        57.777778              45
    7   CALSNIC2      Miami  51.545455        54.545455              11
    8   CALSNIC2   Montreal  53.235294        35.294118              17
    9   CALSNIC2     Quebec  62.647059        76.470588              17
    10  CALSNIC2    Toronto  53.166667        50.000000              18
    11  CALSNIC2       Utah  65.500000        50.000000               2

    ALS demographic:
         Study       Site_y   mean_age        Female (%)      total_subjects
    0   CALSNIC1    Calgary  58.000000        66.666667               9
    1   CALSNIC1   Edmonton  57.500000        45.000000              20
    2   CALSNIC1   Montreal  62.384615        15.384615              13
    3   CALSNIC1    Toronto  52.800000        33.333333              15
    4   CALSNIC1  Vancouver  58.571429        28.571429               7
    5   CALSNIC2    Calgary  55.500000        33.333333               6
    6   CALSNIC2   Edmonton  59.372093        46.511628              43
    7   CALSNIC2      Miami  63.857143        28.571429               7
    8   CALSNIC2   Montreal  59.058824        17.647059              17
    9   CALSNIC2     Quebec  62.285714        28.571429              21
    10  CALSNIC2    Toronto  64.037037        40.740741              27
    11  CALSNIC2       Utah  55.428571        28.571429               7

    Combined Demographics Summary (ALS & Controls):
                	Study	Site	
                	age_Control		sex_Control		num_Control	
                	age_ALS	    	sex_ALS    		num_ALS
    0	CALSNIC1	Calgary		54.3 	60.0 	10	58.0 	66.67	9
    1	CALSNIC1	Edmonton	58.35	45.0 	20	57.5 	45.0 	20
    2	CALSNIC1	Montreal	53.86	28.57	7 	62.38	15.38	13
    3	CALSNIC1	Toronto		46.2 	70.0 	10	52.8 	33.33	15
    4	CALSNIC1	Vancouver	52.87	75.0 	8 	58.57	28.57	7
    5	CALSNIC2	Calgary		59.3 	50.0 	10	55.5 	33.33	6
    6	CALSNIC2	Edmonton	55.69	57.78	45	59.37	46.51	43
    7	CALSNIC2	Miami   	51.55	54.55	11	63.86	28.57	7
    8	CALSNIC2	Montreal	53.24	35.29	17	59.06	17.65	17
    9	CALSNIC2	Quebec  	62.65	76.47	17	62.29	28.57	21
    10	CALSNIC2	Toronto		53.17	50.0 	18	64.04	40.74	27
    11	CALSNIC2	Utah    	65.50	50.0 	2 	55.43	28.57	7

    -----------------------------------------------------------------------

    ALS Onset Region Breakdown:
                                          Region_of_Onset  Count
    0                                     lower_extremity     69
    1                                     upper_extremity     64
    2                                       bulbar_speech     18
    3                                              bulbar     10
    4                   bulbar_speech{@}bulbar_swallowing     10
    5                   upper_extremity{@}lower_extremity      7
    6                     bulbar_speech{@}upper_extremity      2
    7                            bulbar{@}lower_extremity      1
    8                            bulbar{@}upper_extremity      1
    9                 bulbar_swallowing{@}upper_extremity      1
    10                    bulbar_speech{@}lower_extremity      1
    11  bulbar_speech{@}upper_extremity{@}lower_extremity      1
    12                    upper_extremity{@}ftd_cognitive      1

    Note:
        Six ALS patients have missing data for education, handedness, and onset type.

*******************************************************************************
"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import pandas as pd
from globals import  path_results

#------------------------------------------------------------------------------
# Load the cleaned demographic data (created bt the 'code01_demographic_cleaning.py' script)
#------------------------------------------------------------------------------

df = pd.read_csv(path_results + 'data_demographic_clean.csv')

df = df[(df['Diagnosis'] == 'ALS') |
        (df['Diagnosis'].str.contains('Control'))].reset_index(drop = True)

#------------------------------------------------------------------------------
# Stratify data of healthy controls and ALS patients - considering visit 1 data only
#------------------------------------------------------------------------------

# Healthy controls (consider only Visit 1)
df_hc = df[((df['Visit Label'].str.contains('Visit 1')) &
            (df['Diagnosis'] == 'Control'))].reset_index(drop = True)

# ALS subjects (consider only Visit 1)
df_als = df[((df['Visit Label'].str.contains('Visit 1')) &
             (df['Diagnosis'] == 'ALS'))].reset_index(drop = True)

study_als = df_als.Study
site_als = df_als.Site_y

#------------------------------------------------------------------------------
# Calculate the mean age and female percentage for ALS patients and controls
#------------------------------------------------------------------------------

def summarize_demographics(df):
    """
      Summarizes demographic information by calculating mean age, 
      female percentage, and total subject count grouped by 'Study' and 'Site_y'.

      # Note: 1 represents females.
    """
    summary = df.groupby(['Study', 'Site_y']).agg(
        mean_age = ('Age', 'mean'),
        female_percentage = ('Sex', lambda x: (x == 1).mean() * 100),
        total_subjects = ('Filename', 'count')
    ).reset_index()
    return summary

# Summarize demographic information for ALS patients and control subjects
als_demographics_summary = summarize_demographics(df_als)
control_demographics_summary = summarize_demographics(df_hc)

# Merge ALS and control demographics based on 'Study' and 'Site_y' - table 1
demographics_summary = pd.merge(control_demographics_summary,
                                als_demographics_summary,
                                on = ['Study', 'Site_y'],
                                suffixes = ('_Control', '_ALS'))

#------------------------------------------------------------------------------
# Count ALS onset regions - table 2
#------------------------------------------------------------------------------

# Count occurrences of each region of onset for ALS patients
subtype_counts = df_als['Region_of_Onset'].value_counts().reset_index()
subtype_counts.columns = ['Region_of_Onset', 'Count']

#------------------------------------------------------------------------------
# END