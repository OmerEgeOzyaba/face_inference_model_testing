import pandas as pd
from scipy import stats

# Load deepface and yu4u results
deepface_results = pd.read_csv("data/deepface_results.csv")
yu4u_results = pd.read_csv("data/yu4u_results.csv")

# reorder age bins for deepface
age_bin_order = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
deepface_results['true_age_bin'] = pd.Categorical(deepface_results['true_age_bin'], categories=age_bin_order, ordered=True)
# reorder age bins for yu4u
yu4u_results['true_age_bin'] = pd.Categorical(yu4u_results['true_age_bin'], categories=age_bin_order, ordered=True)

###################### DeepFace ANALYSIS ##################
# start analysis for age
# define midpoint for each age bin in FairFace for analysis
fairface_age_midpoints = {
    '0-2': 1,
    '3-9': 6,
    '10-19': 15,
    '20-29': 25,
    '30-39': 35,
    '40-49': 45,
    '50-59': 55,
    '60-69': 65,
    'more than 70': 75
}

deepface_results['true_age_midpoint'] = deepface_results['true_age_bin'].map(fairface_age_midpoints).astype(float)

# compute MAE
deepface_results['age_error'] = abs(deepface_results['deepface_age'] - deepface_results['true_age_midpoint'])
mae_by_age_bin = deepface_results.groupby('true_age_bin')['age_error'].mean()

# start analysis for gender
# map deepface values to fairface values
deepface_genders = {'Man': 'Male',
                    'Woman': 'Female',}
deepface_results['deepface_standardized_gender'] = deepface_results['deepface_gender'].map(deepface_genders)

#compare genders and put t/f 
deepface_results['is_gender_accurate'] = deepface_results['deepface_standardized_gender'] == deepface_results['true_gender']

# compute mean of is_gender_accurate grouped by race group
mean_gender_accuracy = deepface_results.groupby('true_race')['is_gender_accurate'].mean()

#define a parser for the the gender confidence values
import re

def parse_gender_conf(s):
    numbers = re.findall(r'\d+\.\d+', s)
    return {'Woman': float(numbers[0]), 'Man': float(numbers[1])}

deepface_results['gender_confidence'] = deepface_results['gender_confidence'].map(parse_gender_conf)

# extract confidence score for the predicted gender from the confidence dictionary
deepface_results['predicted_gender_confidence'] = deepface_results.apply(
    lambda row: row['gender_confidence'][row['deepface_gender']], axis=1
)
gender_confidence_variance = deepface_results.groupby('true_race')['predicted_gender_confidence'].var()

# save results to deepface_analysis.csv
mae_by_age_bin.to_frame().to_csv('results/df_mae_by_age_bin.csv')
mean_gender_accuracy.to_frame().to_csv('results/df_mean_gender_accuracy.csv')
gender_confidence_variance.to_frame().to_csv('results/df_gender_confidence_variance.csv')
print("Saved all results.")

# kruskall wallis on age_error across true_age_bin
groups = []
for name, group in deepface_results.groupby('true_age_bin'):
    groups.append(group['age_error'].values)

kruskal_age = stats.kruskal(*groups)

# krukal wallis on predicted_gender_confidence across true_race
groups = []
for name, group in deepface_results.groupby('true_race'):
    groups.append(group['predicted_gender_confidence'].values)

kruskal_gender = stats.kruskal(*groups)

# save the analysis 
kruskal_age_result = pd.DataFrame([{
    'test': 'age_error_by_age_bin',
    'statistic': kruskal_age.statistic,
    'pvalue': kruskal_age.pvalue
}])

kruskal_gender_result = pd.DataFrame([{
    'test': 'gender_conf_by_race',
    'statistic': kruskal_gender.statistic,
    'pvalue': kruskal_gender.pvalue
}])

pd.concat([kruskal_age_result, kruskal_gender_result]).to_csv('results/df_kruskal_results.csv', index=False)

############## yu4u ANALYSIS ####################
# compute age midpoints
yu4u_results['true_age_midpoint'] = yu4u_results['true_age_bin'].map(fairface_age_midpoints).astype(float)

# compute MAE
yu4u_results['age_error'] = abs(yu4u_results['yu4u_age'] - yu4u_results['true_age_midpoint'])
yu4u_mae_by_age_bin = yu4u_results.groupby('true_age_bin')['age_error'].mean()

# compare genders
yu4u_results['is_gender_accurate'] = yu4u_results['yu4u_gender'] == yu4u_results['true_gender']

# compute mean gender accuracy grouped by race
yu4u_mean_gender_accuracy = yu4u_results.groupby('true_race')['is_gender_accurate'].mean()

# save results
yu4u_mae_by_age_bin.to_frame().to_csv('results/yu4u_mae_by_age_bin.csv')
yu4u_mean_gender_accuracy.to_frame().to_csv('results/yu4u_mean_gender_accuracy.csv')

# kruskal wallis on age_error across true_age_bin
groups = []
for name, group in yu4u_results.groupby('true_age_bin'):
    groups.append(group['age_error'].values)
yu4u_kruskal_age = stats.kruskal(*groups)

# kruskal wallis on is_gender_accurate across true_race
groups = []
for name, group in yu4u_results.groupby('true_race'):
    groups.append(group['is_gender_accurate'].values)
yu4u_kruskal_gender = stats.kruskal(*groups)

# save kruskal results
yu4u_kruskal_age_result = pd.DataFrame([{
    'test': 'age_error_by_age_bin',
    'statistic': yu4u_kruskal_age.statistic,
    'pvalue': yu4u_kruskal_age.pvalue
}])

yu4u_kruskal_gender_result = pd.DataFrame([{
    'test': 'gender_accuracy_by_race',
    'statistic': yu4u_kruskal_gender.statistic,
    'pvalue': yu4u_kruskal_gender.pvalue
}])

pd.concat([yu4u_kruskal_age_result, yu4u_kruskal_gender_result]).to_csv('results/yu4u_kruskal_results.csv', index=False)
print("yu4u analysis saved.")
