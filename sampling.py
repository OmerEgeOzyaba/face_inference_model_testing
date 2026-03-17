from datasets import load_dataset
import pandas as pd

# Load dataset
ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")
print(f"Loaded {len(ds)} images.")
print()

# Turn into a DataFrame
records = []
age_names = ds.features['age'].names
gender_names = ds.features['gender'].names  
race_names   = ds.features['race'].names

for index, item in enumerate(ds):
    records.append({
        'index'    : index,
        'age_bin': age_names[item['age']],
        'gender' : gender_names[item['gender']],
        'race'   : race_names[item['race']],
    })

df = pd.DataFrame(records)

# Define age bin categories for sampling
keep_all_bins = ['0-2', '3-9', '10-19', 'more than 70']
adult_bins = ['20-29', '30-39', '40-49', '50-59', '60-69']
cap_per_race = 50

# implement sampling logic
samples = []
keep_all = df[df['age_bin'].isin(keep_all_bins)]
samples.append(keep_all)

for bin in adult_bins:
    for race in df['race'].unique():
        cell = df[(df['race'] == race) & (df['age_bin'] == bin)]
        n = min(len(cell), cap_per_race)
        samples.append(cell.sample(n, random_state=1))

samples_df = pd.concat(samples).reset_index(drop=True)

# save sample to csv file
samples_df.to_csv('data/fairface_sample.csv', index=False)
print(f"Total sampled: {len(samples_df)}")
