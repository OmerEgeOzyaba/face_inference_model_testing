import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load dataset
#   - loading the validation split of the dataset to understand the structure
#   - load with 1.25 padding around the face to get more face content

print("Loading FairFace validation split")
ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")
print(f"Loaded {len(ds)} images.\n")

# Print one example data to see what it looks like

example = ds[0]
print("###### Single example ######")
print(example)
print("############################")

# convert labels to pandas dataframe

records = []
age_names    = ds.features['age'].names 
gender_names = ds.features['gender'].names  
race_names   = ds.features['race'].names

for item in ds:
    records.append({
        'age_bin' : age_names[item['age']],
        'gender'  : gender_names[item['gender']],
        'race'    : race_names[item['race']],
    })

df = pd.DataFrame(records)

# print how many values there are for each label

print("### Distribution by race ###")
print(df['race'].value_counts().to_string())

print("### Distribution by gender ###")
print(df['gender'].value_counts().to_string())

print("### Distribution by age_bin ###")
print(df['age_bin'].value_counts().to_string())
print("############################")

# check contingency tables for race x age, race x gender, and age x gender

print("### race x age counts table ###")
crosstab = pd.crosstab(df['race'], df['age_bin'])
print(crosstab.to_string())
print("### Smallest race x age cells ##")
melted = crosstab.stack().reset_index()
melted.columns = ['race', 'age_bin', 'count']
print(melted.nsmallest(10, 'count').to_string(index=False))
print()

print("### race x gender crosstab ###")
crosstab = pd.crosstab(df['race'], df['gender'])
print(crosstab.to_string())
print("### Smallest race x gender cells ##")
melted = crosstab.stack().reset_index()
melted.columns = ['race', 'gender', 'count']
print(melted.nsmallest(10, 'count').to_string(index=False))
print()

print("### age x gender crosstab ###")
crosstab = pd.crosstab(df['age_bin'], df['gender'])
print(crosstab.to_string())
print("### Smallest age x gender cells ##")
melted = crosstab.stack().reset_index()
melted.columns = ['age-bin', 'gender', 'count']
print(melted.nsmallest(10, 'count').to_string(index=False))


print("############################")

# visualize the data distribution
print("Visualising data distribution")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("FairFace Validation Split Label Distributions", fontsize=14)

# race distribution
df['race'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title("By race")
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', rotation=45)

# gender distribution
df['gender'].value_counts().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title("By gender")
axes[1].set_xlabel("")

# age distribution
df['age_bin'].dropna().value_counts().plot(
    kind='bar', ax=axes[2], color='mediumseagreen'
)
axes[2].set_title("By Age Bin")
axes[2].set_xlabel("")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("fairface_distributions.png", dpi=150, bbox_inches='tight')
print("Saved distribution plot to fairface_distributions.png")
print("############################")

