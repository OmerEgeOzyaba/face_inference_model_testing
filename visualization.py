import pandas as pd
import matplotlib.pyplot as plt

################ DeepFace plots #######################
# load age mae, gender accuracy, and gender confidance variance across races 
df_mae_by_age_bin = pd.read_csv("results/df_mae_by_age_bin.csv")
df_mean_gender_accuracy = pd.read_csv("results/df_mean_gender_accuracy.csv")
df_gender_confidence_variance = pd.read_csv("results/df_gender_confidence_variance.csv")

# visualize gender confidance variance across races 
plt.figure(figsize=(10, 5))
plt.bar(df_gender_confidence_variance['true_race'], df_gender_confidence_variance['predicted_gender_confidence'])
plt.title('DeepFace Gender Confidence Variance by Race')
plt.xlabel('Race')
plt.ylabel('Variance')
plt.tight_layout()
plt.savefig('figures/gender_confidence_variance_by_race.png', dpi=150)
plt.close()

############ deepface vs yu4u comparison plots ######
# load yu4u results
yu4u_mae = pd.read_csv("results/yu4u_mae_by_age_bin.csv")
yu4u_gender = pd.read_csv("results/yu4u_mean_gender_accuracy.csv")

# age MAE comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.bar(df_mae_by_age_bin['true_age_bin'], df_mae_by_age_bin['age_error'])
ax1.set_title('DeepFace Age MAE by Age Bin')
ax1.set_xlabel('Age Bin')
ax1.set_ylabel('MAE')
ax1.tick_params(axis='x', rotation=45)

ax2.bar(yu4u_mae['true_age_bin'], yu4u_mae['age_error'])
ax2.set_title('yu4u Age MAE by Age Bin')
ax2.set_xlabel('Age Bin')
ax2.set_ylabel('MAE')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/age_mae_comparison.png', dpi=150)
plt.close()

# gender accuracy comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.bar(df_mean_gender_accuracy['true_race'], df_mean_gender_accuracy['is_gender_accurate'])
ax1.set_title('DeepFace Gender Accuracy by Race')
ax1.set_xlabel('Race')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.4, 0.8)
ax1.tick_params(axis='x', rotation=45)

ax2.bar(yu4u_gender['true_race'], yu4u_gender['is_gender_accurate'])
ax2.set_title('yu4u Gender Accuracy by Race')
ax2.set_xlabel('Race')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0.4, 0.8)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/gender_accuracy_comparison.png', dpi=150)
plt.close()

print("Saved comparison plots.")