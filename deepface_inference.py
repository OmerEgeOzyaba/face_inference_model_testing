from deepface import DeepFace
from datasets import load_dataset
import pandas as pd

# Load data from fairface_samples.csv
samples = pd.read_csv("data/fairface_sample.csv")

# Load data from the dataset
ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")

results = []

# For each sampled data, index it back to dataset and run deepface on it
for i, content in samples.iterrows():
    if i % 100 == 0:
        print(f"Processed {i}/{len(samples)}")
    # get image
    image = ds[int(content['index'])]['image']

    # temporarily save image to a file
    tmp_path = "tmp_img.jpg"
    image.save(tmp_path)

    predicted_age = None
    predicted_gender = None
    gender_confidence = None

    # run deepface on the image
    try:
        analysis = DeepFace.analyze(img_path = tmp_path, actions=["age", "gender"], enforce_detection = False)
        predicted_age = analysis[0]['age']
        predicted_gender = analysis[0]['dominant_gender']
        gender_confidence = analysis[0]['gender']
    except Exception as e:
        print(f"Error in DeepFace inference pipeline: {e}")
    
    results.append({
        'index'            : content['index'],
        'true_age_bin'     : content['age_bin'],
        'true_gender'      : content['gender'],
        'true_race'        : content['race'],
        'deepface_age'     : predicted_age,
        'deepface_gender'  : predicted_gender,
        'gender_confidence': gender_confidence,
    })

# save deepface results as csv file

df = pd.DataFrame(results)
df.to_csv('data/deepface_results.csv', index=False)

print(f"Done. Saved {len(df)} rows to deepface_results.csv")

