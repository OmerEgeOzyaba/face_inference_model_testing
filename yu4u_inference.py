from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from omegaconf import OmegaConf
from src.factory import get_model
from datasets import load_dataset

# get weights
weight_file = "pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5"
margin = 0 

# load the model
model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
model = get_model(cfg)
model.load_weights(weight_file)

#load samples and fairface
samples = pd.read_csv("data/fairface_sample.csv")
ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")

# start inference
results = []

for i, content in samples.iterrows():
    if i % 100 == 0:
        print(f"Processed {i}/{len(samples)}")
    # get image
    image = ds[int(content['index'])]['image']

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (img_size, img_size))
    input_data = np.expand_dims(img_resized, axis=0)
    
    predicted_age = None
    predicted_gender = None
    
    # perform prediction
    try:
        results_model = model.predict(input_data, verbose=0)
        predicted_gender_raw = results_model[0][0][0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_age = float(results_model[1].dot(ages).flatten()[0])
        predicted_gender = "Male" if predicted_gender_raw < 0.5 else "Female"
    except Exception as e:
        print(f"Error: {e}")
      
    results.append({
    'index'       : content['index'],
    'true_age_bin': content['age_bin'],
    'true_gender' : content['gender'],
    'true_race'   : content['race'],
    'yu4u_age'    : predicted_age,
    'yu4u_gender' : predicted_gender,
})

# save results
df = pd.DataFrame(results)
df.to_csv('data/yu4u_results.csv', index=False)
print(f"Done. Saved {len(df)} rows to yu4u_results.csv")
