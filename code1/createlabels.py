import pandas as pd
import os
import numpy as np

def load_patient_labels(csv_path):
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str).str.zfill(3)  # ensures IDs like 1 → "001"
    df['label'] = df['condition'].apply(lambda x: 1 if "Parkinson's" in x and "Atypical" not in x else 0)
    return dict(zip(df['id'], df['label']))

def load_movement_data(movement_path='preprocessed/movement', label_dict=None):
    X = []
    y = []

    for filename in os.listdir(movement_path):
        if filename.endswith('.bin'):
            subject_id = filename.split('_')[0]
            if subject_id in label_dict:
                label = label_dict[subject_id]
                data = np.fromfile(os.path.join(movement_path, filename), dtype=np.float32).reshape(-1, 66)
                X.append(data)
                y.append(label)

    return np.array(X), np.array(y)

# Example usage
if __name__ == "__main__":
    label_dict = load_patient_labels('/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/MLPR_project/patient_data.csv')  # ✅ make sure file is in this path
    X, y = load_movement_data('/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/movement', label_dict)
    print(f"Loaded {len(X)} movement samples.")

for patient_id, label in label_dict.items():
    print(f"Patient ID: {patient_id}, Label: {label}")

import pandas as pd

# Convert the label_dict to a DataFrame
label_df = pd.DataFrame(list(label_dict.items()), columns=["id", "label"])

# Save to CSV
label_df.to_csv("/Users/trushamaheshwari/Desktop/pads-park", index=False)

label_df = pd.DataFrame(list(label_dict.items()), columns=["id", "label"])
label_df.to_csv("/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/MLPR_project/patient_labels.csv", index=False)
print("Saved patient_labels.csv to our files folder.")