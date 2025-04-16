import numpy as np
import os
from tqdm import tqdm

#Path to the folder containing the .bin files
DATA_DIR = "/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/movement"

def load_bin_file(filepath):
    data = np.fromfile(filepath, dtype=np.float32)
    assert data.shape[0] == 128832, f"Expected 128832 floats, got {data.shape[0]}"
    reshaped = data.reshape(132, 976)  # (channels, time_steps)
    return reshaped

def load_all_files(data_dir):
    X = []
    filepaths = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".bin")
    ])
    print(f"Found {len(filepaths)} files.")

    for path in tqdm(filepaths):
        try:
            X.append(load_bin_file(path))
        except Exception as e:
            print(f"Error with {path}: {e}")

    X = np.stack(X)
    return X, filepaths

if __name__ == "__main__":
    X, filepaths = load_all_files(DATA_DIR)
    print("All data loaded successfully.")
    print(f"Shape of X: {X.shape}")  # Should be (469, 132, 976)