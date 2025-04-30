import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from scipy.special import expit  # for post-sigmoid conversion

# --- Load patient labels ---
def load_patient_labels(csv_path):
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str).str.zfill(3)
    df['label'] = df['condition'].apply(lambda x: 1 if "Parkinson's" in x and "Atypical" not in x else 0)
    return dict(zip(df['id'], df['label']))

# --- Load movement data ---
def load_movement_data(movement_path='preprocessed/movement', label_dict=None):
    X = []
    y = []
    for filename in os.listdir(movement_path):
        if filename.endswith('.bin'):
            subject_id = filename.split('_')[0]
            if subject_id in label_dict:
                label = label_dict[subject_id]
                data = np.fromfile(os.path.join(movement_path, filename), dtype=np.float32).reshape(132, 976)
                X.append(data)
                y.append(label)
    return np.array(X), np.array(y)

# --- Define deeper, lightweight CNN model ---
class DeepOmniScaleCNN(nn.Module):
    def __init__(self, input_channels=132, num_classes=1):
        super(DeepOmniScaleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)  # No Sigmoid here
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # Raw logits (no sigmoid)

# --- Load data ---
label_dict = load_patient_labels('/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/MLPR_project/patient_data.csv')
X, y = load_movement_data('/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/movement', label_dict)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Dataloaders
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model, loss, optimizer
model = DeepOmniScaleCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10): 
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = []
    labels = []
    for xb, yb in val_loader:
        out = model(xb)
        preds.append(out)
        labels.append(yb)

    preds = torch.cat(preds).squeeze().numpy()
    labels = torch.cat(labels).squeeze().numpy()

# Apply sigmoid to logits before classification report
probs = expit(preds)
print(classification_report(labels, probs > 0.5))
