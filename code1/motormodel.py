import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import wandb

wandb.init(project="parkinsons-detection", name="OmniScaleCNN")

def load_patient_labels(csv_path):
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str).str.zfill(3)
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
                data = np.fromfile(os.path.join(movement_path, filename), dtype=np.float32).reshape(132, 976)
                X.append(data)
                y.append(label)
    return np.array(X), np.array(y)

label_dict = load_patient_labels('/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/MLPR_project/patient_data.csv')
X, y = load_movement_data('/Users/trushamaheshwari/Downloads/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed/movement', label_dict)


class OmniScaleCNN(nn.Module):
    def __init__(self, input_channels=132, num_classes=1):
        super(OmniScaleCNN, self).__init__()

        # Branch 1 - Short-scale patterns
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 2 - Medium-scale patterns
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 3 - Long-scale patterns
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=11, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Combine branches
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid()  # for binary classification
        )

    def forward(self, x):
        # x: (batch, channels, time)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # Flatten and concatenate outputs
        b1 = b1.view(x.size(0), -1)
        b2 = b2.view(x.size(0), -1)
        b3 = b3.view(x.size(0), -1)

        out = torch.cat([b1, b2, b3], dim=1)
        return self.classifier(out)

# Assume X.shape = (num_samples, 132, 976)
# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Dataloader
from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model, Loss, Optimizer
model = OmniScaleCNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(15): 
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
    wandb.log({"Train Loss": train_loss / len(train_loader), "epoch": epoch+1})


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

# Binary classification report
from sklearn.metrics import classification_report
print(classification_report(labels, preds > 0.5))

from sklearn.metrics import classification_report

report = classification_report(labels, preds > 0.5, output_dict=True)
wandb.log({
    "Precision": report['1']['precision'],
    "Recall": report['1']['recall'],
    "F1 Score": report['1']['f1-score'],
    "Accuracy": report['accuracy']
})

wandb.finish()

