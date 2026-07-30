import csv
import torch
import torch.nn as nn
import torch.optim as optim
from intent_model import PedestrianIntentLSTM
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

SEQUENCE_LENGTH = 20
FEATURE_COUNT = 10

def load_labeled_data(csv_path):
    rows_by_track = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            track_id = int(row[0])
            frame_number = int(row[1])
            features = [float(v) for v in row[2:2 + FEATURE_COUNT]]
            label = row[-1]
            label = int(label) if label not in ("", "None") else None
            rows_by_track.setdefault(track_id, []).append((frame_number, features, label))

    sequences, labels = [], []
    for track_id, rows in rows_by_track.items():
        rows.sort(key=lambda r: r[0])
        for i in range(len(rows) - SEQUENCE_LENGTH):
            window = rows[i:i + SEQUENCE_LENGTH]
            window_labels = [r[2] for r in window if r[2] is not None]
            if not window_labels:
                continue
            seq_label = max(set(window_labels), key=window_labels.count)  # majority vote
            sequences.append([r[1] for r in window])
            labels.append(seq_label)

    return sequences, labels

def normalize_sequences(sequences):
    """Z-score normalize each feature across the whole dataset."""
    arr = np.array(sequences)  # shape: (num_sequences, seq_len, num_features)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    std[std == 0] = 1.0  # avoid divide-by-zero for constant features (like your placeholders)
    normalized = (arr - mean) / std
    return normalized.tolist(), mean, std

def train(csv_path="labeled_pedestrian_features.csv", epochs=30, lr=0.001):
    sequences, labels = load_labeled_data(csv_path)
    print(f"Loaded {len(sequences)} training sequences.")

    if len(sequences) == 0:
        print("No labeled sequences found — check your CSV has enough labeled rows per pedestrian.")
        return

    sequences, mean, std = normalize_sequences(sequences)
    np.save(os.path.join(script_dir, "feature_mean.npy"), mean) 
    np.save(os.path.join(script_dir, "feature_std.npy"), std)

    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    model = PedestrianIntentLSTM(input_size=FEATURE_COUNT, hidden_size=64)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(script_dir, "pedestrian_intent_lstm.pth"))     
    print("Model saved to pedestrian_intent_lstm.pth")

if __name__ == "__main__":
    train(csv_path="labeled_pedestrian_features_combined.csv")