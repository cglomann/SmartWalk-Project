import torch
import torch.nn as nn

class PedestrianIntentLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return torch.sigmoid(logits)