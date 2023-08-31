import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)  # Leaky ReLU activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.batch_norm2 = nn.BatchNorm1d(128)  # Batch normalization
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.batch_norm1(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.leaky_relu(out)
        out = self.fc1(out)
        out = self.batch_norm2(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.fc3(out)
        return out

# Instantiate the model
input_size = num_features
hidden_size = 64
num_layers = 4
dropout_prob = 0.2
model = LSTMModel(input_size, hidden_size, num_layers, dropout_prob)
