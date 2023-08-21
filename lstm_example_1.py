import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Dataset preparation
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Generate synthetic data
data = torch.randn(100, 10)  # 100 samples, 10 features per sample
labels = torch.randint(2, (100,))  # Binary labels

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

# Step 2: Dataloader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Step 3: Module class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Step 4: Training class
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, model_path, patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_path = model_path
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train(self, epochs):
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")

            # Evaluate on validation set and save the best model
            val_loss = self.evaluate()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print("Saved best model!")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered!")
                    break
            
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Step 5: Evaluation class
class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            all_labels = []
            all_preds = []
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                total_loss += loss.item()
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

            avg_loss = total_loss / len(self.val_loader)
            print(f"Validation Loss: {avg_loss:.4f}")
            return avg_loss

# Step 6: Prediction class
class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

# Step 7: Main execution
input_size = 10
hidden_size = 32
num_layers = 2
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
trainer.train(epochs=10)

# Validation
trainer.evaluate()

# Testing
evaluator = Evaluator(model, test_loader, device)
evaluator.evaluate()

# Predicting
sample_input = torch.randn(1, 5, input_size).to(device)
predictor = Predictor(model, device)
predicted_class = predictor.predict(sample_input)
print(f"Predicted class: {predicted_class}")
