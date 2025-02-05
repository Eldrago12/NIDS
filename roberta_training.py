import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel, RobertaConfig
import matplotlib.pyplot as plt

# Load the NSL-KDD dataset
df = pd.read_csv('/storage/research/data/nids/NSL-KDD/KDDTrain+.csv')

# Basic preprocessing
categorical_columns = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Split features and labels
X = df.drop(columns=['xAttack'])
y = df['xAttack']

# Encode the target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_features = X_train.shape[1]
num_classes = len(np.unique(y))

print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")

# Modified RoBERTa model
class ModifiedRobertaForNSLKDD(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_hidden_layers = 6
        config.hidden_size = 256
        config.num_attention_heads = 8
        config.max_position_embeddings = num_features
        config.vocab_size = num_features

        self.roberta = RobertaModel(config)
        self.input_layer = nn.Linear(num_features, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        inputs_embeds = self.input_layer(x).unsqueeze(1)
        outputs = self.roberta(inputs_embeds=inputs_embeds)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)

# CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# DNN model
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, test_loader, epochs=10):
    device = torch.device('cpu')  # Use CPU instead of GPU
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}, Accuracy: {100 * correct / total:.2f}%')

# Initialize and train models
roberta_model = ModifiedRobertaForNSLKDD(num_features, num_classes)
cnn_lstm_model = CNNLSTM(num_features, hidden_dim=128, num_classes=num_classes)
dnn_model = DNN(num_features, hidden_dims=[256, 128, 64], num_classes=num_classes)

models = [
    ('Modified RoBERTa', roberta_model),
    ('CNN-LSTM', cnn_lstm_model),
    ('DNN', dnn_model)
]

results = {}

for name, model in models:
    print(f"\nTraining {name} model:")
    train_model(model, train_loader, test_loader)

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    results[name] = accuracy

# Plot results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, (model, accuracy) in enumerate(results.items()):
    plt.text(i, accuracy, f'{accuracy:.4f}', ha='center', va='bottom')
plt.show()
