import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import logging
import sys
from io import StringIO

logging.basicConfig(filename='cnn_lstm_dnn.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def capture_print(func):
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            func(*args, **kwargs)
            output = sys.stdout.getvalue()
            logging.info(output)
        finally:
            sys.stdout = old_stdout
    return wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load and preprocess data
df = pd.read_csv('/storage/research/data/nids/NSL-KDD/KDDTrain+.csv')
categorical_columns = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Split features and labels
X = df.drop(columns=['xAttack'])
y = df['xAttack']
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_features = X_train_pca.shape[1]
num_classes = len(np.unique(y))

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

@capture_print
def train_model(model, train_loader, test_loader, epochs=20):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    accuracy = evaluate_model(model, test_loader)
    return accuracy

def evaluate_model(model, test_loader):
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
    return 100 * correct / total

cnn_lstm_model = CNNLSTM(input_dim=num_features, hidden_dim=128, num_classes=num_classes)
dnn_model = DNN(input_dim=num_features, hidden_dims=[256, 128, 64], num_classes=num_classes)

logging.info("Training CNN-LSTM")
train_model(cnn_lstm_model, train_loader, test_loader)
logging.info("Training DNN")
train_model(dnn_model, train_loader, test_loader)

cnn_lstm_model.to(device)
batch = next(iter(test_loader))[0].to(device)

torch.backends.cudnn.enabled = False
cnn_lstm_model.train()
explainer_cnn_lstm = shap.GradientExplainer(cnn_lstm_model, batch)
shap_values_cnn_lstm = explainer_cnn_lstm.shap_values(batch)

shap.summary_plot(shap_values_cnn_lstm, batch.cpu().numpy(), feature_names=[f'Feature {i}' for i in range(num_features)])
plt.savefig('shap_summary_cnn_lstm.png')
cnn_lstm_model.eval()
torch.backends.cudnn.enabled = True

dnn_model.to(device)

torch.backends.cudnn.enabled = False
dnn_model.train()
explainer_dnn = shap.GradientExplainer(dnn_model, batch)
shap_values_dnn = explainer_dnn.shap_values(batch)

shap.summary_plot(shap_values_dnn, batch.cpu().numpy(), feature_names=[f'Feature {i}' for i in range(num_features)])
plt.savefig('shap_summary_dnn.png')
dnn_model.eval()
torch.backends.cudnn.enabled = True

logging.info("SHAP analysis completed for CNN-LSTM and DNN.")
