import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig
import matplotlib.pyplot as plt
import shap
import logging
import sys
from io import StringIO

logging.basicConfig(filename='bert_training.log', level=logging.INFO,
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

df = pd.read_csv('/storage/research/data/nids/NSL-KDD/KDDTrain+.csv')
logging.info(f"Dataset shape: {df.shape}")
categorical_columns = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Split features and labels
X = df.drop(columns=['xAttack'])
y = df['xAttack']
le_target = LabelEncoder()
y = le_target.fit_transform(y)
num_classes = len(np.unique(y))

logging.info(f"Number of classes: {num_classes}")
logging.info(f"Unique labels after encoding: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

logging.info(f"Original number of features: {X_train.shape[1]}")
logging.info(f"Number of features after PCA: {X_train_pca.shape[1]}")

train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_features = X_train_pca.shape[1]
num_classes = len(np.unique(y))

print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")

class ImprovedBertForNSLKDD(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_hidden_layers = 12
        config.hidden_size = 512
        config.num_attention_heads = 8
        config.max_position_embeddings = num_features
        config.vocab_size = num_features

        self.bert = BertModel(config)
        self.input_embeddings = nn.Embedding(num_features, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, seq_len = x.shape
        input_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)

        try:
            inputs_embeds = self.input_embeddings(input_ids) * x.unsqueeze(-1)
        except Exception as e:
            raise ValueError(f"Input embedding error: {e}, input shape: {x.shape}, seq_len: {seq_len}")

        outputs = self.bert(inputs_embeds=inputs_embeds)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

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
def train_improved_bert(model, train_loader, test_loader, epochs=50):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate_model(model, test_loader)

        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%")

    return history


@capture_print
def train_model(model, train_loader, test_loader, epochs=20):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        accuracy = evaluate_model(model, test_loader)

        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%")

    return history


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

improved_bert_model = ImprovedBertForNSLKDD(num_features=num_features, num_classes=num_classes)
cnn_lstm_model = CNNLSTM(input_dim=num_features, hidden_dim=128, num_classes=num_classes)
dnn_model = DNN(input_dim=num_features, hidden_dims=[256, 128, 64], num_classes=num_classes)

models = [
    ('Improved BERT', improved_bert_model),
    ('CNN-LSTM', cnn_lstm_model),
    ('DNN', dnn_model)
]

best_model = None
best_model_name = ""
best_accuracy = 0
results = {}
history_dict = {}

for name, model in models:
    logging.info(f"Training {name}")
    history = train_model(model, train_loader, test_loader)
    accuracy = evaluate_model(model, test_loader)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

    history_dict[name] = history
    results[name] = accuracy

model_filename = None
if best_model is not None:
    model_filename = f"{best_model_name.replace(' ', '_')}_alpha.pth"
    torch.save(best_model.state_dict(), model_filename)
    logging.info(f"Best model saved as '{model_filename}' with accuracy: {best_accuracy:.4f}")
else:
    logging.error("No model was found to be the best. Please check the training code.")

plt.figure(figsize=(12, 6))
for name, history in history_dict.items():
    if history:
        plt.plot(history['accuracy'], label=f'{name} Accuracy')
        plt.plot(history['loss'], label=f'{name} Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Accuracy and Loss per Model')
plt.legend()
plt.savefig('bert_training.png')
logging.info("Training progress plot saved.")


#SHAP
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
