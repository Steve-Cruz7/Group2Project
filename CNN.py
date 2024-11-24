import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        
        cnn_output_size = 64 * (input_dim // 4)
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Training Function
def train(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy

# Evaluation Function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy, np.array(all_preds), np.array(all_labels)

def plot_metrics(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load Data
    data_file = "./data/synthetic/malware.npz"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    data = np.load(data_file)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Inspect raw data statistics
    print(f"Raw X_train mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
    print(f"Raw X_test mean: {np.mean(X_test):.4f}, std: {np.std(X_test):.4f}")

    # Clip extreme values to handle outliers
    X_train = np.clip(X_train, -10, 10)
    X_test = np.clip(X_test, -10, 10)

    # Normalize Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Log normalized data statistics
    print(f"Normalized X_train mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
    print(f"Normalized X_test mean: {np.mean(X_test):.4f}, std: {np.std(X_test):.4f}")

    # Setup hyperparameters
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 10
    
    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    # Training tracking
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, scheduler)
        test_loss, test_accuracy, preds, labels = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Save best model
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    plot_metrics(train_losses, test_losses, train_accs, test_accs)
    
    cm = confusion_matrix(labels, preds)
    class_names = [f"Class {i}" for i in range(num_classes)]
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    main()
