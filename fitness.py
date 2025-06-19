import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def bin_calories(y):
    return pd.cut(y, bins=[0, 200, 350, 1000], labels=[0, 1, 2]).astype(int)


def load_data_from_csv(path='fitness_data.csv'):
    df = pd.read_csv(path)
    y_class = bin_calories(df['calories_burned'])

    if y_class.isnull().any():
        df = df[~y_class.isnull()]
        y_class = bin_calories(df['calories_burned'])  # Re-bin clean data

    X = df.drop(columns=['calories_burned']).values
    y = y_class.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split to preserve class balance
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )


class FitnessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_model(input_size=5, num_classes=3):
    return nn.Sequential(
        nn.Linear(input_size, 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, num_classes)
    )


def train_model(model, dataloader, val_loader=None, epochs=15, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Optional: print validation accuracy
        if val_loader:
            acc = evaluate_model(model, val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.2%}")


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total


def save_model(model, path='fitness_model_class.pth'):
    torch.save(model.state_dict(), path)


def load_model(path='fitness_model_class.pth', input_size=5, num_classes=3):
    model = build_model(input_size, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data_from_csv('fitness_data.csv')

    train_dataset = FitnessDataset(X_train, y_train)
    test_dataset = FitnessDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = build_model(input_size=X_train.shape[1])
    train_model(model, train_loader, val_loader=test_loader, epochs=15)

    accuracy = evaluate_model(model, test_loader)
    print(f"🎯 Final Test Accuracy: {accuracy:.2%}")

    save_model(model, 'fitness_model_class.pth')
    print("💾 Model saved to 'fitness_model_class.pth'")
