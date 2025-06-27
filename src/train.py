# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import CNNModel
from src.data_loader import get_data_loaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs=5, lr=0.001, batch_size=64):
    train_loader, test_loader = get_data_loaders(batch_size)
    model = CNNModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/digit_cnn.pth")
    print("✅ Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
