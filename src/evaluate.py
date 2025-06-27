# src/evaluate.py

import torch
from models.cnn_model import CNNModel
from src.data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path="models/digit_cnn.pth"):
    _, test_loader = get_data_loaders()
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
