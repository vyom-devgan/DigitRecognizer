# src/visualize.py

import torch
import matplotlib.pyplot as plt
from models.cnn_model import CNNModel
from src.data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_predictions():
    _, test_loader = get_data_loaders(batch_size=6)
    model = CNNModel()
    model.load_state_dict(torch.load("models/digit_cnn.pth"))
    model.to(device)
    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    images = images.cpu().numpy()

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png")
    print("✅ Saved sample predictions to outputs/sample_predictions.png")

if __name__ == "__main__":
    show_predictions()
