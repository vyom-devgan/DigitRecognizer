# app.py

import streamlit as st
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from models.cnn_model import CNNModel

# Load model
model = CNNModel()
model.load_state_dict(torch.load("models/digit_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🖼️ Digit Recognizer")
st.write("Upload a 28x28 handwritten digit image (white digit on black background).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
        st.success(f"🎯 Predicted Digit: **{predicted}**")
