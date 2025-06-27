# 🧠 Digit Recognizer

A handwritten digit recognition system built using PyTorch and deployed with Streamlit.

## 🔍 Features
- CNN model trained on MNIST dataset
- Real-time digit prediction via Streamlit web app
- CLI support via argparse
- Unit tested with Pytest
- Modular structure with `src/`, `models/`, and `tests/`

## 🚀 Demo
Run the Streamlit app:

```bash
streamlit run app.py
```

## 📦 Installation
Clone the repository:

```bash
git clone git@github.com:vyom-devgan/DigitRecognizer.git
cd DigitRecognizer
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install torch streamlit pytest
```

Train the model (if needed):

```bash
python src/train.py
```

## 🧪 Run Tests

```bash
pytest tests/
```

## 📁 Project Structure

```
DigitRecognizer/
│
├── models/
│   └── cnn_model.py
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── data_loader.py
├── tests/
│   └── test_model.py
├── app.py
├── README.md
├── LICENSE
└── .gitignore
```
