# test/test_model.py

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
from models.cnn_model import CNNModel

def test_model_output_shape():
    model = CNNModel()
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    assert output.shape == (1, 10)
