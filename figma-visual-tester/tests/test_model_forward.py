import torch

from src.model import TinyDiffCNN


def test_tiny_diff_cnn_forward():
    m = TinyDiffCNN()
    m.eval()
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 2)
