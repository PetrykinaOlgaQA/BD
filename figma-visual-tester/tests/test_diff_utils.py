from PIL import Image

from src.diff_utils import create_diff_map


def test_create_diff_map_shape():
    a = Image.new("RGB", (100, 80), color=(200, 10, 10))
    b = Image.new("RGB", (120, 90), color=(200, 200, 10))
    r = create_diff_map(a, b, target_size=(64, 64))
    assert r.diff_gray_64.shape == (64, 64)
    assert r.canvas_size[0] >= 120
