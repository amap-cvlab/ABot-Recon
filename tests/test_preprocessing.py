import numpy as np
import torch

from abot_recon.preprocessing import preprocess_image


def test_landscape_image_uses_mean_padding():
    image = np.zeros((100, 400, 3), dtype=np.uint8)
    tensor, transform = preprocess_image(image)
    assert tensor.shape == (3, 280, 504)
    assert transform.resized_height == 126
    assert transform.pad_top == 77
    assert transform.pad_bottom == 77
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([0.485, 0.456, 0.406]))


def test_portrait_image_uses_center_crop():
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    tensor, transform = preprocess_image(image)
    assert tensor.shape == (3, 280, 504)
    assert transform.crop_top == transform.crop_bottom == 364


def test_bicubic_boundary_values_are_not_clamped():
    image = np.zeros((376, 1241, 3), dtype=np.uint8)
    image[:, 600:] = 255
    tensor, _ = preprocess_image(image)
    assert float(tensor.min()) < 0.0 and float(tensor.max()) > 1.0
