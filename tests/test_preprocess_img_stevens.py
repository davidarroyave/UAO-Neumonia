import sys
import os

# Agregar la carpeta raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import pytest
from src.data import preprocess_img


def test_resize_image_shape():
    # Imagen dummy de 100x100
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = preprocess_img.resize_image(dummy, target_size=(512, 512))
    assert resized.shape[0] == 512
    assert resized.shape[1] == 512


def test_convert_to_grayscale_channels():
    # Imagen dummy en color
    dummy = np.ones((50, 50, 3), dtype=np.uint8) * 255
    gray = preprocess_img.convert_to_grayscale(dummy)
    assert len(gray.shape) == 2  # debe ser 2D si está en escala de grises


def test_apply_clahe_does_not_fail():
    dummy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    clahe_img = preprocess_img.apply_clahe(dummy)
    assert clahe_img.shape == dummy.shape
    assert clahe_img.dtype == np.uint8


def test_normalize_image_range():
    dummy = np.array([[0, 128, 255]], dtype=np.uint8)
    norm = preprocess_img.normalize_image(dummy)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_add_batch_dimension_shape():
    dummy = np.zeros((512, 512), dtype=np.uint8)
    batch = preprocess_img.add_batch_dimension(dummy)
    # Debe ser 4D: (1, H, W, 1)
    assert batch.shape == (1, 512, 512, 1)


def test_preprocess_full_pipeline():
    dummy = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)
    result = preprocess_img.preprocess(dummy)
    # Debe devolver batch 4D con tamaño (1, 512, 512, 1)
    assert result is not None
    assert result.shape == (1, 512, 512, 1)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

