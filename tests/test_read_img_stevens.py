import sys
import os

# Agregar la carpeta raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image
import cv2
import pytest

from src.data import read_img


def test_read_jpg_file_with_dummy(tmp_path):
    # Crear imagen dummy y guardarla como JPG
    dummy_array = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(dummy_path), dummy_array)

    img_processed, img2show = read_img.read_jpg_file(str(dummy_path))

    # Verificar que la imagen procesada no sea None
    assert img_processed is not None
    assert isinstance(img2show, Image.Image)
    assert img_processed.shape[0] == 10
    assert img_processed.shape[1] == 10


def test_read_image_file_invalid_extension(tmp_path):
    # Crear archivo con extensión no soportada
    invalid_path = tmp_path / "archivo.txt"
    invalid_path.write_text("contenido de prueba")

    img_processed, img2show = read_img.read_image_file(str(invalid_path))

    # La función debería devolver (None, None)
    assert img_processed is None
    assert img2show is None
