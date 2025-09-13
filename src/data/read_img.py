#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
read_img.py

Script que lee imágenes en formato DICOM y otros formatos (JPG, PNG) para
visualizarlas en la interfaz gráfica. Además, las convierte a arreglo para
su preprocesamiento posterior.
"""

import pydicom as dicom
import cv2
import numpy as np
from PIL import Image


def read_dicom_file(path):
    """
    Lee un archivo DICOM y lo convierte a formato RGB para procesamiento.
    
    Args:
        path (str): Ruta del archivo DICOM
        
    Returns:
        tuple: (img_RGB, img2show)
            - img_RGB: Imagen en formato RGB como numpy array
            - img2show: Imagen PIL para visualización
    """
    try:
        img = dicom.dcmread(path)
        img_array = img.pixel_array
        img2show = Image.fromarray(img_array)
        
        # Normalizar la imagen
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        
        # Convertir a RGB
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        return img_RGB, img2show
        
    except Exception as e:
        print(f"Error al leer archivo DICOM: {e}")
        return None, None


def read_jpg_file(path):
    """
    Lee un archivo de imagen en formato JPG/PNG y lo procesa.
    
    Args:
        path (str): Ruta del archivo de imagen
        
    Returns:
        tuple: (img_processed, img2show)
            - img_processed: Imagen procesada como numpy array
            - img2show: Imagen PIL para visualización
    """
    try:
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img2show = Image.fromarray(img_array)
        
        # Normalizar la imagen
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        
        return img2, img2show
        
    except Exception as e:
        print(f"Error al leer archivo de imagen: {e}")
        return None, None


def read_image_file(path):
    """
    Función genérica que detecta el tipo de archivo y llama a la función apropiada.
    
    Args:
        path (str): Ruta del archivo de imagen
        
    Returns:
        tuple: (img_processed, img2show)
    """
    file_extension = path.lower().split('.')[-1]
    
    if file_extension == 'dcm':
        return read_dicom_file(path)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        return read_jpg_file(path)
    else:
        print(f"Formato de archivo no soportado: {file_extension}")
        return None, None