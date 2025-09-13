#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess_img.py

Script que recibe el arreglo proveniente de read_img.py y realiza las siguientes
modificaciones:
- Resize a 512x512
- Conversión a escala de grises
- Ecualización del histograma con CLAHE
- Normalización de la imagen entre 0 y 1
- Conversión del arreglo de imagen a formato de batch (tensor)
"""

import cv2
import numpy as np


def preprocess(array):
    """
    Preprocesa una imagen para el modelo de red neuronal.
    
    Args:
        array (numpy.ndarray): Imagen como array numpy
        
    Returns:
        numpy.ndarray: Imagen preprocesada en formato batch
    """
    try:
        # 1. Resize a 512x512
        array = cv2.resize(array, (512, 512))
        
        # 2. Conversión a escala de grises
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        
        # 3. Ecualización del histograma con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        
        # 4. Normalización entre 0 y 1
        array = array / 255.0
        
        # 5. Conversión a formato de batch (tensor)
        # Agregar dimensión de canal
        array = np.expand_dims(array, axis=-1)
        # Agregar dimensión de batch
        array = np.expand_dims(array, axis=0)
        
        return array
        
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
        return None


def resize_image(array, target_size=(512, 512)):
    """
    Redimensiona una imagen al tamaño objetivo.
    
    Args:
        array (numpy.ndarray): Imagen como array numpy
        target_size (tuple): Tamaño objetivo (ancho, alto)
        
    Returns:
        numpy.ndarray: Imagen redimensionada
    """
    return cv2.resize(array, target_size)


def convert_to_grayscale(array):
    """
    Convierte una imagen a escala de grises.
    
    Args:
        array (numpy.ndarray): Imagen como array numpy
        
    Returns:
        numpy.ndarray: Imagen en escala de grises
    """
    if len(array.shape) == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array


def apply_clahe(array, clip_limit=2.0, tile_grid_size=(4, 4)):
    """
    Aplica ecualización adaptativa del histograma (CLAHE).
    
    Args:
        array (numpy.ndarray): Imagen en escala de grises
        clip_limit (float): Límite de recorte para CLAHE
        tile_grid_size (tuple): Tamaño de la cuadrícula de azulejos
        
    Returns:
        numpy.ndarray: Imagen con CLAHE aplicado
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(array)


def normalize_image(array):
    """
    Normaliza una imagen entre 0 y 1.
    
    Args:
        array (numpy.ndarray): Imagen como array numpy
        
    Returns:
        numpy.ndarray: Imagen normalizada
    """
    return array / 255.0


def add_batch_dimension(array):
    """
    Convierte una imagen a formato de batch para el modelo.
    
    Args:
        array (numpy.ndarray): Imagen como array numpy
        
    Returns:
        numpy.ndarray: Imagen en formato batch
    """
    # Agregar dimensión de canal si no existe
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)
    
    # Agregar dimensión de batch
    array = np.expand_dims(array, axis=0)
    
    return array