#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
load_model.py

Script que lee el archivo binario del modelo de red neuronal convolucional
previamente entrenado llamado 'conv_MLP_84.h5'.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import os


def model_fun():
    """
    Carga el modelo de red neuronal convolucional entrenado.
    
    Returns:
        tensorflow.keras.Model: Modelo cargado listo para predicción
    """
    try:
        model_path = 'src/models/conv_MLP_84.h5'
        
        # Verificar que el archivo del modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo del modelo '{model_path}' no fue encontrado")
        
        # Cargar el modelo sin compilar para evitar problemas de compatibilidad
        model = load_model(model_path, compile=False)
        
        # Recompilar con configuración compatible con TensorFlow 2.20.0
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Modelo '{model_path}' cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


def load_custom_model(model_path):
    """
    Carga un modelo desde una ruta específica.
    
    Args:
        model_path (str): Ruta al archivo del modelo
        
    Returns:
        tensorflow.keras.Model: Modelo cargado o None si hay error
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo del modelo '{model_path}' no fue encontrado")
        
        # Cargar el modelo sin compilar para evitar problemas de compatibilidad
        model = load_model(model_path, compile=False)
        
        # Recompilar con configuración compatible con TensorFlow 2.20.0
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Modelo '{model_path}' cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"Error al cargar el modelo desde '{model_path}': {e}")
        return None


def get_model_info(model):
    """
    Obtiene información básica del modelo cargado.
    
    Args:
        model (tensorflow.keras.Model): Modelo de TensorFlow/Keras
        
    Returns:
        dict: Diccionario con información del modelo
    """
    if model is None:
        return None
    
    try:
        info = {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'num_layers': len(model.layers),
            'num_parameters': model.count_params(),
            'layer_names': [layer.name for layer in model.layers]
        }
        return info
        
    except Exception as e:
        print(f"Error al obtener información del modelo: {e}")
        return None


def verify_model_architecture(model, expected_conv_layer='conv10_thisone'):
    """
    Verifica que el modelo tenga la arquitectura esperada para Grad-CAM.
    
    Args:
        model (tensorflow.keras.Model): Modelo a verificar
        expected_conv_layer (str): Nombre de la capa convolucional esperada
        
    Returns:
        bool: True si la arquitectura es correcta, False en caso contrario
    """
    if model is None:
        return False
    
    try:
        # Verificar que existe la capa convolucional necesaria para Grad-CAM
        layer_names = [layer.name for layer in model.layers]
        
        if expected_conv_layer not in layer_names:
            print(f"Advertencia: La capa '{expected_conv_layer}' no fue encontrada en el modelo")
            print(f"Capas disponibles: {layer_names}")
            return False
        
        print(f"Modelo verificado: contiene la capa '{expected_conv_layer}'")
        return True
        
    except Exception as e:
        print(f"Error al verificar la arquitectura del modelo: {e}")
        return False