#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
integrator.py

Módulo que integra los demás scripts y retorna solamente lo necesario para ser
visualizado en la interfaz gráfica. Retorna la clase, la probabilidad y una
imagen del mapa de calor generado por Grad-CAM.
"""

import numpy as np
from src.data.load_model import model_fun
from src.data.preprocess_img import preprocess
from src.data.grad_cam import grad_cam


def predict(array):
    """
    Función principal que integra todos los módulos para realizar la predicción
    completa de neumonía en una imagen radiográfica.
    
    Args:
        array (numpy.ndarray): Imagen como array numpy
        
    Returns:
        tuple: (label, proba, heatmap)
            - label (str): Etiqueta de la predicción ('bacteriana', 'normal', 'viral')
            - proba (float): Probabilidad de la predicción (0-100)
            - heatmap (numpy.ndarray): Imagen con mapa de calor Grad-CAM superpuesto
    """
    try:
        # 1. Preprocesar la imagen
        batch_array_img = preprocess(array)
        if batch_array_img is None:
            return None, 0, None
        
        # 2. Cargar el modelo y realizar predicción
        model = model_fun()
        if model is None:
            return None, 0, None
        
        # Obtener predicción
        predictions = model.predict(batch_array_img)
        prediction_index = np.argmax(predictions)
        proba = np.max(predictions) * 100
        
        # 3. Convertir índice a etiqueta
        label = get_class_label(prediction_index)
        
        # 4. Generar mapa de calor Grad-CAM
        heatmap = grad_cam(array)
        
        return label, proba, heatmap
        
    except Exception as e:
        print(f"Error en predict: {e}")
        return None, 0, None


def get_class_label(prediction_index):
    """
    Convierte el índice de predicción a etiqueta de clase.
    
    Args:
        prediction_index (int): Índice de la clase predicha
        
    Returns:
        str: Etiqueta de la clase
    """
    class_labels = {
        0: "bacteriana",
        1: "normal", 
        2: "viral"
    }
    
    return class_labels.get(prediction_index, "desconocido")


def predict_single_image(image_array):
    """
    Realiza predicción en una sola imagen.
    
    Args:
        image_array (numpy.ndarray): Array de la imagen
        
    Returns:
        dict: Diccionario con resultados de la predicción
    """
    try:
        label, proba, heatmap = predict(image_array)
        
        result = {
            'label': label,
            'probability': proba,
            'heatmap': heatmap,
            'success': True if label is not None else False
        }
        
        return result
        
    except Exception as e:
        return {
            'label': None,
            'probability': 0,
            'heatmap': None,
            'success': False,
            'error': str(e)
        }


def get_prediction_confidence(probability):
    """
    Determina el nivel de confianza de la predicción.
    
    Args:
        probability (float): Probabilidad de la predicción
        
    Returns:
        str: Nivel de confianza ('Alta', 'Media', 'Baja')
    """
    if probability >= 80:
        return "Alta"
    elif probability >= 60:
        return "Media"
    else:
        return "Baja"


def format_prediction_result(label, probability, confidence_level=None):
    """
    Formatea el resultado de la predicción para mostrar en la interfaz.
    
    Args:
        label (str): Etiqueta de la clase
        probability (float): Probabilidad de la predicción
        confidence_level (str): Nivel de confianza (opcional)
        
    Returns:
        dict: Resultado formateado
    """
    if confidence_level is None:
        confidence_level = get_prediction_confidence(probability)
    
    return {
        'diagnosis': label.capitalize() if label else "No disponible",
        'probability': f"{probability:.2f}%",
        'confidence': confidence_level,
        'raw_probability': probability
    }


def validate_prediction_inputs(image_array):
    """
    Valida que los inputs para la predicción sean correctos.
    
    Args:
        image_array (numpy.ndarray): Array de la imagen
        
    Returns:
        bool: True si los inputs son válidos, False en caso contrario
    """
    if image_array is None:
        print("Error: Array de imagen es None")
        return False
    
    if not isinstance(image_array, np.ndarray):
        print("Error: El input debe ser un numpy array")
        return False
    
    if len(image_array.shape) < 2:
        print("Error: La imagen debe tener al menos 2 dimensiones")
        return False
    
    return True


def batch_predict(image_arrays):
    """
    Realiza predicciones en lote para múltiples imágenes.
    
    Args:
        image_arrays (list): Lista de arrays de imágenes
        
    Returns:
        list: Lista de resultados de predicción
    """
    results = []
    
    for i, image_array in enumerate(image_arrays):
        try:
            result = predict_single_image(image_array)
            result['image_index'] = i
            results.append(result)
            
        except Exception as e:
            results.append({
                'image_index': i,
                'success': False,
                'error': str(e)
            })
    
    return results