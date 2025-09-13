#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción
y la capa convolucional de interés para obtener las características relevantes
de la imagen usando Grad-CAM (Gradient-weighted Class Activation Mapping).
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from src.data.load_model import model_fun
from src.data.preprocess_img import preprocess


def grad_cam(array, conv_layer_name="conv10_thisone"):
    """
    Genera un mapa de calor Grad-CAM para visualizar las regiones importantes
    de la imagen que influyen en la predicción del modelo.
    
    Args:
        array (numpy.ndarray): Imagen original como array numpy
        conv_layer_name (str): Nombre de la capa convolucional para Grad-CAM
        
    Returns:
        numpy.ndarray: Imagen con el mapa de calor superpuesto
    """
    try:
        # 1. Preprocesar la imagen
        img = preprocess(array)
        if img is None:
            print("Error: No se pudo preprocesar la imagen")
            return None
        
        # 2. Cargar el modelo
        model = model_fun()
        if model is None:
            print("Error: No se pudo cargar el modelo")
            return None
        
        # 3. Obtener la predicción
        preds = model.predict(img, verbose=0)
        class_idx = np.argmax(preds[0])
        
        # 4. Crear modelo para extraer características de la capa convolucional
        try:
            last_conv_layer = model.get_layer(conv_layer_name)
        except ValueError:
            print(f"Error: No se encontró la capa '{conv_layer_name}' en el modelo")
            return None
        
        # 5. Crear modelo que retorna tanto las características como las predicciones
        grad_model = tf.keras.models.Model(
            model.inputs, [last_conv_layer.output, model.output]
        )
        
        # 6. Usar GradientTape para calcular gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            # Manejar el caso donde predictions puede ser una lista
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, class_idx]
        
        # 7. Calcular gradientes
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # 8. Calcular la importancia de cada canal
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        
        # 9. Obtener pesos promedio de los gradientes
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        
        # 10. Crear el mapa de activación
        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        
        # Ponderar y sumar
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        
        # 11. Aplicar ReLU y normalizar
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        
        # 12. Redimensionar al tamaño de la imagen original
        cam = cv2.resize(cam, (512, 512))
        
        # 13. Convertir a mapa de calor
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 14. Preparar imagen original
        if len(array.shape) == 2:  # Imagen en escala de grises
            img_original = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        else:
            img_original = array.copy()
        
        img_resized = cv2.resize(img_original, (512, 512))
        
        # 15. Superponer el mapa de calor
        alpha = 0.4  # Transparencia del heatmap
        superimposed_img = cv2.addWeighted(img_resized, 1-alpha, heatmap, alpha, 0)
        
        # 16. Convertir de BGR a RGB si es necesario
        if len(superimposed_img.shape) == 3 and superimposed_img.shape[2] == 3:
            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        return superimposed_img
        
    except Exception as e:
        print(f"Error en grad_cam: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_heatmap(conv_layer_output, pooled_grads):
    """
    Crea el mapa de calor a partir de la salida de la capa convolucional
    y los gradientes ponderados.
    
    Args:
        conv_layer_output (numpy.ndarray): Salida de la capa convolucional
        pooled_grads (numpy.ndarray): Gradientes ponderados
        
    Returns:
        numpy.ndarray: Mapa de calor normalizado
    """
    # Ponderar las características
    for i in range(conv_layer_output.shape[-1]):
        conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # Crear el mapa de calor
    heatmap = np.mean(conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalizar
    
    return heatmap


def apply_colormap_to_heatmap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Aplica un mapa de colores al mapa de calor.
    
    Args:
        heatmap (numpy.ndarray): Mapa de calor normalizado
        colormap: Mapa de colores de OpenCV
        
    Returns:
        numpy.ndarray: Mapa de calor con colores aplicados
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    return cv2.applyColorMap(heatmap_uint8, colormap)


def superimpose_heatmap(original_img, heatmap, alpha=0.8):
    """
    Superpone el mapa de calor sobre la imagen original.
    
    Args:
        original_img (numpy.ndarray): Imagen original
        heatmap (numpy.ndarray): Mapa de calor con colores
        alpha (float): Factor de transparencia del mapa de calor
        
    Returns:
        numpy.ndarray: Imagen con mapa de calor superpuesto
    """
    # Redimensionar imagen original si es necesario
    if original_img.shape[:2] != heatmap.shape[:2]:
        original_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))
    
    # Aplicar transparencia al mapa de calor
    transparency = heatmap * alpha
    transparency = transparency.astype(np.uint8)
    
    # Superponer
    superimposed = cv2.add(transparency, original_img)
    return superimposed.astype(np.uint8)