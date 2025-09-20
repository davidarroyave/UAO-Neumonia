#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_docker_no_gui.py
Script de prueba para validar el funcionamiento del contenedor Docker
sin dependencias de GUI.
"""

import numpy as np
import sys
import os

def test_imports():
    """Prueba la importación de todos los módulos críticos."""
    print("🧪 Probando importaciones...")
    
    tests = [
        ("TensorFlow", lambda: __import__('tensorflow')),
        ("OpenCV", lambda: __import__('cv2')),
        ("NumPy", lambda: __import__('numpy')),
        ("Pandas", lambda: __import__('pandas')),
        ("Pillow", lambda: __import__('PIL')),
        ("Matplotlib", lambda: __import__('matplotlib')),
        ("Scikit-learn", lambda: __import__('sklearn')),
        ("SciPy", lambda: __import__('scipy')),
    ]
    
    results = []
    for name, import_func in tests:
        try:
            module = import_func()
            version = getattr(module, '__version__', 'N/A')
            print(f"   ✅ {name}: {version}")
            results.append(True)
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            results.append(False)
    
    return all(results)

def test_project_modules():
    """Prueba la importación de módulos del proyecto."""
    print("\n🧪 Probando módulos del proyecto...")
    
    tests = [
        ("read_img", "src.data.read_img", "read_image_file"),
        ("preprocess_img", "src.data.preprocess_img", "preprocess_image"),
        ("integrator", "src.data.integrator", "predict"),
        ("detector_neumonia", "src.data.detector_neumonia", None),
        ("load_model", "src.data.load_model", None),
        ("grad_cam", "src.data.grad_cam", None),
    ]
    
    results = []
    for name, module_path, function_name in tests:
        try:
            module = __import__(module_path, fromlist=[function_name] if function_name else [''])
            if function_name:
                getattr(module, function_name)
            print(f"   ✅ {name}")
            results.append(True)
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            results.append(False)
        except AttributeError as e:
            print(f"   ❌ {name}: {e}")
            results.append(False)
    
    return all(results)

def test_tensorflow_basic():
    """Prueba básica de TensorFlow."""
    print("\n🧪 Probando TensorFlow básico...")
    
    try:
        import tensorflow as tf
        
        # Crear un tensor simple
        a = tf.constant([1, 2, 3], dtype=tf.float32)
        b = tf.constant([4, 5, 6], dtype=tf.float32)
        c = tf.add(a, b)
        
        print(f"   ✅ Operación TensorFlow: {c.numpy()}")
        print(f"   ✅ GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
        print(f"   ✅ CPU threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
        return True
    except Exception as e:
        print(f"   ❌ Error en TensorFlow: {e}")
        return False

def test_opencv_basic():
    """Prueba básica de OpenCV."""
    print("\n🧪 Probando OpenCV básico...")
    
    try:
        import cv2
        import numpy as np
        
        # Crear una imagen de prueba
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [255, 255, 255]  # Cuadrado blanco
        
        # Aplicar un filtro
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        
        print(f"   ✅ Imagen creada: {img.shape}")
        print(f"   ✅ Filtro aplicado: {blurred.shape}")
        print(f"   ✅ OpenCV build info disponible: {len(cv2.getBuildInformation()) > 0}")
        return True
    except Exception as e:
        print(f"   ❌ Error en OpenCV: {e}")
        return False

def test_environment():
    """Prueba las variables de entorno."""
    print("\n🧪 Probando variables de entorno...")
    
    env_vars = {
        "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'No definida'),
        "PYTHONDONTWRITEBYTECODE": os.environ.get('PYTHONDONTWRITEBYTECODE', 'No definida'),
        "PYTHONPATH": os.environ.get('PYTHONPATH', 'No definida'),
        "DISPLAY": os.environ.get('DISPLAY', 'No definida'),
        "USER": os.environ.get('USER', 'No definido'),
    }
    
    for var, value in env_vars.items():
        print(f"   ✅ {var}: {value}")
    
    print(f"   ✅ Working Directory: {os.getcwd()}")
    print(f"   ✅ Python Executable: {sys.executable}")
    print(f"   ✅ Python Version: {sys.version}")
    
    return True

def main():
    """Función principal de prueba."""
    print("🐳 VALIDACIÓN COMPLETA DEL CONTENEDOR DOCKER")
    print("=" * 50)
    print(f"🕐 Timestamp: {__import__('datetime').datetime.now()}")
    print("=" * 50)
    
    tests = [
        ("📦 Importaciones de librerías", test_imports),
        ("🔧 Módulos del proyecto", test_project_modules),
        ("🤖 TensorFlow básico", test_tensorflow_basic),
        ("📷 OpenCV básico", test_opencv_basic),
        ("🌍 Variables de entorno", test_environment),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            print(f"\n   {status}")
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 TODAS LAS PRUEBAS PASARON ({passed}/{total})")
        print("✅ El contenedor Docker está COMPLETAMENTE FUNCIONAL!")
        print("🚀 Listo para ejecutar la aplicación con GUI usando Xming")
        return 0
    else:
        print(f"⚠️  ALGUNAS PRUEBAS FALLARON ({passed}/{total})")
        print("🔧 Revisa los errores mostrados arriba")
        return 1

if __name__ == "__main__":
    sys.exit(main())