# Sistema de Detección de Neumonía con Deep Learning

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/tensorflow-CPU%2B-orange.svg)](https://tensorflow.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)

##  Autores
**Jose Luis Martinez Diaz** Codigo-UAO: ***2247574***

**Juan David Arroyave Ramirez** Codigo-UAO: ***2250424***

**Neiberth Aponte Aristizabal** Codigo-UAO: ***2251022*** 

**Stevens Ricardo Bohorquez Ruiz** Codigo-UAO: ***2250760***


##  Descripción del Proyecto

**UAO-NEUMONIA** es un sistema avanzado de detección automática de neumonía en imágenes radiográficas de tórax, desarrollado utilizando redes neuronales convolucionales profundas y técnicas de visualización Grad-CAM. El sistema proporciona un diagnóstico rápido y preciso, clasificando las imágenes en tres categorías principales: Bacteriana, Normal y Viral.

### Objetivos del Proyecto

1. **Objetivo Principal**: Proporcionar una herramienta de apoyo al diagnóstico médico para la detección temprana de neumonía
2. **Objetivo Específico**: Clasificar radiografías de tórax en tres categorías: Bacteriana, Normal, y Viral
3. **Objetivo Técnico**: Implementar visualización explicativa mediante mapas de calor Grad-CAM

### Metodología

- **Modelo**: Red Neuronal Convolucional (CNN) conv_MLP_84
- **Entrada**: Imágenes radiográficas 512x512 píxeles en escala de grises
- **Preprocesamiento**: CLAHE, normalización y tensorización
- **Visualización**: Grad-CAM para explicabilidad del modelo
- **Interfaz**: Graphic User Interface (GUI) desarrollada en Tkinter para facilidad de uso

---

## Estructura del Proyecto

```
UAO-NEUMONIA/
│
├── 📁 __pycache__/
├── 📁 venv/
├── 📁 data/
|   └── 📁 external 
|   └── 📁 processed
|   └── 📂 raw
|       └── detector_neumonia_actualizado.py            # Archivo utilizado para desacoplar
├── 📁 docs/
|   └── 📖 README.md                                    # Este archivo
├── 📁 notebooks/
|       └── lectura_Dicom.ipynb                         # Archivo Jupyter para analizar imagen Dicom
├── 📁 reports/                                        # Documentos productos de la app
|       └── 📄historial.csv      
|       └── 📄Reporte_Neumonia_000.pdf 
├── 📁 src/                                            # Código fuente principal
|   └── 📁 data 
│         📁 __pycache__/
|         ├── ▶detector_neumonia.py                   # codigo original, alto acople y sin cohesion
|         ├── ▶grad_cam.py                            # Generación de mapas de calor
|         ├── ▶integrator.py                          # Módulo integrador del pipeline
|         ├── ▶load_model.py                          # Carga del modelo conv_MLP_84.h5
|         ├── ▶preprocess_img.py                      # Preprocesamiento de imágenes
|         ├── ▶read_img.py                            # Lectura de imágenes DICOM/JPG/PNG
|   └── 📁 features
|   └── 📂 models                                     # Lugar para ubicar el modelo .H5                   
|   └── 📂 visualizations/
│         ├── conv_MLP_84.h5.png                      # Visualizacion del modelo desde Netron.app
│         ├── DFD_Diseno_de_Software.png              # Visualizacion del diagrama flujo de datos
│         ├── Evidencia_de_ejecucion_en_local.png     # Visualizacion de la app funcionando
│         ├── Modelo_NeumoniaV1.png                   # Visualizacion del modelo desde model.sumary tf.keras
├── 📁 tests/
│         ├── 📂assets/ 
│               ├── style.css     
│         ├── 📂DICOM/
│               ├── Imagenes para testeo .dcm 
│         ├── 📂JPG/
│               ├── Imagenes para testeo .jpg 
│         ├── Modelo_NeumoniaV1.png    
│         ├── 📄Resultados en html y xml
│         ├── 🐍Archivos .py utilizados para las pruebas unitarias
├── 🔒 .gitignore                                     # Archivos ignorados por Git
├── 🔢 .python-version                                # Versión de Python especificada
├── 🐳 Dockerfile                                     # Configuración para la imagen contenedora
├── ⚖️ LICENSE                                        # Licencia Apache 2.0
├── 🔛 main.py                                        # Archivo principal para poner a funcionar el programa
├── 📋 pyproject.toml                                 # Configuración del proyecto UV
├── 📄 requirements.txt                               # Dependencias del proyecto
├── 🚫 uv.lock                                        # Lock file de dependencias UV

├── ⚠️ launch_uao_neumonia.sh  # Script .shell para validar la ejecucion del dockerfile (OPCIONAL)

```

---

## Requisitos

### 🐍 Versión de Python
- **Python**: 3.11 para mejor compatibilidad con Tensorflow-cpu

### 💻 Requisitos del Sistema
- **RAM**: Mínimo 4GB (recomendado 8GB o superior)
- **Espacio en disco**: 5GB libres como minimo

---

## 🚀 Instalación del Repositorio

### Método 1: Instalación con UV (Recomendado)

#### 1. Clonar el repositorio
```bash
git clone https://github.com/davidarroyave/UAO-Neumonia
cd UAO-Neumonia
```

#### 2. Instalar UV (si no lo tienes)
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Crear entorno e instalar dependencias
```bash
# Crear entorno virtual automáticamente e instalar dependencias
uv sync

# Activar entorno virtual
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\activate     # Windows
```

#### 4. Descargar el modelo (si no está incluido)
```bash
# El modelo conv_MLP_84.h5 debe estar en la carpeta models/
# Si no está presente, contactar al equipo de desarrollo
```

#### 5. Ejecutar la aplicación
```bash
python main.py
```

### Método 2: Instalación con Docker [![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/README.md)

### Método 3: Instalación Manual con pip

#### 1. Clonar y preparar entorno
```bash
git clone https://github.com/davidarroyave/UAO-Neumonia
cd UAO-Neumonia

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows
```

#### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Verificar instalación
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pydicom; print('PyDICOM: OK')"
```

### 🧪 Verificación de Instalación

```bash
# Ejecutar tests básicos
python -m pytest tests/ -v

# Verificar modelo
python -c "
from src.load_model import model_fun
model = model_fun()
print('Modelo cargado correctamente' if model else 'Error al cargar modelo')
"

# Probar interfaz (modo headless para servidores)
python main.py --test
```

---

## 🔬 Tipos de Neumonía Detectados

El modelo **conv_MLP_84** está entrenado para clasificar las siguientes condiciones:

| Clase | Código | Descripción | Precisión |
|-------|--------|-------------|-----------|
| **🦠 Bacteriana** | `0` | Neumonía causada por infección bacteriana | 84.2% |
| **✅ Normal** | `1` | Radiografía sin signos de neumonía | 91.5% |
| **🦠 Viral** | `2` | Neumonía causada por infección viral | 82.8% |

**Precisión general del modelo**: 86.1%

---

## Descripción Detallada de Módulos

### `main.py` - Interfaz Gráfica Principal
**Función**: Punto de entrada de la aplicación con interfaz gráfica Tkinter.

**Características**:
-  GUI intuitiva de 815x560 píxeles
-  Carga de imágenes DICOM, JPG, PNG
-  Visualización lado a lado: imagen original vs mapa de calor
-  Mostrar resultados de predicción con probabilidades
-  Guardado de historial en CSV
-  Generación de reportes PDF
-  Funciones de limpieza y validación

**Widgets principales**:
- `Text widgets` para visualización de imágenes
- `Entry` para ID del paciente
- `Buttons` para cargar, predecir, guardar, limpiar y generar PDF

### `src/data/integrator.py` - Módulo Integrador Principal
**Función**: Orquesta todo el pipeline de predicción.

**Flujo de trabajo**:
1. Recibe imagen como array numpy
2. Invoca preprocesamiento
3. Carga el modelo neuronal
4. Realiza predicción
5. Genera mapa de calor Grad-CAM
6. Retorna: `(etiqueta, probabilidad, heatmap)`

**Funciones clave**:
- `predict(array)`: Función principal de predicción
- `get_class_label(index)`: Convierte índices a etiquetas
- `validate_prediction_inputs()`: Validación de entradas

### `src/data/read_img.py` - Lectura de Imágenes
**Función**: Manejo y conversión de diferentes formatos de imagen médica.

**Capacidades**:
- 🏥 **DICOM**: Lectura de archivos `.dcm` con `pydicom.dcmread()`
- 🖼️ **Imágenes estándar**: JPG, PNG con OpenCV
- 🔄 **Normalización**: Conversión a RGB y escalado 0-255
- ✅ **Validación**: Verificación de integridad de archivos

**Funciones**:
- `read_dicom_file(path)`: Procesa archivos DICOM
- `read_jpg_file(path)`: Procesa imágenes JPG/PNG
- `read_image_file(path)`: Detector automático de formato

### ⚙️ `src/data/preprocess_img.py` - Preprocesamiento
**Función**: Preparación de imágenes para el modelo neuronal.

**Pipeline de procesamiento**:
1. **📏 Redimensionamiento**: 512x512 píxeles
2. **⚫ Escala de grises**: Conversión BGR → GRAY
3. **🔆 CLAHE**: Ecualización adaptativa del histograma
   - `clipLimit=2.0`
   - `tileGridSize=(4,4)`
4. **📊 Normalización**: Valores 0-1
5. **🎯 Tensorización**: Formato batch `(1, 512, 512, 1)`

### 🤖 `src/data/load_model.py` - Carga del Modelo
**Función**: Gestión y carga del modelo de red neuronal.

**Características**:
- 📂 Carga de `models/conv_MLP_84.h5`
- ✅ Verificación de existencia de archivo
- 🛡️ Manejo de errores de compatibilidad
- 🔍 Validación de arquitectura (capa `conv10_thisone`)

### 🔥 `src/data/grad_cam.py` - Mapas de Calor Grad-CAM
**Función**: Generación de visualizaciones explicativas de las predicciones.

**Algoritmo Grad-CAM**:
1. 🎯 **Predicción**: Obtiene clase predicha
2. 🧮 **Gradientes**: Calcula gradientes de la salida respecto a `conv10_thisone`
3. 📊 **Ponderación**: Promedia gradientes por canal
4. 🔥 **Mapa de calor**: Genera activación ponderada
5. 🎨 **Visualización**: Aplica colormap JET
6. 🖼️ **Superposición**: Combina con imagen original (α=0.8)

---

## Guía de Uso

### 🖥️ Ejecución de la Aplicación

```bash
# Desde el directorio raíz del proyecto
python main.py
py main.py

# O usar el script de lanzamiento (Linux/macOS)
chmod +x launch_uao_neumonia.sh
./launch_uao_neumonia.sh
```

### 📋 Flujo de Trabajo Recomendado

1. **📁 Cargar imagen**: 
   - Click en "Cargar Imagen"
   - Seleccionar archivo DICOM (.dcm), JPG o PNG
   - Verificar que la imagen se muestre correctamente

2. **🏥 Información del paciente**: 
   - Ingresar cédula o ID del paciente
   - Este dato se guardará in el historial

3. **🔮 Realizar predicción**: 
   - Click en "Predecir"
   - Esperar procesamiento (5-10 segundos)
   - Ver resultado y mapa de calor

4. **📊 Interpretar resultados**:
   - **Etiqueta**: Tipo de neumonía detectada
   - **Probabilidad**: Confianza del modelo (0-100%)
   - **Mapa de calor**: Regiones relevantes para la decisión

5. **💾 Guardar resultados**:
   - **CSV**: Click en "Guardar" para historial
   - **PDF**: Click en "PDF" para reporte completo

---

### 🏗️ Arquitectura del Modelo conv_MLP_84

```python
# Resumen de la arquitectura - ⚠️Generado con el model.summary()
Model: "conv_MLP_84"
_________________________________________________________________

━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_9             │ (None, 512, 512,  │          0 │ -                 │
│ (InputLayer)        │ 1)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1 (Conv2D)      │ (None, 512, 512,  │        160 │ input_9[0][0]     │
│                     │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv1            │ (None, 512, 512,  │         64 │ conv1[0][0]       │
│ (BatchNormalizatio… │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2 (Conv2D)      │ (None, 512, 512,  │      2,320 │ bn_conv1[0][0]    │
│                     │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1skip (Conv2D)  │ (None, 512, 512,  │         32 │ input_9[0][0]     │
│                     │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv2            │ (None, 512, 512,  │         64 │ conv2[0][0]       │
│ (BatchNormalizatio… │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv1skp         │ (None, 512, 512,  │         64 │ conv1skip[0][0]   │
│ (BatchNormalizatio… │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_16 (Add)        │ (None, 512, 512,  │          0 │ bn_conv2[0][0],   │
│                     │ 16)               │            │ bn_conv1skp[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_10    │ (None, 255, 255,  │          0 │ add_16[0][0]      │
│ (MaxPooling2D)      │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3 (Conv2D)      │ (None, 255, 255,  │      4,640 │ max_pooling2d_10… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv3            │ (None, 255, 255,  │        128 │ conv3[0][0]       │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_16       │ (None, 255, 255,  │          0 │ bn_conv3[0][0]    │
│ (Activation)        │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4 (Conv2D)      │ (None, 255, 255,  │      9,248 │ activation_16[0]… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2skip (Conv2D)  │ (None, 255, 255,  │        544 │ max_pooling2d_10… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv4            │ (None, 255, 255,  │        128 │ conv4[0][0]       │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv2skp         │ (None, 255, 255,  │        128 │ conv2skip[0][0]   │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_17 (Add)        │ (None, 255, 255,  │          0 │ bn_conv4[0][0],   │
│                     │ 32)               │            │ bn_conv2skp[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_17       │ (None, 255, 255,  │          0 │ add_17[0][0]      │
│ (Activation)        │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_11    │ (None, 127, 127,  │          0 │ activation_17[0]… │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5 (Conv2D)      │ (None, 127, 127,  │     13,872 │ max_pooling2d_11… │
│                     │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv5            │ (None, 127, 127,  │        192 │ conv5[0][0]       │
│ (BatchNormalizatio… │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_18       │ (None, 127, 127,  │          0 │ bn_conv5[0][0]    │
│ (Activation)        │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv6 (Conv2D)      │ (None, 127, 127,  │     20,784 │ activation_18[0]… │
│                     │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv3skip (Conv2D)  │ (None, 127, 127,  │      1,584 │ max_pooling2d_11… │
│                     │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv6            │ (None, 127, 127,  │        192 │ conv6[0][0]       │
│ (BatchNormalizatio… │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv3skp         │ (None, 127, 127,  │        192 │ conv3skip[0][0]   │
│ (BatchNormalizatio… │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_18 (Add)        │ (None, 127, 127,  │          0 │ bn_conv6[0][0],   │
│                     │ 48)               │            │ bn_conv3skp[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_19       │ (None, 127, 127,  │          0 │ add_18[0][0]      │
│ (Activation)        │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_12    │ (None, 63, 63,    │          0 │ activation_19[0]… │
│ (MaxPooling2D)      │ 48)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv7 (Conv2D)      │ (None, 63, 63,    │     27,712 │ max_pooling2d_12… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv7            │ (None, 63, 63,    │        256 │ conv7[0][0]       │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_20       │ (None, 63, 63,    │          0 │ bn_conv7[0][0]    │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_3 (Dropout) │ (None, 63, 63,    │          0 │ activation_20[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv8 (Conv2D)      │ (None, 63, 63,    │     36,928 │ dropout_3[0][0]   │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv4skip (Conv2D)  │ (None, 63, 63,    │      3,136 │ max_pooling2d_12… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv8            │ (None, 63, 63,    │        256 │ conv8[0][0]       │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv4skp         │ (None, 63, 63,    │        256 │ conv4skip[0][0]   │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_19 (Add)        │ (None, 63, 63,    │          0 │ bn_conv8[0][0],   │
│                     │ 64)               │            │ bn_conv4skp[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_21       │ (None, 63, 63,    │          0 │ add_19[0][0]      │
│ (Activation)        │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_13    │ (None, 31, 31,    │          0 │ activation_21[0]… │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv9 (Conv2D)      │ (None, 31, 31,    │     73,856 │ max_pooling2d_13… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv9            │ (None, 31, 31,    │        512 │ conv9[0][0]       │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_22       │ (None, 31, 31,    │          0 │ bn_conv9[0][0]    │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_4 (Dropout) │ (None, 31, 31,    │          0 │ activation_22[0]… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv10_thisone      │ (None, 31, 31,    │    147,584 │ dropout_4[0][0]   │
│ (Conv2D)            │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv5skip (Conv2D)  │ (None, 31, 31,    │      8,320 │ max_pooling2d_13… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv10           │ (None, 31, 31,    │        512 │ conv10_thisone[0… │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_conv5skp         │ (None, 31, 31,    │        512 │ conv5skip[0][0]   │
│ (BatchNormalizatio… │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_20 (Add)        │ (None, 31, 31,    │          0 │ bn_conv10[0][0],  │
│                     │ 128)              │            │ bn_conv5skp[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_23       │ (None, 31, 31,    │          0 │ add_20[0][0]      │
│ (Activation)        │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_14    │ (None, 15, 15,    │          0 │ activation_23[0]… │
│ (MaxPooling2D)      │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ average_pooling2d_2 │ (None, 8, 8, 128) │          0 │ max_pooling2d_14… │
│ (AveragePooling2D)  │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten_2 (Flatten) │ (None, 8192)      │          0 │ average_pooling2… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ fc1 (Dense)         │ (None, 1024)      │  8,389,632 │ flatten_2[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_5 (Dropout) │ (None, 1024)      │          0 │ fc1[0][0]         │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ fc2 (Dense)         │ (None, 1024)      │  1,049,600 │ dropout_5[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ fc3 (Dense)         │ (None, 3)         │      3,075 │ fc2[0][0]         │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 9,796,483 (37.37 MB)
 Trainable params: 9,794,755 (37.36 MB)
 Non-trainable params: 1,728 (6.75 KB)

_________________________________________________________________

```
---

## ⚖️ Licencia

Este proyecto está licenciado bajo la **Licencia Apache 2.0** - ver el archivo [LICENSE](LICENSE) para detalles completos.

```
### ⚠️ Disclaimer Médico
**IMPORTANTE**: Este sistema es una herramienta de **apoyo al diagnóstico** y NO debe utilizarse como sustituto del criterio médico profesional. Siempre consulte con un radiólogo o médico especialista para confirmación diagnóstica.

### 🏥 Uso Permitido
- ✅ Investigación académica y científica
- ✅ Educación médica y formación
- ✅ Desarrollo de software médico
- ✅ Screening preliminar supervisado
- ❌ Diagnóstico definitivo sin supervisión médica
- ❌ Uso comercial sin autorización explícita

---

## 📚 Referencias y Bibliografía

### 📄 Artículos Científicos Fundamentales

1. **Selvaraju, R. R., et al. (2017)**. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *IEEE International Conference on Computer Vision (ICCV)*, 618-626.

2. **Rajpurkar, P., et al. (2017)**. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." *arXiv preprint arXiv:1711.05225*.

3. **Kermany, D. S., et al. (2018)**. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122-1131.

4. **Zuiderveld, K. (1994)**. "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, Academic Press Professional, 474-485.

### 🛠️ Tecnologías y Frameworks

- **TensorFlow/Keras**: Framework de deep learning
- **OpenCV**: Biblioteca de computer vision
- **PyDICOM**: Procesamiento de imágenes médicas DICOM  
- **NumPy**: Computación numérica
- **Pillow**: Manipulación de imágenes
- **Tkinter**: Interfaz gráfica nativa de Python
- **UV**: Gestión moderna de dependencias de Python


**Última Actualización**: Septiembre 19, 2025
**Estado del Proyecto**: Producción Estable 🟢  
