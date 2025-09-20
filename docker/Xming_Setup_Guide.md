# 🖥️ Guía de Configuración Xming para GUI Docker

## 📋 Resumen

Esta guía te ayudará a configurar Xming en Windows para ejecutar la interfaz gráfica del contenedor Docker de detección de neumonía.

---

## ✅ Estado del Contenedor

**🎉 CONTENEDOR VALIDADO EXITOSAMENTE**

- ✅ TensorFlow 2.20.0 funcionando
- ✅ OpenCV 4.11.0 funcionando  
- ✅ Todos los módulos del proyecto funcionando
- ✅ Launcher script configurado correctamente
- ✅ **Solo falta configurar Xming para la GUI**

---

## 📥 Paso 1: Descargar e Instalar Xming

1. Ve a: https://sourceforge.net/projects/xming/
2. Descarga **Xming** (no Xming-fonts)
3. Ejecuta el instalador en Windows
4. **IMPORTANTE**: Durante la instalación, asegúrate de seleccionar:
   - ☑️ **"Disable access control"** (CRÍTICO)

---

## 🚀 Paso 2: Configurar y Ejecutar Xming

### Opción A: Ejecutar Xming Directamente (Recomendado)
1. Busca **"Xming"** en el menú inicio de Windows
2. Ejecuta **Xming** (no XLaunch)
3. Verás un ícono de Xming en la bandeja del sistema
4. Haz clic derecho en el ícono → **"Exit"** si está ejecutándose
5. Ejecuta de nuevo con: `Xming :0 -multiwindow -clipboard -ac`

### Opción B: Usar XLaunch (Configuración Manual)
1. Busca **"XLaunch"** en el menú inicio
2. Ejecuta **XLaunch** y configura:
   - **Display settings**: ☑️ Multiple windows
   - **Client startup**: ☑️ Start no client  
   - **Extra settings**: 
     - ☑️ Clipboard
     - ☑️ **Disable access control** (CRÍTICO)
   - **Display number**: 0
3. Guarda la configuración para uso futuro

---

## 🔧 Paso 3: Verificar que Xming está Funcionando

En Windows, abre **Command Prompt** y ejecuta:
```cmd
netstat -an | find "6000"
```

Deberías ver algo como:
```
TCP    0.0.0.0:6000    0.0.0.0:0    LISTENING
```

Si no ves esto, Xming no está ejecutándose correctamente.

---

## 🐳 Paso 4: Ejecutar el Contenedor con GUI

Regresa a tu terminal WSL2 y ejecuta:

### Opción A - Script Helper (Recomendado):
```bash
./run_docker_xming.sh
```

### Opción B - Comando Manual:
```bash
# Configurar variables
WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3}')
export DISPLAY="$WINDOWS_IP:0.0"

# Ejecutar contenedor
docker run --rm \
    -e DISPLAY="$DISPLAY" \
    -e GDK_BACKEND=x11 \
    -e XDG_SESSION_TYPE=x11 \
    -e QT_QPA_PLATFORM=xcb \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --net=host \
    uao-neumonia:latest \
    ./launch_uao_neumonia.sh
```

---

## ✅ Resultado Esperado

Si todo está configurado correctamente, deberías ver:

1. **En la terminal WSL2**:
   ```
   🚀 Iniciando Sistema de Detección de Neumonía UAO...
   📦 Activando entorno virtual...
   🔧 Configuración del entorno:
   🩺 Iniciando aplicación de detección de neumonía...
   ```

2. **En Windows**: Una ventana con la interfaz gráfica de la aplicación de neumonía

---

## 🔧 Troubleshooting

### ❌ Error: "Connection refused" 
**Problema**: Xming no está ejecutándose o configurado incorrectamente
**Soluciones**:
1. Verifica que Xming esté en la bandeja del sistema
2. Reinicia Xming con: `Xming :0 -multiwindow -clipboard -ac`
3. Verifica el puerto 6000: `netstat -an | find "6000"`

### ❌ Error: "Display connection error"
**Problema**: Configuración de display incorrecta
**Soluciones**:
1. Verifica la IP de Windows: `ip route show | grep default`
2. Configura DISPLAY: `export DISPLAY="172.27.208.1:0.0"` (usar tu IP)
3. Reinicia el contenedor

### ❌ Error: "Access denied"
**Problema**: Access control no deshabilitado
**Soluciones**:
1. **CRÍTICO**: Asegúrate de que "Disable access control" esté marcado
2. Ejecuta Xming con `-ac`: `Xming :0 -multiwindow -clipboard -ac`
3. En WSL2: `xhost +local:root`

### ❌ La ventana no aparece
**Soluciones**:
1. Verifica que Xming esté ejecutándose (ícono en bandeja)
2. Prueba test simple: 
   ```bash
   docker run --rm -e DISPLAY="$WINDOWS_IP:0.0" --net=host uao-neumonia:latest python -c "
   import tkinter as tk
   root = tk.Tk()
   root.title('Test X11')
   root.geometry('300x200')
   root.mainloop()
   "
   ```

---

## 📊 Información Técnica

### Configuración Actual Detectada:
- **IP de Windows**: 172.27.208.1
- **Puerto X11**: 6000
- **DISPLAY**: 172.27.208.1:0.0
- **Contenedor**: uao-neumonia:latest (✅ Funcional)

### Comandos Útiles:
```bash
# Verificar IP de Windows
ip route show | grep default

# Test de conectividad X11
nc -zv $WINDOWS_IP 6000

# Variables de entorno
echo $DISPLAY

# Status del contenedor
docker images uao-neumonia
```

---

## 🎯 Una vez que funcione

Cuando la GUI se ejecute correctamente, podrás:

1. **Cargar imágenes**: DICOM, PNG, JPG
2. **Realizar predicciones**: Neumonía vs Normal
3. **Ver heatmaps**: Grad-CAM visualization
4. **Generar reportes PDF**: Documentación médica
5. **Guardar resultados**: Base de datos CSV

---

## 📞 Siguientes Pasos

1. **Configura Xming** siguiendo esta guía
2. **Ejecuta el contenedor** con GUI
3. **Prueba la aplicación** con imágenes de muestra
4. **Documenta el flujo** para otros usuarios

---

*Guía creada el 19/09/2025 - Contenedor Docker UAO Neumonía*