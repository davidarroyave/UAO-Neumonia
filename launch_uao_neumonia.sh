#!/bin/bash

echo "🚀 Iniciando Sistema de Detección de Neumonía UAO..."
echo "=================================================="

# Configurar entorno gráfico para X11 (necesario para PDF)
export GDK_BACKEND=x11
export XDG_SESSION_TYPE=x11 
export QT_QPA_PLATFORM=xcb
unset WAYLAND_DISPLAY

# Activar entorno virtual
echo "📦 Activando entorno virtual..."
source .venv/bin/activate

# Verificar configuración
echo "🔧 Configuración del entorno:"
echo "   • GDK_BACKEND: $GDK_BACKEND"
echo "   • XDG_SESSION_TYPE: $XDG_SESSION_TYPE" 
echo "   • DISPLAY: $DISPLAY"
echo "   • Python: $(python --version)"
echo ""

# Ejecutar aplicación
echo "🩺 Iniciando aplicación de detección de neumonía..."
echo "   • Todas las funcionalidades habilitadas"
echo "   • Generación de PDF funcionando correctamente"
echo "   • Grad-CAM (heatmap) activo"
echo ""

python main.py

echo "✅ Aplicación finalizada."
