#!/bin/bash

echo "🐳 UAO Neumonía - Docker con Xming"
echo "=================================="
echo ""

# Detectar IP de Windows (WSL2)
WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3}')

if [ -z "$WINDOWS_IP" ]; then
    echo "⚠️  No se pudo detectar la IP de Windows. Usando localhost..."
    WINDOWS_IP="localhost"
fi

echo "🖥️  IP de Windows detectada: $WINDOWS_IP"
echo ""

# Configurar DISPLAY para Xming
export DISPLAY="$WINDOWS_IP:0.0"

echo "📋 Configuración para Xming:"
echo "   • DISPLAY: $DISPLAY"
echo "   • Puerto X11: 6000"
echo ""

echo "📝 INSTRUCCIONES PARA XMING:"
echo "   1. Descarga e instala Xming: https://sourceforge.net/projects/xming/"
echo "   2. Ejecuta Xming con estas configuraciones:"
echo "      - Multiple Windows mode"
echo "      - Display number: 0"
echo "      - Disable access control (importante)"
echo "   3. Asegúrate de que Xming esté ejecutándose antes de continuar"
echo ""

read -p "🤔 ¿Ya tienes Xming ejecutándose? (y/n): " xming_ready

if [[ $xming_ready != "y" && $xming_ready != "Y" ]]; then
    echo "❌ Por favor inicia Xming primero y luego ejecuta este script de nuevo."
    exit 1
fi

echo ""
echo "🚀 Ejecutando contenedor con GUI..."
echo "   • Esto abrirá la aplicación de neumonía en una ventana"
echo "   • Si no aparece la ventana, verifica la configuración de Xming"
echo ""

# Ejecutar el contenedor con X11 forwarding
docker run --rm \
    -e DISPLAY="$DISPLAY" \
    -e GDK_BACKEND=x11 \
    -e XDG_SESSION_TYPE=x11 \
    -e QT_QPA_PLATFORM=xcb \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume /home/juanito/.Xauthority:/root/.Xauthority:ro \
    --net=host \
    uao-neumonia:latest \
    ./launch_uao_neumonia.sh

echo ""
echo "✅ Contenedor finalizado."