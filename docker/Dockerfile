# Multi-stage build para optimizar el tamaño de la imagen final
FROM python:3.11-slim AS builder

# Instalar dependencias del sistema necesarias para el build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV de manera más eficiente
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración de dependencias
COPY pyproject.toml uv.lock ./

# Instalar dependencias usando UV con lockfile actualizado
RUN uv sync --frozen

# Imagen final más ligera
FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    tk-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copiar el entorno virtual desde el builder
COPY --from=builder /app/.venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Copiar el código de la aplicación
COPY --chown=app:app . .

# Hacer el launcher ejecutable (antes de cambiar a usuario no-root)
RUN chmod +x launch_uao_neumonia.sh

# Cambiar al usuario no-root
USER app

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/home/app
ENV DISPLAY=:0

# Puerto si tu app lo necesita (por ejemplo, para una API)
# EXPOSE 8000

# Comando para ejecutar la app
CMD ["python", "main.py"]