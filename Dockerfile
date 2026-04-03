FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar pyproject.toml y código fuente antes del install
# para que pip pueda instalar el proyecto y sus dependencias en un solo paso
COPY pyproject.toml .
COPY src/ ./src/

# Instala todas las dependencias declaradas en pyproject.toml
# + registra el paquete src (necesario para que los imports absolutos funcionen)
RUN pip install --no-cache-dir .

# Crear directorio de datos (se monta como volumen en producción)
RUN mkdir -p /app/data

# Exponer puerto de la API
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
