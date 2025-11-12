FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# --- INICIO DE LA SOLUCIÓN ---
# Instalar dependencias del S.O. (para OpenCV)
# 1. libgl1: Resuelve el 'libGL.so.1'
# 2. libglib2.0-0: Resuelve el 'libgthread-2.0.so.0'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt
# --- FIN DE LA SOLUCIÓN ---

COPY src/ ./src/

EXPOSE 8000

# Comando para ejecutar uvicorn
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]