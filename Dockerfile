FROM python:3.11-slim

# NO incluimos las variables ENV OMP_NUM_THREADS
# Esto permite a OpenCV y Sklearn usar MÚLTIPLES núcleos (¡VELOCIDAD!)

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt \
    && python -m nltk.downloader stopwords \
    && python -m nltk.downloader punkt_tab

COPY src/ ./src/

EXPOSE 8000

# --- INICIO DE LA SOLUCIÓN ---
# Usar "--workers 1" le dice a Uvicorn que corra en modo de producción simple.
# Esto deshabilita el "reloader" (arreglando el bug de reinicio)
# ¡PERO SÍ PERMITE que el proceso use todos tus núcleos de CPU!
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# --- FIN DE LA SOLUCIÓN ---