# SCHIMBĂ ACEASTĂ LINIE (era 3.9-slim)
FROM python:3.11-slim

# ... restul fișierului rămâne EXACT la fel ...
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
WORKDIR /app

# Asigură-te că ai păstrat modificarea cu libgl1 de mai devreme
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "src/reorganize_project.py"]