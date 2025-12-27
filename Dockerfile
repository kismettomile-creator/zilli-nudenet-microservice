FROM python:3.11-slim

# Çalışma dizini oluştur
WORKDIR /app

# System dependencies for headless image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama kodunu ekle
COPY . .

# Microservice'i başlat (same as backend: port 3000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]

# Çalışma dizini oluştur
WORKDIR /app

# System dependencies for headless image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama kodunu ekle
COPY . .

# Microservice'i başlat (same as backend: port 3000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
