FROM python:3.11-slim

# Çalışma dizini oluştur
WORKDIR /app

# System dependencies for image processing and compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama kodunu ekle
COPY . .

# Microservice'i başlat (2 worker - daha az resource kullanımı)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
