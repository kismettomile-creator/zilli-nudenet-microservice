FROM python:3.12-slim

# Çalışma dizini oluştur
WORKDIR /app

# System dependencies for potential image processing
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama kodunu ekle
COPY . .

# Microservice'i başlat (2 worker - daha az resource kullanımı)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
