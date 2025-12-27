FROM python:3.11-slim

# Çalışma dizini oluştur
WORKDIR /app

# System dependencies for OpenCV, NudeNet, and DeepFace (headless)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu ekle
COPY . .

# Microservice'i başlat (same as backend: port 3000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
