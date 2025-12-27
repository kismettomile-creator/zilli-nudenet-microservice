FROM python:3.11-slim

# Çalışma dizini oluştur
WORKDIR /app

# Force apt cache refresh and install all required libraries for OpenCV, DeepFace, NudeNet
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libxi6 \
    libxrandr2 \
    libxtst6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu ekle
COPY . .

# Microservice'i başlat (same as backend: port 3000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
