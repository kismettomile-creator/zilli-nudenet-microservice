# Kısmet Microservices

Kısmet uygulaması için ağır işlemleri yürüten mikroservis koleksiyonu.

**🌐 Production Domain:** `micro.zilli-app.com`

## 🎯 Amaç

Ana API'den ayrıştırılan ağır işlemler:
- **NudeNet content moderation** (tamamen buraya taşındı)
- Background job'lar  
- Ağır hesaplama gerektiren operasyonlar
- External API integrations

## 🚀 Kurulum

```bash
# Dependencies yükle
pip install -r requirements.txt

# Servisi başlat
python main.py

# Ya da uvicorn ile
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## 🐳 Docker ile Çalıştırma

```bash
# Build
docker build -t kismet-microservices .

# Run
docker run -p 8001:8001 kismet-microservices
```

## 📚 API Dokümantasyonu

Servis başladıktan sonra:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- Health Check: `http://localhost:8001/health`

## 🏗️ Proje Yapısı

```
microservices/
├── main.py              # Ana FastAPI app
├── core/                # Core utilities
│   ├── __init__.py
│   └── timezone_utils.py
├── services/            # Service layer
│   ├── __init__.py
│   └── cache_service.py
├── routes/              # API endpoints (eklenecek)
├── requirements.txt     # Dependencies
├── Dockerfile          # Container config
└── README.md           # Bu dosya
```

## 🔧 Configuration

Environment variables:
- `REDIS_URL`: Redis connection string
- `LOG_LEVEL`: Logging seviyesi (default: INFO)

## 🚀 Deployment (Coolify)

1. Repository'yi Coolify'a bağla
2. Build type: Dockerfile
3. Port: 8001
4. Environment variables'ları ayarla

## 📈 Monitoring

Health check endpoint: `/health`

Response örneği:
```json
{
  "status": "healthy",
  "service": "microservices", 
  "timestamp": "2024-12-26T...",
  "uptime_seconds": 3600,
  "cache_connected": true,
  "version": "1.0.0"
}
```

Kısmet uygulaması için ağır işlemleri yürüten mikroservis koleksiyonu.

**🌐 Production Domain:** `micro.zilli-app.com`

## 🎯 Amaç

Ana API'den ayrıştırılan ağır işlemler:
- **NudeNet content moderation** (tamamen buraya taşındı)
- Background job'lar  
- Ağır hesaplama gerektiren operasyonlar
- External API integrations

## 🚀 Kurulum

```bash
# Dependencies yükle
pip install -r requirements.txt

# Servisi başlat
python main.py

# Ya da uvicorn ile
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## 🐳 Docker ile Çalıştırma

```bash
# Build
docker build -t kismet-microservices .

# Run
docker run -p 8001:8001 kismet-microservices
```

## 📚 API Dokümantasyonu

Servis başladıktan sonra:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- Health Check: `http://localhost:8001/health`

## 🏗️ Proje Yapısı

```
microservices/
├── main.py              # Ana FastAPI app
├── core/                # Core utilities
│   ├── __init__.py
│   └── timezone_utils.py
├── services/            # Service layer
│   ├── __init__.py
│   └── cache_service.py
├── routes/              # API endpoints (eklenecek)
├── requirements.txt     # Dependencies
├── Dockerfile          # Container config
└── README.md           # Bu dosya
```

## 🔧 Configuration

Environment variables:
- `REDIS_URL`: Redis connection string
- `LOG_LEVEL`: Logging seviyesi (default: INFO)

## 🚀 Deployment (Coolify)

1. Repository'yi Coolify'a bağla
2. Build type: Dockerfile
3. Port: 8001
4. Environment variables'ları ayarla

## 📈 Monitoring

Health check endpoint: `/health`

Response örneği:
```json
{
  "status": "healthy",
  "service": "microservices", 
  "timestamp": "2024-12-26T...",
  "uptime_seconds": 3600,
  "cache_connected": true,
  "version": "1.0.0"
}
```
