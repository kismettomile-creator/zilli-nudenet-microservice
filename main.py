"""
🚀 Kısmet Microservices API
Ağır işlemler için ayrılmış mikroservis (NudeNet, background jobs, vb.)
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# ==================== LIFESPAN EVENT HANDLER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Kısmet Microservices starting up...")
    
    # Initialize cache service (Redis connection)
    logger.info("🔗 Initializing cache service...")
    try:
        from services.cache_service import cache_service
        cache_stats = cache_service.get_stats()
        logger.info(f"✅ Cache service ready: {cache_stats.get('backend', 'unknown')}")
    except Exception as e:
        logger.warning(f"⚠️ Cache service failed: {e}")
    
    # 🔥 Warmup NudeNet model for content moderation
    logger.info("🧠 Warming up NudeNet model for content moderation...")
    try:
        from routes.content_routes import warmup_nudenet
        await warmup_nudenet()
        logger.info("✅ NudeNet model warmed up successfully")
    except Exception as e:
        logger.error(f"❌ NudeNet warmup failed: {e}")
        logger.warning("⚠️ Content moderation may be slower on first request")
    
    logger.info("✅ Microservices startup completed!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Kısmet Microservices shutting down...")

# ==================== FASTAPI SETUP ====================
app = FastAPI(
    title="Kısmet Microservices API",
    description="Heavy processing microservices for Kısmet app",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== INCLUDE ROUTERS ====================
from routes.content_routes import router as content_router
app.include_router(content_router, prefix="/content", tags=["Content Moderation"])

# ==================== HEALTH CHECK ====================
@app.get("/health")
async def health_check(request: Request):
    """
    Mikroservis sağlık kontrolü
    
    Returns:
        dict: Servis durumu ve temel bilgiler
    """
    try:
        from services.cache_service import cache_service
        cache_healthy = cache_service.redis_client is not None
    except:
        cache_healthy = False
    
    return {
        "status": "healthy",
        "service": "microservices",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - startup_time),
        "cache_connected": cache_healthy,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Kısmet Microservices API",
        "docs": "/docs",
        "health": "/health"
    }

# Global startup time for uptime calculation
startup_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
