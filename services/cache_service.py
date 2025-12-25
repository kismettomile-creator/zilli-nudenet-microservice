"""
Simple Cache Service for Microservices
Redis bağlantısı ve basit cache operasyonları
"""

import json
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

class InMemoryCache:
    """Redis olmadan çalışan in-memory cache (fallback)"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.expiry = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            # Expire kontrolü
            if key in self.expiry:
                if datetime.now() > self.expiry[key]:
                    del self.cache[key]
                    del self.expiry[key]
                    return None
            
            if key in self.cache:
                # LRU: En son kullanılanı sona taşı
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        with self.lock:
            # Max size kontrolü (LRU eviction)
            if len(self.cache) >= self.max_size and key not in self.cache:
                # En eski key'i sil
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.expiry:
                    del self.expiry[oldest_key]
            
            self.cache[key] = value
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
            self.cache.move_to_end(key)
    
    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.expiry:
                del self.expiry[key]
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)

class CacheService:
    """
    Redis cache servisi - Microservices için basitleştirilmiş versiyon
    """
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = InMemoryCache(max_size=1000)
        self.use_redis = False
        self._connect()
    
    def _connect(self):
        """Redis'e bağlan (opsiyonel)"""
        try:
            import redis
            import os
            
            # Production Redis URL (Backend ile aynı)
            PRODUCTION_REDIS_URL = "redis://default:7yDTru9ycIuL6nObyzoffK69kmRWx7W08GLGLpW0yw4iRhUyrBMqGkhB2p0lgdCu@gswwsc8coskck44448occos4:6379/0"
            
            # Environment'a göre Redis URL seç
            redis_url = os.getenv('REDIS_URL', PRODUCTION_REDIS_URL)
            
            # Redis URL ile bağlan
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                max_connections=100,  # Microservice için daha az connection
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Bağlantı testi
            self.redis_client.ping()
            self.use_redis = True
            
            # URL'den host bilgisini parse et (log için)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(redis_url)
                host_info = f"{parsed.hostname}:{parsed.port}"
            except:
                host_info = "remote"
            
            logger.info(f"✅ Redis cache servisi aktif ({host_info})")
                
        except ImportError:
            logger.info("ℹ️ Redis paketi yok, in-memory cache kullanılıyor")
            self.redis_client = None
            self.use_redis = False
        except Exception as e:
            logger.info("ℹ️ Redis bağlanamadı, in-memory cache kullanılıyor (fallback)")
            logger.debug(f"Redis bağlantı hatası: {e}")
            self.redis_client = None
            self.use_redis = False
    
    def get(self, key: str) -> Optional[Any]:
        """Cache'den veri oku (Redis veya in-memory)"""
        try:
            if self.use_redis and self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    logger.debug(f"🎯 Redis HIT: {key}")
                    return json.loads(data)
                else:
                    logger.debug(f"❌ Redis MISS: {key}")
                    return None
            else:
                # In-memory'den oku
                data = self.memory_cache.get(key)
                if data:
                    logger.debug(f"🎯 Memory HIT: {key}")
                else:
                    logger.debug(f"❌ Memory MISS: {key}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"❌ Cache JSON decode hatası ({key}): {e}")
            self.delete(key)
            return None
        except Exception as e:
            logger.error(f"❌ Cache okuma hatası ({key}): {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Cache'e veri kaydet (Redis veya in-memory)"""
        try:
            if self.use_redis and self.redis_client:
                # ✅ Set'leri list'e çevir (JSON serializable)
                if isinstance(value, set):
                    value = list(value)
                
                # Redis'e kaydet
                json_data = json.dumps(value, ensure_ascii=False, default=str)
                self.redis_client.setex(key, ttl, json_data)
                logger.debug(f"💾 Redis SET: {key} (TTL: {ttl}s)")
            else:
                # In-memory'e kaydet (set olarak kalabilir)
                self.memory_cache.set(key, value, ttl)
                logger.debug(f"💾 Memory SET: {key} (TTL: {ttl}s)")
        except TypeError as e:
            logger.error(f"❌ Cache serialize hatası ({key}): {e}")
            logger.error(f"   Value type: {type(value)}")
        except Exception as e:
            logger.error(f"❌ Cache yazma hatası ({key}): {e}")
    
    def delete(self, key: str):
        """Cache'den veri sil (Redis veya in-memory)"""
        try:
            if self.use_redis and self.redis_client:
                self.redis_client.delete(key)
                logger.debug(f"🗑️ Redis DELETE: {key}")
            else:
                self.memory_cache.delete(key)
                logger.debug(f"🗑️ Memory DELETE: {key}")
        except Exception as e:
            logger.error(f"❌ Cache silme hatası ({key}): {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistikleri (Redis veya in-memory)"""
        try:
            if self.use_redis and self.redis_client:
                info = self.redis_client.info()
                return {
                    "enabled": True,
                    "backend": "redis",
                    "status": "active",
                    "total_keys": self.redis_client.dbsize(),
                    "memory_used": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                }
            else:
                return {
                    "enabled": True,
                    "backend": "in-memory",
                    "status": "active",
                    "total_keys": self.memory_cache.size(),
                    "max_size": self.memory_cache.max_size,
                }
        except Exception as e:
            logger.error(f"❌ Cache stats hatası: {e}")
            return {
                "enabled": False,
                "status": "error",
                "error": str(e)
            }

# Global cache service instance
cache_service = CacheService()
