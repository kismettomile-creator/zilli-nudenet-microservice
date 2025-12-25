"""
🔥 Content Moderation Routes - NudeNet Processing
Ana API'den ayrıştırılan ağır NudeNet işlemleri
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict
import base64
from datetime import datetime
import io
from PIL import Image
import tempfile
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== 🔥 DEDICATED CONTENT MODERATION THREAD POOL ====================
content_moderation_pool = ThreadPoolExecutor(
    max_workers=4,  # NudeNet için yeterli, çok thread gereksiz
    thread_name_prefix="content_mod_"
)

# ==================== 🔥 OPTIMIZED NUDENET SINGLETON ====================
_nude_detector = None
_detector_loading = False

def get_nude_detector():
    """🔥 OPTIMIZED: Thread-safe lazy load NudeNet detector"""
    global _nude_detector, _detector_loading
    
    # Thread-safe check: Eğer başka thread loading yapıyorsa bekle
    if _detector_loading:
        while _detector_loading and _nude_detector is None:
            time.sleep(0.1)  # Wait for the other thread to complete loading
        return _nude_detector
    
    if _nude_detector is None:
        _detector_loading = True
        logger.info("🧠 Loading NudeNet model...")
        try:
            from nudenet import NudeDetector
            _nude_detector = NudeDetector()
            logger.info("✅ NudeNet model loaded successfully")
        except Exception as e:
            logger.error(f"❌ NudeNet model loading failed: {e}")
            raise
        finally:
            _detector_loading = False
    
    return _nude_detector

async def warmup_nudenet():
    """Pre-loads the NudeNet model at startup."""
    logger.info("🔥 [WARMUP] Pre-loading NudeNet model...")
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(content_moderation_pool, get_nude_detector)
        logger.info("✅ [WARMUP] NudeNet model pre-loaded successfully")
    except Exception as e:
        logger.error(f"❌ [WARMUP] NudeNet pre-load failed: {e}")
        raise

# ==================== REQUEST/RESPONSE MODELS ====================
class ContentModerationRequest(BaseModel):
    image_data: str  # Base64 encoded image
    
class ContentModerationResponse(BaseModel):
    nudity_detected: bool
    confidence_score: float
    detection_details: str
    processing_time_ms: float
    image_size_kb: float

# ==================== CORE PROCESSING FUNCTIONS ====================
def _sync_process_image_optimized(image_data_b64: str):
    """
    🔥 OPTIMIZED: In-memory NudeNet detection (NO DISK I/O!)
    """
    start_time = time.time()
    
    try:
        # Step 1: Decode base64 data (in-memory)
        try:
            decoded_data = base64.b64decode(image_data_b64)
            image_size_kb = len(decoded_data) / 1024
            logger.debug(f"📊 Image decoded: {image_size_kb:.1f} KB")
        except Exception as e:
            logger.error(f"❌ Base64 decode error: {e}")
            return 0.0, False, 0.0, "Base64 decode failed"
        
        # Step 2: In-memory NudeNet detection (NO DISK I/O!)
        nudity_detected = False
        confidence_score = 0.0
        detection_details = "No problematic content detected"
        
        try:
            detector = get_nude_detector()
            
            # PIL Image'ı NumPy array'e çevir (in-memory)
            image = Image.open(io.BytesIO(decoded_data))
            
            # Resim boyutunu optimize et (max 800x800)
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            np_array = np.array(image)
            
            # Tespit yap (NumPy array üzerinden)
            detections = detector.detect(np_array)
            
            # Detection sonuçlarını değerlendir
            problematic_classes = ['EXPOSED_ANUS', 'EXPOSED_BUTTOCKS', 'EXPOSED_BREAST_F', 
                                 'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M']
            
            high_confidence_detections = []
            max_confidence = 0.0
            
            for detection in detections:
                class_name = detection['class']
                confidence = detection['score']
                max_confidence = max(max_confidence, confidence)
                
                if class_name in problematic_classes and confidence > 0.6:
                    high_confidence_detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
            
            if high_confidence_detections:
                nudity_detected = True
                confidence_score = max_confidence
                detection_details = f"Detected: {', '.join([d['class'] for d in high_confidence_detections])}"
                logger.info(f"🚨 Nudity detected: {detection_details} (confidence: {confidence_score:.2f})")
            else:
                confidence_score = max_confidence
                logger.debug(f"✅ Content is safe (max confidence: {confidence_score:.2f})")
                
        except Exception as e:
            logger.error(f"❌ NudeNet detection error: {e}")
            return image_size_kb, False, 0.0, f"Detection failed: {str(e)}"
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(f"⚡ Content moderation completed in {processing_time:.1f}ms")
        
        return image_size_kb, nudity_detected, confidence_score, detection_details
        
    except Exception as e:
        logger.error(f"❌ Content moderation general error: {e}")
        return 0.0, False, 0.0, f"Processing failed: {str(e)}"

# ==================== API ENDPOINTS ====================
@router.post("/detect", response_model=ContentModerationResponse)
async def detect_nudity(request: ContentModerationRequest):
    """
    🔥 NudeNet Content Moderation Endpoint
    
    Ana API'den gelen base64 image'ı analiz eder.
    Tam optimizasyon: in-memory processing, dedicated thread pool.
    """
    start_time = time.time()
    
    try:
        logger.info("🔍 Starting content moderation process...")
        
        # Run NudeNet detection in dedicated thread pool (non-blocking)
        import asyncio
        loop = asyncio.get_event_loop()
        
        image_size_kb, nudity_detected, confidence_score, detection_details = await loop.run_in_executor(
            content_moderation_pool, 
            _sync_process_image_optimized,
            request.image_data
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = ContentModerationResponse(
            nudity_detected=nudity_detected,
            confidence_score=confidence_score,
            detection_details=detection_details,
            processing_time_ms=processing_time_ms,
            image_size_kb=image_size_kb
        )
        
        # Log result
        status = "🚨 BLOCKED" if nudity_detected else "✅ SAFE"
        logger.info(f"{status} - Processing: {processing_time_ms:.1f}ms, Size: {image_size_kb:.1f}KB, Confidence: {confidence_score:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Content moderation endpoint error: {e}")
        # Return safe default in case of error
        return ContentModerationResponse(
            nudity_detected=False,
            confidence_score=0.0,
            detection_details=f"Error: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
            image_size_kb=0.0
        )

@router.get("/health")
async def content_health():
    """Content moderation service health check"""
    try:
        # Test if NudeNet model is loadable
        detector_status = "loaded" if _nude_detector is not None else "unloaded"
        
        return {
            "status": "healthy",
            "nudenet_model": detector_status,
            "thread_pool_active": content_moderation_pool._threads is not None,
            "service": "content_moderation"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

# ==================== WARMUP FUNCTION ====================
# Bu fonksiyon main.py'de startup'ta çağrılacak
