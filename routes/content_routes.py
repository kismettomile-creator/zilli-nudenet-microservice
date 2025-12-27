"""
🔥 Content Moderation Routes - NudeNet + Age Detection Processing
Ana API'den ayrıştırılan ağır NudeNet işlemleri + 18+ yaş kontrolü
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
    sensitivity: Optional[str] = "normal"  # "high", "normal", "low"
    
class ContentModerationResponse(BaseModel):
    nudity_detected: bool
    confidence_score: float
    detection_details: str
    processing_time_ms: float
    image_size_kb: float
    sensitivity_used: str

# ==================== CORE PROCESSING FUNCTIONS ====================
def _sync_process_image_optimized(image_data_b64: str, sensitivity: str = "normal"):
    """
    🔥 OPTIMIZED: In-memory NudeNet detection + 18+ Age Verification
    
    ⚠️⚠️⚠️ CHILD SAFETY: 18 YAŞ ALTI TESPİT EDİLİRSE NOT SAFE! ⚠️⚠️⚠️
    - Bebek, çocuk, teenager → NOT SAFE (nudity_detected=True)
    - 18 yaş altı herhangi bir kişi → NOT SAFE
    
    Sensitivity modes:
    - "high": Profil fotoğrafı/story için - nudity threshold: 0.45, yaş threshold: 20
    - "normal": Video call için - nudity threshold: 0.6, yaş threshold: 18
    - "low": Daha toleranslı - nudity threshold: 0.75, yaş threshold: 18
    """
    start_time = time.time()
    
    # Hassasiyet ayarlarını belirle
    if sensitivity == "high":
        nudity_threshold = 0.45
        age_threshold = 16  # Profil/story için: 16 yaş altı ret
        logger.info("🔍 HIGH sensitivity mode: nudity_threshold=0.45, age_threshold=16")
    elif sensitivity == "low":
        nudity_threshold = 0.75
        age_threshold = 16
        logger.info("🔍 LOW sensitivity mode: nudity_threshold=0.75, age_threshold=16")
    else:  # normal
        nudity_threshold = 0.6
        age_threshold = 16  # Video call için: 16 yaş altı ret
        logger.info("🔍 NORMAL sensitivity mode: nudity_threshold=0.6, age_threshold=16")
    
    try:
        # Step 1: Decode base64 data (in-memory)
        try:
            decoded_data = base64.b64decode(image_data_b64)
            image_size_kb = len(decoded_data) / 1024
            logger.debug(f"📊 Image decoded: {image_size_kb:.1f} KB")
        except Exception as e:
            logger.error(f"❌ Base64 decode error: {e}")
            return 0.0, False, 0.0, "Base64 decode failed"
        
        # Step 2: PIL Image oluştur (hem NudeNet hem DeepFace için)
        try:
            image = Image.open(io.BytesIO(decoded_data))
            
            # Convert to RGB if image has alpha channel (RGBA/LA) or other modes
            if image.mode != 'RGB':
                logger.debug(f"🔄 Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resim boyutunu optimize et (max 800x800)
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            np_array = np.array(image)
        except Exception as e:
            logger.error(f"❌ Image loading error: {e}")
            return image_size_kb, False, 0.0, f"Image load failed: {str(e)}"
        
        # ⚠️⚠️⚠️ STEP 2A: 18 YAŞ ALTI KONTROLÜ (ÖNCELİKLİ!) ⚠️⚠️⚠️
        underage_detected = False
        age_details = ""
        
        try:
            from deepface import DeepFace
            
            # DeepFace ile yaş tahmini yap
            logger.info("🔍 [AGE_CHECK] Analyzing age...")
            
            # Yüz tespit et ve yaş tahmin et
            analysis = DeepFace.analyze(
                img_path=np_array,
                actions=['age'],
                enforce_detection=False,  # Yüz tespit edilemezse hata verme
                detector_backend='opencv',  # Hızlı detector
                silent=True
            )
            
            # Analysis sonucunu kontrol et (list veya dict olabilir)
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}
            
            estimated_age = analysis.get('age', None)
            
            if estimated_age is not None:
                logger.info(f"📊 [AGE_CHECK] Estimated age: {estimated_age}")
                
                # ⚠️ CRITICAL: Yaş kontrolü (hassasiyet moduna göre)
                if estimated_age < age_threshold:
                    underage_detected = True
                    age_details = f"UNDERAGE DETECTED: Estimated age {estimated_age} (< {age_threshold})"
                    logger.warning(f"🚨 [AGE_CHECK] {age_details}")
                    
                    # Yaş eşiği altı tespit edildi → NOT SAFE!
                    return image_size_kb, True, 1.0, age_details
                else:
                    logger.info(f"✅ [AGE_CHECK] Age verification passed: {estimated_age} >= {age_threshold}")
                    age_details = f"Age OK: {estimated_age}"
            else:
                # Yüz tespit edilemedi, yaş tahmin edilemedi
                logger.info("⚠️ [AGE_CHECK] No face detected or age could not be estimated")
                age_details = "Age verification: No face detected"
                
        except Exception as e:
            # DeepFace hatası - güvenli varsayılan olarak devam et
            logger.warning(f"⚠️ [AGE_CHECK] Age detection failed: {e}")
            age_details = f"Age verification failed: {str(e)}"
            # Yaş kontrolü başarısız oldu ama nudity kontrolüne devam et
        
        # Step 3: NudeNet ile nudity detection (yaş 18+ onaylandıysa)
        nudity_detected = False
        confidence_score = 0.0
        detection_details = "No problematic content detected"
        
        try:
            detector = get_nude_detector()
            
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
                
                # Hassasiyet moduna göre threshold kullan
                if class_name in problematic_classes and confidence > nudity_threshold:
                    high_confidence_detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
            
            if high_confidence_detections:
                nudity_detected = True
                confidence_score = max_confidence
                detection_details = f"Nudity: {', '.join([d['class'] for d in high_confidence_detections])}"
                if age_details:
                    detection_details = f"{age_details} | {detection_details}"
                logger.info(f"🚨 Nudity detected: {detection_details} (confidence: {confidence_score:.2f})")
            else:
                confidence_score = max_confidence
                detection_details = age_details if age_details else f"Content is safe (max confidence: {confidence_score:.2f})"
                logger.debug(f"✅ {detection_details}")
                
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
    🔥 NudeNet Content Moderation Endpoint + 18+ Age Verification
    
    Ana API'den gelen base64 image'ı analiz eder.
    ⚠️ CHILD SAFETY: Yaş eşiği altı tespit edilirse NOT SAFE döner!
    
    Sensitivity modes:
    - "high": Profil fotoğrafı/story için - Daha sıkı kontrol (nudity: 0.45, age: 20)
    - "normal": Video call için - Standart kontrol (nudity: 0.6, age: 18)
    - "low": Daha toleranslı kontrol (nudity: 0.75, age: 18)
    
    Tam optimizasyon: in-memory processing, dedicated thread pool.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Starting content moderation process (sensitivity: {request.sensitivity})...")
        
        # Run NudeNet detection + Age verification in dedicated thread pool (non-blocking)
        import asyncio
        loop = asyncio.get_event_loop()
        
        image_size_kb, nudity_detected, confidence_score, detection_details = await loop.run_in_executor(
            content_moderation_pool, 
            _sync_process_image_optimized,
            request.image_data,
            request.sensitivity  # ⚡ Hassasiyet parametresi eklendi
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = ContentModerationResponse(
            nudity_detected=nudity_detected,
            confidence_score=confidence_score,
            detection_details=detection_details,
            processing_time_ms=processing_time_ms,
            image_size_kb=image_size_kb,
            sensitivity_used=request.sensitivity  # ⚡ Kullanılan hassasiyet
        )
        
        # Log result
        status = "🚨 BLOCKED" if nudity_detected else "✅ SAFE"
        logger.info(f"{status} [{request.sensitivity.upper()}] - Processing: {processing_time_ms:.1f}ms, Size: {image_size_kb:.1f}KB, Confidence: {confidence_score:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Content moderation endpoint error: {e}")
        # Return safe default in case of error
        return ContentModerationResponse(
            nudity_detected=False,
            confidence_score=0.0,
            detection_details=f"Error: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
            image_size_kb=0.0,
            sensitivity_used=request.sensitivity
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
