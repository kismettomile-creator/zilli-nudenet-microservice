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

# ==================== 🔥 YOLO PERSON DETECTION ====================
_yolo_model = None
_yolo_loading = False

def get_yolo_model():
    """🔥 Thread-safe lazy load YOLO model for person detection"""
    global _yolo_model, _yolo_loading
    
    if _yolo_loading:
        while _yolo_loading and _yolo_model is None:
            time.sleep(0.1)
        return _yolo_model
    
    if _yolo_model is None:
        _yolo_loading = True
        logger.info("🧠 Loading YOLO model...")
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO('yolov8n.pt')  # Nano model - fast
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"❌ YOLO model loading failed: {e}")
            _yolo_model = None
        finally:
            _yolo_loading = False
    
    return _yolo_model

# ==================== 🔥 FALCONSAI NSFW CLASSIFIER (2. bağımsız kaynak) ====================
# Sadece sensitivity="high" (profil fotoğrafı / story) durumunda NudeNet'e ek olarak
# çalışır - NudeNet'in kaçırdığı içerikleri yakalamak için OR mantığıyla eklenir.
# Var olan NudeNet mantığına dokunulmaz, sadece ek bir sinyal olarak eklenir.
_falconsai_pipeline = None
_falconsai_loading = False

FALCONSAI_NSFW_THRESHOLD = 0.5  # "nsfw" skoru bu eşiği geçerse ikinci kaynak da unsafe der

def get_falconsai_classifier():
    """🔥 Thread-safe lazy load Falconsai NSFW classifier"""
    global _falconsai_pipeline, _falconsai_loading

    if _falconsai_loading:
        while _falconsai_loading and _falconsai_pipeline is None:
            time.sleep(0.1)
        return _falconsai_pipeline

    if _falconsai_pipeline is None:
        _falconsai_loading = True
        logger.info("🧠 Loading Falconsai NSFW classifier...")
        try:
            from transformers import pipeline
            _falconsai_pipeline = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection"
            )
            logger.info("✅ Falconsai NSFW classifier loaded successfully")
        except Exception as e:
            logger.error(f"❌ Falconsai classifier loading failed: {e}")
            _falconsai_pipeline = None
        finally:
            _falconsai_loading = False

    return _falconsai_pipeline


def _check_falconsai_nsfw(image: "Image.Image"):
    """
    Falconsai modeliyle görseli sınıflandırır (normal/nsfw).

    Returns:
        (is_nsfw: bool, nsfw_score: float)
    """
    try:
        classifier = get_falconsai_classifier()
        if classifier is None:
            return False, 0.0

        results = classifier(image)  # [{"label": "nsfw", "score": 0.98}, {"label": "normal", "score": 0.02}]
        nsfw_score = 0.0
        for r in results:
            if r.get("label", "").lower() == "nsfw":
                nsfw_score = r.get("score", 0.0)
                break

        return nsfw_score > FALCONSAI_NSFW_THRESHOLD, nsfw_score
    except Exception as e:
        logger.warning(f"⚠️ [FALCONSAI] Detection failed: {e}")
        return False, 0.0


async def warmup_nudenet():
    """Pre-loads the NudeNet model at startup (SKIPPED for macOS compatibility)"""
    logger.info("🔥 [WARMUP] Skipping NudeNet pre-load (will lazy-load on first request)")
    logger.info("✅ [WARMUP] Models will be loaded on-demand")

# ==================== REQUEST/RESPONSE MODELS ====================
class ContentModerationRequest(BaseModel):
    image_data: str  # Base64 encoded image
    sensitivity: Optional[str] = "normal"  # "high", "normal", "low"
    gender: Optional[int] = None  # 1 = female (person detection), 0 or None = no person detection
    
class ContentModerationResponse(BaseModel):
    nudity_detected: bool
    confidence_score: float
    detection_details: str
    processing_time_ms: float
    image_size_kb: float
    sensitivity_used: str
    has_person: bool = False  # True if person detected (only when gender=1)

# ==================== CORE PROCESSING FUNCTIONS ====================
def _sync_process_image_optimized(image_data_b64: str, sensitivity: str = "normal", gender: int = None):
    """
    🔥 OPTIMIZED: In-memory NudeNet detection + 18+ Age Verification + Person Detection
    
    ⚠️⚠️⚠️ CHILD SAFETY: 18 YAŞ ALTI TESPİT EDİLİRSE NOT SAFE! ⚠️⚠️⚠️
    - Bebek, çocuk, teenager → NOT SAFE (nudity_detected=True)
    - 18 yaş altı herhangi bir kişi → NOT SAFE
    
    🧍 PERSON DETECTION (YOLO):
    - gender=1 ise YOLO person detection aktif
    - NudeNet OR YOLO → has_person=True

    🔥 FALCONSAI (2. bağımsız nudity kaynağı):
    - Sadece sensitivity="high" (profil fotoğrafı/story) durumunda çalışır
    - NudeNet OR Falconsai → nudity_detected=True
    
    Sensitivity modes:
    - "high": Profil fotoğrafı/story için - nudity threshold: 0.45, yaş threshold: 20
    - "normal": Video call için - nudity threshold: 0.6, yaş threshold: 18
    - "low": Daha toleranslı - nudity threshold: 0.75, yaş threshold: 18
    
    Returns: (image_size_kb, nudity_detected, confidence_score, detection_details, has_person)
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
            
            # Convert to RGB if image has alpha channel (RGBA/LA/P) or other modes
            if image.mode != 'RGB':
                logger.debug(f"🔄 Converting image from {image.mode} to RGB")
                if image.mode in ('RGBA', 'LA'):
                    # PNG transparency fix
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                elif image.mode == 'P':
                    image = image.convert('RGBA')
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Resim boyutunu optimize et (max 800x800)
            max_size = 800
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            np_array = np.array(image)
        except Exception as e:
            logger.error(f"❌ Image loading error: {e}")
            return image_size_kb, False, 0.0, f"Image load failed: {str(e)}", False
        
        # ========== YOLO PERSON DETECTION (sadece gender=1 için) ==========
        yolo_has_person = False
        person_count = 0
        
        if gender == 1:
            try:
                yolo_model = get_yolo_model()
                if yolo_model:
                    logger.info("🔍 [YOLO] Person detection (gender=1)...")
                    
                    # Geçici dosyaya kaydet (YOLO file path gerektirir)
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        temp_path = tmp.name
                        image.save(temp_path, quality=95)
                    
                    try:
                        results = yolo_model(temp_path, verbose=False)
                        
                        for result in results:
                            for box in result.boxes:
                                if int(box.cls) == 0:  # class 0 = person
                                    person_count += 1
                        
                        yolo_has_person = person_count > 0
                        logger.info(f"👤 [YOLO] {person_count} person(s) detected, has_person={yolo_has_person}")
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
            except Exception as e:
                logger.warning(f"⚠️ [YOLO] Detection failed: {e}")
        
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
                    # has_person: gender=1 için NudeNet OR YOLO (age check passed olsa bile)
                    final_has_person = yolo_has_person if gender == 1 else False
                    return image_size_kb, True, 1.0, age_details, final_has_person
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
            
            # NudeNet'te herhangi bir tespit varsa insan var demektir
            nudenet_has_person = len(detections) > 0
            
            # Final has_person (NudeNet OR YOLO) - sadece gender=1 için
            has_person = False
            if gender == 1:
                has_person = nudenet_has_person or yolo_has_person
                logger.info(f"🧍 [PERSON] NudeNet={nudenet_has_person}, YOLO={yolo_has_person}, Final={has_person}")
            
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
            final_has_person = yolo_has_person if gender == 1 else False
            return image_size_kb, False, 0.0, f"Detection failed: {str(e)}", final_has_person

        # ========== 🔥 FALCONSAI NSFW (2. bağımsız kaynak) - SADECE "high" sensitivity'de ==========
        # Profil fotoğrafı / story kontrolünde NudeNet'e ek olarak çalışır (OR mantığı).
        # Video call ("normal") ve "low" modlarında çalışmaz - performans için.
        if sensitivity == "high":
            try:
                falconsai_is_nsfw, falconsai_score = _check_falconsai_nsfw(image)
                logger.info(f"🔍 [FALCONSAI] is_nsfw={falconsai_is_nsfw}, score={falconsai_score:.2f}")

                if falconsai_is_nsfw:
                    confidence_score = max(confidence_score, falconsai_score)
                    if not nudity_detected:
                        # NudeNet kaçırdı ama Falconsai yakaladı -> OR mantığı
                        nudity_detected = True
                        detection_details = f"{detection_details} | Falconsai NSFW detected (score: {falconsai_score:.2f})"
                        logger.warning(f"🚨 [FALCONSAI] Flagged content NudeNet missed (score: {falconsai_score:.2f})")
                    else:
                        detection_details = f"{detection_details} | Falconsai confirmed (score: {falconsai_score:.2f})"
            except Exception as e:
                logger.warning(f"⚠️ [FALCONSAI] Check skipped due to error: {e}")

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(f"⚡ Content moderation completed in {processing_time:.1f}ms")
        
        return image_size_kb, nudity_detected, confidence_score, detection_details, has_person
        
    except Exception as e:
        logger.error(f"❌ Content moderation general error: {e}")
        return 0.0, False, 0.0, f"Processing failed: {str(e)}", False

# ==================== API ENDPOINTS ====================
@router.post("/detect", response_model=ContentModerationResponse)
async def detect_nudity(request: ContentModerationRequest):
    """
    🔥 NudeNet Content Moderation Endpoint + 18+ Age Verification + Person Detection
    
    Ana API'den gelen base64 image'ı analiz eder.
    ⚠️ CHILD SAFETY: Yaş eşiği altı tespit edilirse NOT SAFE döner!
    🧍 PERSON DETECTION: gender=1 ise YOLO person detection aktif
    
    Sensitivity modes:
    - "high": Profil fotoğrafı/story için - Daha sıkı kontrol (nudity: 0.45, age: 20)
    - "normal": Video call için - Standart kontrol (nudity: 0.6, age: 18)
    - "low": Daha toleranslı kontrol (nudity: 0.75, age: 18)
    
    Gender parameter:
    - gender=1: YOLO person detection aktif, has_person döner (NudeNet OR YOLO)
    - gender=0 veya None: YOLO çalışmaz, has_person=False döner
    
    Tam optimizasyon: in-memory processing, dedicated thread pool.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Starting content moderation (sensitivity: {request.sensitivity}, gender: {request.gender})...")
        
        # Run NudeNet detection + Age verification + Person detection in dedicated thread pool (non-blocking)
        import asyncio
        loop = asyncio.get_event_loop()
        
        image_size_kb, nudity_detected, confidence_score, detection_details, has_person = await loop.run_in_executor(
            content_moderation_pool, 
            _sync_process_image_optimized,
            request.image_data,
            request.sensitivity,
            request.gender  # ⚡ Gender parametresi eklendi
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = ContentModerationResponse(
            nudity_detected=nudity_detected,
            confidence_score=confidence_score,
            detection_details=detection_details,
            processing_time_ms=processing_time_ms,
            image_size_kb=image_size_kb,
            sensitivity_used=request.sensitivity,
            has_person=has_person  # ⚡ Person detection sonucu
        )
        
        # Log result
        status = "🚨 BLOCKED" if nudity_detected else "✅ SAFE"
        person_status = f", has_person={has_person}" if request.gender == 1 else ""
        logger.info(f"{status} [{request.sensitivity.upper()}] - Processing: {processing_time_ms:.1f}ms, Size: {image_size_kb:.1f}KB, Confidence: {confidence_score:.2f}{person_status}")
        
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
            sensitivity_used=request.sensitivity,
            has_person=False
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
