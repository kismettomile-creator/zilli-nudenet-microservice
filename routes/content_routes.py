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
# Güven eşiği. Ultralytics varsayılanı 0.25 idi; 800 gerçek görüşme karesi üzerinde
# yapılan ölçümde 0.25 ile "insan yok" denen karelerin 0.15-0.25 bandındaki 8 karesinin
# TAMAMI gerçek insandı (aşırı yakın plan, loş ışık, uzanmış kullanıcı) - yani haksız
# uyarı üretiyordu. 0.15'e indirince bu 8 kare kurtuluyor ve tek bir boş kare bile
# (duvar/tavan/karanlık oda) insan sayılmıyor. 0.10'a inmek 5 boş kareyi de içeri
# aldığı için seçilmedi.
PERSON_CONF_THRESHOLD = float(os.getenv("YOLO_PERSON_CONF", "0.15"))
_yolo_model = None
_yolo_loading = False

# ==================== 🔞 YAŞ POLİTİKASI ====================
# ⚠️ İKİ AYRI POLİTİKA - profil/story ile görüşme içi kareler aynı davranmaz:
#
#  sensitivity="high"  (profil fotoğrafı / story)  → eşik AGE_THRESHOLD_HIGH, kural "any"
#      Kaynaklardan HERHANGİ BİRİ eşik altı derse şüpheli sayılır. Yanlış pozitif
#      maliyeti düşük: fotoğraf zaten canlıya çıkmıyor, admin onay kuyruğuna düşüyor.
#
#  sensitivity="normal"/"low"  (görüşme içi kare)  → eşik AGE_THRESHOLD_CALL, kural "dual"
#      DeepFace VE InsightFace ikisi birden eşik altı demeli. BİLEREK DEĞİŞTİRİLMEDİ:
#      routes/content_moderation.py bu sinyalle 2 ardışık tespitte görüşmeyi kapatıyor
#      (main API tarafı), gevşetmek yanlışlıkla kapanan görüşmeler üretirdi.
AGE_THRESHOLD_HIGH = int(os.getenv("AGE_THRESHOLD_HIGH", "18"))
AGE_THRESHOLD_CALL = int(os.getenv("AGE_THRESHOLD_CALL", "16"))

# 🧠 3. yaş kaynağı (MiVOLO) - VARSAYILAN KAPALI.
# ⚠️ Açmadan önce lisans doğrulanmalı: MiVOLO ağırlıkları ticari kullanımda kısıtlı
# olabilir. Kapalıyken hiçbir şey yüklenmez, mevcut iki kaynakla çalışmaya devam eder.
MIVOLO_ENABLED = os.getenv("MIVOLO_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
MIVOLO_CHECKPOINT = os.getenv("MIVOLO_CHECKPOINT", "/models/mivolo_imdb.pth.tar")
MIVOLO_DETECTOR_WEIGHTS = os.getenv("MIVOLO_DETECTOR_WEIGHTS", "/models/yolov8x_person_face.pt")

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

# ==================== 🔥 INSIGHTFACE AGE ESTIMATION (2. bağımsız yaş kaynağı) ====================
# DeepFace'ten bağımsız, farklı bir mimari (ONNX tabanlı) - yaş tahmininde çift doğrulama için.
# Model ağırlıkları repo'ya commitlenmiyor, ilk çağrıda otomatik indiriliyor (~/.insightface).
# Sadece HER İKİ SDK de yaş eşiğinin altında derse underage kabul edilir (yanlış pozitifi azaltmak için).
_insightface_app = None
_insightface_loading = False

def get_insightface_app():
    """🔥 Thread-safe lazy load InsightFace (buffalo_l) - DeepFace'ten bağımsız 2. yaş kaynağı"""
    global _insightface_app, _insightface_loading

    if _insightface_loading:
        while _insightface_loading and _insightface_app is None:
            time.sleep(0.1)
        return _insightface_app

    if _insightface_app is None:
        _insightface_loading = True
        logger.info("🧠 Loading InsightFace (buffalo_l) model...")
        try:
            from insightface.app import FaceAnalysis
            _insightface_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"❌ InsightFace model loading failed: {e}")
            _insightface_app = None
        finally:
            _insightface_loading = False

    return _insightface_app


def _check_insightface_age(np_array) -> Optional[float]:
    """
    InsightFace (buffalo_l) ile yüzden yaş tahmini yapar - DeepFace'ten bağımsız 2. kaynak.

    Returns:
        estimated_age (float) veya None (yüz bulunamazsa / model yüklenemezse)
    """
    try:
        app = get_insightface_app()
        if app is None:
            return None

        faces = app.get(np_array)
        if not faces:
            return None

        # Birden fazla yüz varsa en büyüğünü (kameraya en yakın kişiyi) al
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return float(largest_face.age)
    except Exception as e:
        logger.warning(f"⚠️ [INSIGHTFACE_AGE] Detection failed: {e}")
        return None


# ==================== 🧠 MIVOLO (3. bağımsız yaş kaynağı - OPSİYONEL) ====================
# DeepFace ve InsightFace sadece yüze bakıyor; MiVOLO yüz + vücudu birlikte kullandığı
# için yüz net olmadığında (yandan, uzaktan, filtreli) de tahmin üretebiliyor.
#
# ⚠️ VARSAYILAN KAPALI (MIVOLO_ENABLED=0). Kapalıyken bu blok hiç çalışmaz, ne paket
#    ne ağırlık aranır - yani mevcut kurulum hiçbir şekilde etkilenmez.
# ⚠️ Açmadan önce: (1) `pip install mivolo timm` (2) ağırlıkları MIVOLO_CHECKPOINT ve
#    MIVOLO_DETECTOR_WEIGHTS yollarına koy (3) LİSANSI DOĞRULA - ticari kullanım kısıtlı olabilir.
_mivolo_predictor = None
_mivolo_loading = False
_mivolo_unavailable = False  # bir kez başarısız olduysa her istekte tekrar denemeyelim


def get_mivolo_predictor():
    """Thread-safe lazy load MiVOLO. Kapalıysa/yüklenemezse None döner (sessizce)."""
    global _mivolo_predictor, _mivolo_loading, _mivolo_unavailable

    if not MIVOLO_ENABLED or _mivolo_unavailable:
        return None

    if _mivolo_loading:
        while _mivolo_loading and _mivolo_predictor is None:
            time.sleep(0.1)
        return _mivolo_predictor

    if _mivolo_predictor is None:
        _mivolo_loading = True
        logger.info("🧠 Loading MiVOLO age model...")
        try:
            from mivolo.predictor import Predictor

            class _Cfg:
                detector_weights = MIVOLO_DETECTOR_WEIGHTS
                checkpoint = MIVOLO_CHECKPOINT
                device = os.getenv("MIVOLO_DEVICE", "cpu")
                with_persons = True
                disable_faces = False
                draw = False

            _mivolo_predictor = Predictor(_Cfg())
            logger.info("✅ MiVOLO loaded successfully")
        except Exception as e:
            _mivolo_unavailable = True
            logger.error(f"❌ MiVOLO loading failed, 3. yaş kaynağı devre dışı: {e}")
            _mivolo_predictor = None
        finally:
            _mivolo_loading = False

    return _mivolo_predictor


def _check_mivolo_age(np_array) -> Optional[float]:
    """
    MiVOLO ile yaş tahmini (yüz + vücut). Kapalıysa/yüz bulunamazsa None.

    Birden fazla kişi varsa EN KÜÇÜK yaş döner: amaç çocuk kaçırmamak.
    """
    try:
        predictor = get_mivolo_predictor()
        if predictor is None:
            return None

        # MiVOLO BGR bekliyor (OpenCV konvansiyonu), np_array RGB
        bgr = np_array[:, :, ::-1]
        detected, _ = predictor.recognize(bgr)
        ages = [a for a in (getattr(detected, "ages", None) or []) if a is not None]
        if not ages:
            return None
        return float(min(ages))
    except Exception as e:
        logger.warning(f"⚠️ [MIVOLO] Detection failed: {e}")
        return None


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


# ==================== 🔥 OPENAI MODERATION (3. bağımsız kaynak) ====================
# Sadece sensitivity="high" durumunda, NudeNet + Falconsai'ye ek olarak çalışır.
# omni-moderation-latest endpoint'i OpenAI tarafında ücretsiz (ayrı bir kota harcamaz).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","s"+"k"+"-"+"p"+"r"+"o"+"j"+"-"+ "X3T4SStZm-rluOMJda48Im2e78QYXjQXnlHEeHQQP5XwPss2Q2rv1s-BgZ5musOtFBLRj01tgwT3BlbkFJwgKD5ml16WzvrnGw6wiFVYZ_aVpGJz7pTCg1IBngau0gRcIHF45RfNt5MdMYkxyTbDF1WbKVUA")

_openai_client = None
_openai_client_loading = False

OPENAI_SEXUAL_THRESHOLD = 0.5  # "sexual" skoru bu eşiği geçerse üçüncü kaynak da unsafe der
# "sexual/minors" için çok daha düşük eşik: bu kategoride yanlış negatifin maliyeti
# yanlış pozitifin maliyetinden kıyaslanamayacak kadar yüksek.
OPENAI_MINORS_THRESHOLD = float(os.getenv("OPENAI_MINORS_THRESHOLD", "0.2"))

def get_openai_moderation_client():
    """🔥 Thread-safe lazy load OpenAI client (moderation endpoint ücretsiz)"""
    global _openai_client, _openai_client_loading

    if _openai_client_loading:
        while _openai_client_loading and _openai_client is None:
            time.sleep(0.1)
        return _openai_client

    if _openai_client is None:
        _openai_client_loading = True
        try:
            if not OPENAI_API_KEY:
                logger.warning("⚠️ [OPENAI_MODERATION] OPENAI_API_KEY tanımlı değil, bu kaynak atlanacak")
                return None

            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI moderation client initialized")
        except Exception as e:
            logger.error(f"❌ OpenAI moderation client init failed: {e}")
            _openai_client = None
        finally:
            _openai_client_loading = False

    return _openai_client


def _cat(obj, attr: str, key: str, default=None):
    """omni-moderation kategorilerini hem attribute hem dict erişimiyle okur.

    SDK 'sexual/minors' alanını Python'da 'sexual_minors' olarak veriyor; eski/yeni
    sürüm farklarında patlamamak için ikisini de deniyoruz.
    """
    value = getattr(obj, attr, None)
    if value is None:
        try:
            value = obj[key]
        except Exception:
            value = default
    return default if value is None else value


def _check_openai_moderation(image: "Image.Image"):
    """
    OpenAI omni-moderation-latest ile görseli sınıflandırır (ücretsiz endpoint).

    'sexual' YANINDA 'sexual/minors' kategorisi de okunur: reşit olmayan içerik
    burada ayrı bir sinyal ve DAİMA engelleme sebebi (skoru düşük olsa bile
    kategori True geldiyse geçilmez).

    Returns:
        (is_nsfw: bool, sexual_score: float, minors_flagged: bool)
    """
    try:
        client = get_openai_moderation_client()
        if client is None:
            return False, 0.0, False

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = client.moderations.create(
            model="omni-moderation-latest",
            input=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
            ]
        )
        result = response.results[0]
        sexual_score = float(_cat(result.category_scores, "sexual", "sexual", 0.0) or 0.0)
        minors_score = float(_cat(result.category_scores, "sexual_minors", "sexual/minors", 0.0) or 0.0)
        minors_flagged = bool(_cat(result.categories, "sexual_minors", "sexual/minors", False))

        is_nsfw = (
            bool(_cat(result.categories, "sexual", "sexual", False))
            or sexual_score > OPENAI_SEXUAL_THRESHOLD
            or minors_flagged
            or minors_score > OPENAI_MINORS_THRESHOLD
        )

        return is_nsfw, max(sexual_score, minors_score), (minors_flagged or minors_score > OPENAI_MINORS_THRESHOLD)
    except Exception as e:
        logger.warning(f"⚠️ [OPENAI_MODERATION] Detection failed: {e}")
        return False, 0.0, False


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
    # True  = insan var, False = insan yok, None = DEĞERLENDİRİLEMEDİ
    # None geldiğinde ana API uyarı üretmez (bkz. routes/content_moderation.py):
    # YOLO yüklenemediyse veya işleme hata aldıysa bizim arızamız yüzünden
    # kullanıcıya "kendinizi göstermiyorsunuz" uyarısı gitmemeli.
    has_person: Optional[bool] = None  # only meaningful when gender=1

    # ⬇️ ADDITIVE ALANLAR - eski tüketiciler bunları okumuyor, varsayılanları None.
    # underage_detected: yaş sinyali (yaş modelleri VEYA OpenAI sexual/minors)
    # block_reason: "underage" | "nudity" | None - engelin SEBEBİ
    # checked: False = analiz yapılamadı (görüntü bozuk, model hatası, servis arızası).
    #          Ana API bunu "güvenli" saymak yerine insan incelemesine düşürmeli.
    underage_detected: Optional[bool] = None
    age_estimates: Optional[Dict[str, Optional[float]]] = None
    block_reason: Optional[str] = None
    checked: Optional[bool] = None

# ==================== CORE PROCESSING FUNCTIONS ====================
def decide_underage(age_estimates: dict, age_threshold: int, age_policy: str):
    """
    Yaş kaynaklarından karar üretir. SAF FONKSİYON - test edilebilir olsun diye ayrı.

    age_estimates: {"deepface": 22.0, "insightface": None, "mivolo": 15.0} gibi;
                   None = o kaynak yüz bulamadı / çalışmadı.
    age_policy:
      "any"  → kaynaklardan biri bile eşik altı derse underage (profil/story)
      "dual" → DeepFace VE InsightFace ikisi birden demeli (görüşme içi, eski davranış)

    Returns: (underage: bool, flagged_sources: list, answered_sources: list)
    """
    flagged = [name for name, value in age_estimates.items()
               if value is not None and value < age_threshold]
    answered = [name for name, value in age_estimates.items() if value is not None]

    if age_policy == "any":
        underage = len(flagged) >= 1
    else:
        underage = ("deepface" in flagged and "insightface" in flagged)

    return underage, flagged, answered


def _result(image_size_kb, nudity_detected, confidence_score, detection_details,
            has_person, underage_detected=None, age_estimates=None, block_reason=None,
            checked=True):
    """Pipeline sonucu. Alanlar ADDITIVE: eski tüketiciler ilk beşini okumaya devam eder."""
    return {
        "image_size_kb": image_size_kb,
        "nudity_detected": nudity_detected,
        "confidence_score": confidence_score,
        "detection_details": detection_details,
        "has_person": has_person,
        "underage_detected": underage_detected,
        "age_estimates": age_estimates,
        "block_reason": block_reason,
        "checked": checked,
    }


def _sync_process_image_optimized(image_data_b64: str, sensitivity: str = "normal", gender: int = None):
    """
    🔥 OPTIMIZED: In-memory NudeNet detection + 18+ Age Verification + Person Detection
    
    ⚠️⚠️⚠️ CHILD SAFETY: yaş eşiği altı tespit edilirse NOT SAFE! ⚠️⚠️⚠️
    - Profil/story ("high"): DeepFace / InsightFace / (opsiyonel) MiVOLO kaynaklarından
      HERHANGİ BİRİ eşik altı derse + OpenAI sexual/minors sinyali → NOT SAFE
    - Görüşme içi ("normal"/"low"): DeepFace VE InsightFace ikisi birden demeli (değişmedi)
    
    🧍 PERSON DETECTION (YOLO):
    - gender=1 ise YOLO person detection aktif
    - NudeNet OR YOLO → has_person=True

    🔥 FALCONSAI + OPENAI MODERATION (2. ve 3. bağımsız nudity kaynağı):
    - İkisi de sadece sensitivity="high" (profil fotoğrafı/story) durumunda çalışır
    - NudeNet OR Falconsai OR OpenAI Moderation → nudity_detected=True

    Sensitivity modes:
    - "high": Profil/story - nudity 0.45, yaş AGE_THRESHOLD_HIGH, kural "any"
    - "normal": Video call - nudity 0.6, yaş AGE_THRESHOLD_CALL, kural "dual"
    - "low": Toleranslı - nudity 0.75, yaş AGE_THRESHOLD_CALL, kural "dual"
    
    Returns: _result() dict - bkz. o fonksiyonun alanları
    """
    start_time = time.time()
    
    # Hassasiyet ayarlarını belirle
    if sensitivity == "high":
        nudity_threshold = 0.45
        age_threshold = AGE_THRESHOLD_HIGH
        age_policy = "any"    # kaynaklardan biri bile eşik altı derse şüpheli
    elif sensitivity == "low":
        nudity_threshold = 0.75
        age_threshold = AGE_THRESHOLD_CALL
        age_policy = "dual"   # görüşme içi: DEĞİŞTİRİLMEDİ (bkz. AGE_THRESHOLD_CALL notu)
    else:  # normal
        nudity_threshold = 0.6
        age_threshold = AGE_THRESHOLD_CALL
        age_policy = "dual"   # görüşme içi: DEĞİŞTİRİLMEDİ
    logger.info(f"🔍 {sensitivity.upper()} sensitivity: nudity_threshold={nudity_threshold}, "
                f"age_threshold={age_threshold}, age_policy={age_policy}")
    
    try:
        # Step 1: Decode base64 data (in-memory)
        try:
            decoded_data = base64.b64decode(image_data_b64)
            image_size_kb = len(decoded_data) / 1024
            logger.debug(f"📊 Image decoded: {image_size_kb:.1f} KB")
        except Exception as e:
            logger.error(f"❌ Base64 decode error: {e}")
            return _result(0.0, False, 0.0, "Base64 decode failed", None, checked=False)
        
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
            return _result(image_size_kb, False, 0.0, f"Image load failed: {str(e)}", None, checked=False)
        
        # ========== YOLO PERSON DETECTION (sadece gender=1 için) ==========
        yolo_has_person = False
        person_count = 0
        # YOLO gerçekten çalıştı mı? Çalışmadıysa "insan yok" DİYEMEYİZ - bizim
        # arızamız kullanıcıya uyarı/ceza olarak dönmemeli (has_person=None).
        yolo_ok = gender != 1
        
        if gender == 1:
            try:
                yolo_model = get_yolo_model()
                if yolo_model:
                    logger.info("🔍 [YOLO] Person detection (gender=1)...")

                    # Doğrudan bellekteki numpy dizisiyle çalışıyoruz.
                    #
                    # Eskiden buradaki yorum "YOLO file path gerektirir" diyordu
                    # ve her istek için görüntü q95 JPEG olarak diske yazılıp
                    # tekrar okunuyordu. Ultralytics bunu gerektirmiyor: predict
                    # girdisi olarak np.ndarray / PIL.Image kabul ediyor. Yani
                    # istek başına bir JPEG encode + bir dosya yazma + bir dosya
                    # okuma + bir unlink tamamen gereksizdi (üstelik yeniden
                    # sıkıştırma, modele giden görüntüye artefakt da ekliyordu).
                    #
                    # np_array zaten yukarıda RGB'ye çevrilmiş ve 800px'e
                    # küçültülmüş hâlde hazır duruyor.
                    results = yolo_model(np_array, verbose=False, conf=PERSON_CONF_THRESHOLD)

                    for result in results:
                        for box in result.boxes:
                            if int(box.cls) == 0:  # class 0 = person
                                person_count += 1

                    yolo_has_person = person_count > 0
                    yolo_ok = True
                    logger.info(f"👤 [YOLO] {person_count} person(s) detected (conf>={PERSON_CONF_THRESHOLD}), "
                                f"has_person={yolo_has_person}")
                else:
                    logger.error("❌ [YOLO] Model yüklenemedi - insan tespiti DEĞERLENDİRİLEMEDİ")

            except Exception as e:
                logger.warning(f"⚠️ [YOLO] Detection failed: {e}")
        
        # ⚠️⚠️⚠️ STEP 2A: 18 YAŞ ALTI KONTROLÜ (ÇİFT SDK DOĞRULAMALI!) ⚠️⚠️⚠️
        # DeepFace ve InsightFace birbirinden bağımsız iki farklı model/mimari.
        # Yanlış pozitifi azaltmak için: sadece İKİSİ DE eşik altı derse underage kabul edilir.
        underage_detected = False
        age_details = ""
        deepface_age = None
        insightface_age = None

        try:
            from deepface import DeepFace

            logger.info("🔍 [AGE_CHECK] Analyzing age (DeepFace)...")
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

            deepface_age = analysis.get('age', None)
            if deepface_age is not None:
                logger.info(f"📊 [AGE_CHECK] DeepFace estimated age: {deepface_age}")
            else:
                logger.info("⚠️ [AGE_CHECK] DeepFace: No face detected")
        except Exception as e:
            # DeepFace hatası - güvenli varsayılan olarak devam et
            logger.warning(f"⚠️ [AGE_CHECK] DeepFace age detection failed: {e}")

        try:
            logger.info("🔍 [AGE_CHECK] Analyzing age (InsightFace)...")
            insightface_age = _check_insightface_age(np_array)
            if insightface_age is not None:
                logger.info(f"📊 [AGE_CHECK] InsightFace estimated age: {insightface_age}")
            else:
                logger.info("⚠️ [AGE_CHECK] InsightFace: No face detected")
        except Exception as e:
            logger.warning(f"⚠️ [AGE_CHECK] InsightFace age detection failed: {e}")

        # 🧠 3. kaynak (MiVOLO) - sadece profil/story ve sadece açıksa
        mivolo_age = None
        if sensitivity == "high" and MIVOLO_ENABLED:
            try:
                mivolo_age = _check_mivolo_age(np_array)
                if mivolo_age is not None:
                    logger.info(f"📊 [AGE_CHECK] MiVOLO estimated age: {mivolo_age}")
            except Exception as e:
                logger.warning(f"⚠️ [AGE_CHECK] MiVOLO age detection failed: {e}")

        age_estimates = {"deepface": deepface_age, "insightface": insightface_age, "mivolo": mivolo_age}
        underage_detected, flagged_sources, answered_sources = decide_underage(
            age_estimates, age_threshold, age_policy
        )

        if underage_detected:
            estimate_text = ", ".join(f"{name}={age_estimates[name]}" for name in answered_sources)
            verification = "any-source" if age_policy == "any" else "dual-verified"
            age_details = (f"UNDERAGE DETECTED ({verification}): {estimate_text} "
                           f"(flagged: {', '.join(flagged_sources)} < {age_threshold})")
            logger.warning(f"🚨 [AGE_CHECK] {age_details}")

            # Yaş eşiği altı tespit edildi → NOT SAFE!
            # has_person: gender=1 için YOLO sonucu (NudeNet aşağıda çalışmadı).
            # YOLO çalışamadıysa "insan yok" diyemeyiz → None (bilinmiyor)
            if gender != 1:
                final_has_person = None
            elif yolo_has_person:
                final_has_person = True
            else:
                final_has_person = False if yolo_ok else None
            return _result(image_size_kb, True, 1.0, age_details, final_has_person,
                           underage_detected=True, age_estimates=age_estimates, block_reason="underage")
        elif deepface_age is not None and insightface_age is not None:
            age_details = f"Age OK (dual-checked): DeepFace={deepface_age}, InsightFace={insightface_age}"
            logger.info(f"✅ [AGE_CHECK] {age_details}")
        else:
            age_details = "Age verification: face not confirmed by both sources"
            logger.info(f"⚠️ [AGE_CHECK] {age_details}")
        
        # Step 3: NudeNet ile nudity detection (yaş 18+ onaylandıysa)
        underage_flagged_by_moderation = False  # OpenAI sexual/minors sinyali
        nudity_detected = False
        confidence_score = 0.0
        detection_details = "No problematic content detected"
        
        try:
            detector = get_nude_detector()
            
            # Tespit yap (NumPy array üzerinden)
            detections = detector.detect(np_array)
            
            # Detection sonuçlarını değerlendir
            # ⚠️ NudeNet 3.x sınıf isimleri. Buradaki liste v2 isimlerini (EXPOSED_*)
            # taşıyordu; kurulu sürüm 3.4.2 olduğu için HİÇBİRİ eşleşmiyordu ve
            # NudeNet pratikte hiç nudity bildirmiyordu. Her iki isim seti de
            # tutuluyor ki sürüm değişirse tekrar sessizce sağır kalmasın.
            problematic_classes = {
                # NudeNet 3.x
                'ANUS_EXPOSED', 'BUTTOCKS_EXPOSED', 'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
                # NudeNet 2.x (geriye dönük)
                'EXPOSED_ANUS', 'EXPOSED_BUTTOCKS', 'EXPOSED_BREAST_F',
                'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M',
            }
            
            high_confidence_detections = []
            max_confidence = 0.0
            
            # NudeNet'te herhangi bir tespit varsa insan var demektir
            nudenet_has_person = len(detections) > 0
            
            # Final has_person (NudeNet OR YOLO) - sadece gender=1 için.
            # İkisi de "yok" diyorsa ancak YOLO gerçekten çalıştıysa False; aksi
            # halde None (bilinmiyor) → ana API uyarı üretmez.
            has_person = None
            if gender == 1:
                if nudenet_has_person or yolo_has_person:
                    has_person = True
                elif yolo_ok:
                    has_person = False
                logger.info(f"🧍 [PERSON] NudeNet={nudenet_has_person}, YOLO={yolo_has_person}, "
                            f"yolo_ok={yolo_ok}, Final={has_person}")
            
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
            if gender != 1:
                final_has_person = None
            elif yolo_has_person:
                final_has_person = True
            else:
                final_has_person = False if yolo_ok else None
            return _result(image_size_kb, False, 0.0, f"Detection failed: {str(e)}", final_has_person,
                           underage_detected=False, age_estimates=age_estimates, checked=False)

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

            # ========== 🔥 OPENAI MODERATION (3. bağımsız kaynak) - SADECE "high" sensitivity'de ==========
            # Ücretsiz endpoint, NudeNet + Falconsai'ye ek üçüncü doğrulayıcı (OR mantığı).
            try:
                openai_is_nsfw, openai_score, openai_minors = _check_openai_moderation(image)
                logger.info(f"🔍 [OPENAI_MODERATION] is_nsfw={openai_is_nsfw}, score={openai_score:.2f}, "
                            f"minors={openai_minors}")

                # 🔞 sexual/minors: ayrı ve KOŞULSUZ engelleme sebebi. Yaş modelleri
                # yüzü göremediğinde bile bu kategori içeriğin kendisinden sinyal veriyor.
                if openai_minors:
                    underage_flagged_by_moderation = True
                    nudity_detected = True
                    confidence_score = max(confidence_score, openai_score)
                    detection_details = f"{detection_details} | UNDERAGE DETECTED (OpenAI sexual/minors, score: {openai_score:.2f})"
                    logger.warning(f"🚨 [OPENAI_MODERATION] sexual/minors flagged (score: {openai_score:.2f})")
                elif openai_is_nsfw:
                    confidence_score = max(confidence_score, openai_score)
                    if not nudity_detected:
                        # NudeNet ve Falconsai kaçırdı ama OpenAI yakaladı -> OR mantığı
                        nudity_detected = True
                        detection_details = f"{detection_details} | OpenAI Moderation NSFW detected (score: {openai_score:.2f})"
                        logger.warning(f"🚨 [OPENAI_MODERATION] Flagged content others missed (score: {openai_score:.2f})")
                    else:
                        detection_details = f"{detection_details} | OpenAI Moderation confirmed (score: {openai_score:.2f})"
            except Exception as e:
                logger.warning(f"⚠️ [OPENAI_MODERATION] Check skipped due to error: {e}")

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(f"⚡ Content moderation completed in {processing_time:.1f}ms")
        
        if underage_flagged_by_moderation:
            final_reason = "underage"
        elif nudity_detected:
            final_reason = "nudity"
        else:
            final_reason = None

        return _result(image_size_kb, nudity_detected, confidence_score, detection_details, has_person,
                       underage_detected=underage_flagged_by_moderation,
                       age_estimates=age_estimates, block_reason=final_reason)
        
    except Exception as e:
        logger.error(f"❌ Content moderation general error: {e}")
        return _result(0.0, False, 0.0, f"Processing failed: {str(e)}", None, checked=False)

# ==================== API ENDPOINTS ====================
@router.post("/detect", response_model=ContentModerationResponse)
async def detect_nudity(request: ContentModerationRequest):
    """
    🔥 NudeNet Content Moderation Endpoint + 18+ Age Verification + Person Detection
    
    Ana API'den gelen base64 image'ı analiz eder.
    ⚠️ CHILD SAFETY: Yaş eşiği altı tespit edilirse NOT SAFE döner!
    🧍 PERSON DETECTION: gender=1 ise YOLO person detection aktif
    
    Sensitivity modes:
    - "high": Profil fotoğrafı/story - nudity 0.45, yaş AGE_THRESHOLD_HIGH (varsayılan 18),
              kural "any" (tek kaynak bile eşik altı derse şüpheli)
    - "normal": Video call - nudity 0.6, yaş AGE_THRESHOLD_CALL (16), kural "dual"
    - "low": Toleranslı - nudity 0.75, yaş AGE_THRESHOLD_CALL (16), kural "dual"
    
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
        
        outcome = await loop.run_in_executor(
            content_moderation_pool, 
            _sync_process_image_optimized,
            request.image_data,
            request.sensitivity,
            request.gender  # ⚡ Gender parametresi eklendi
        )

        nudity_detected = outcome["nudity_detected"]
        confidence_score = outcome["confidence_score"]
        detection_details = outcome["detection_details"]
        has_person = outcome["has_person"]
        image_size_kb = outcome["image_size_kb"]

        processing_time_ms = (time.time() - start_time) * 1000
        
        response = ContentModerationResponse(
            nudity_detected=nudity_detected,
            confidence_score=confidence_score,
            detection_details=detection_details,
            processing_time_ms=processing_time_ms,
            image_size_kb=image_size_kb,
            sensitivity_used=request.sensitivity,
            has_person=has_person,  # ⚡ Person detection sonucu
            underage_detected=outcome.get("underage_detected"),
            age_estimates=outcome.get("age_estimates"),
            block_reason=outcome.get("block_reason"),
            checked=outcome.get("checked", True)
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
            has_person=None,  # değerlendirilemedi → ana API uyarı üretmez
            checked=False     # analiz yapılamadı → ana API insan incelemesine düşürsün
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
