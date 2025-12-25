from datetime import datetime, timezone, timedelta

# Türkiye saati için timezone offset (UTC+3)
TURKEY_TIMEZONE = timezone(timedelta(hours=3))

def turkey_now():
    """Türkiye saati (UTC+3) döndürür"""
    return datetime.now(TURKEY_TIMEZONE)

def turkey_now_naive():
    """Timezone bilgisi olmadan Türkiye saati döndürür (mevcut API uyumluluğu için)"""
    return datetime.now(TURKEY_TIMEZONE).replace(tzinfo=None)

def turkey_date():
    """Türkiye saati ile bugünün tarihini döndürür"""
    return turkey_now().date()

def turkey_fromtimestamp(timestamp):
    """Timestamp'i Türkiye saatine çevirir"""
    return datetime.fromtimestamp(timestamp, TURKEY_TIMEZONE).replace(tzinfo=None)
