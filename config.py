"""
Supabase 설정 및 환경 변수 관리
"""

import os
from typing import Optional

def _get_supabase_url():
    """Supabase URL 가져오기 (환경변수, top-level secrets, nested secrets 지원)"""
    url = os.getenv("SUPABASE_URL", "").strip()
    if url:
        return url

    # Streamlit secrets 지원 (streamlit이 import된 경우에만)
    try:
        import streamlit as st
        # Top-level secrets
        url = st.secrets.get("SUPABASE_URL", "").strip()
        if url:
            return url
        # Nested secrets
        url = st.secrets.supabase.url
        if url:
            return url
    except:
        pass

    return ""

def _get_supabase_key():
    """Supabase Key 가져오기 (환경변수, top-level secrets, nested secrets 지원)"""
    key = os.getenv("SUPABASE_KEY", "").strip()
    if key:
        return key

    # Streamlit secrets 지원 (streamlit이 import된 경우에만)
    try:
        import streamlit as st
        # Top-level secrets
        key = st.secrets.get("SUPABASE_KEY", "").strip()
        if key:
            return key
        # Nested secrets
        key = st.secrets.supabase.key
        if key:
            return key
    except:
        pass

    return ""

class SupabaseConfig:
    """Supabase 연결 설정"""

    # 환경 변수 또는 secrets에서 설정 로드 (top-level, nested 모두 지원)
    SUPABASE_URL: str = _get_supabase_url()
    SUPABASE_KEY: str = _get_supabase_key()

    # 테이블 이름
    GRADUATES_TABLE = "graduation_employment"
    STATS_CACHE_TABLE = "employment_stats_cache"

    @classmethod
    def is_configured(cls) -> bool:
        """Supabase 설정이 완료되었는지 확인"""
        return bool(cls.SUPABASE_URL and cls.SUPABASE_KEY)

    @classmethod
    def validate(cls) -> tuple[bool, str]:
        """설정 유효성 검사"""
        if not cls.SUPABASE_URL:
            return False, "SUPABASE_URL이 설정되지 않았습니다"
        if not cls.SUPABASE_KEY:
            return False, "SUPABASE_KEY가 설정되지 않았습니다"
        if not cls.SUPABASE_URL.startswith("https://"):
            return False, "유효하지 않은 SUPABASE_URL 형식입니다"
        return True, "설정이 유효합니다"


class AppConfig:
    """애플리케이션 설정"""

    # 앱 메타데이터
    APP_TITLE = "정보컴퓨터공학부 취업 현황"
    APP_ICON = "📊"
    APP_VERSION = "v2.0 (Supabase 연동)"

    # 데이터 설정
    CSV_FILE = "졸업자취업현황_20_21_22_23_24.csv"
    CACHE_TTL = 3600  # 1시간
    BATCH_SIZE = 1000  # 대량 삽입 배치 크기

    # 제외 대상
    EXCLUDE_CATEGORIES = ['진학', '외국인']

    # 인코딩 옵션
    ENCODING_OPTIONS = ['utf-8', 'cp949', 'euc-kr']

    # 색상 팔레트
    COLORS = {
        'primary': '#007bff',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }


# 설정 인스턴스
supabase_config = SupabaseConfig()
app_config = AppConfig()
