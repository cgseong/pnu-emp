"""
Supabase 설정 및 클라이언트 초기화
"""

import streamlit as st
import os


# =====================
# Supabase 클라이언트 초기화
# =====================

@st.cache_resource
def init_supabase():
    """Supabase 클라이언트 초기화"""
    from supabase import create_client, Client

    try:
        # Streamlit secrets에서 먼저 시도
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except (KeyError, FileNotFoundError):
        # 환경변수로 폴백
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "Supabase 설정을 찾을 수 없습니다. "
                ".streamlit/secrets.toml 또는 환경변수를 확인하세요."
            )

    return create_client(url, key)
