# 설정 페이지

# 1. 데이터 관리 설정
# 데이터 수집 주기: 데이터 수집 간격 설정 (예: 1초, 5초, 10초 등)
# 데이터 저장 기간: 수집된 데이터의 보관 기간 설정
# 데이터 백업 설정: 데이터 자동 백업 활성화 및 백업 주기/경로 설정

# 2. 시각화 설정
# 그래프 표시 기간: 대시보드 그래프에 표시할 데이터 기간 설정 (예: 최근 1시간, 6시간, 24시간 등)
# 그래프 색상 테마: 대시보드 그래프의 색상 테마 선택
# 데이터 포인트 표시 개수: 그래프에 표시될 데이터 포인트의 최대 개수 설정 (성능 최적화를 위해)

# 3. 알림 설정
# 알림 활성화/비활성화: 이상 감지 시 알림 수신 여부 설정
# 최적 임계값 적용: ROC 커브 기반으로 계산된 최적 임계값 자동 적용/수동 설정 옵션
# 현재 적용된 임계값: 현재 시스템에 적용 중인 이상치 탐지 임계값 표시
# 알림 방식: 알림 수신 방법 설정 (예: 데스크톱 알림, 이메일, SMS 등)
# 알림 수신자: 알림을 받을 사용자(들) 지정 (예: 이메일 주소 입력)

# 4. 사용자 프로필 (선택 사항)
# 사용자 이름: 현재 사용자의 이름 표시/수정
# 이메일 주소: 알림 수신 등에 사용될 이메일 주소 입력/수정

import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import smtplib # 이메일 전송을 위한 라이브러리 추가
from email.mime.text import MIMEText # 이메일 본문 작성을 위한 라이브러리 추가

# 페이지 설정
st.set_page_config(
    page_title="설정",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main() -> None:
    st.title("⚙️ 설정 페이지")
    st.markdown("---")

    # 1. 데이터 관리 설정
    st.header("데이터 관리 설정")

    st.subheader("데이터 저장 기간")
    data_storage_duration = st.number_input(
        "데이터를 저장할 기간 (일)을 입력하세요:",
        min_value=7,
        max_value=365,
        value=30,
        step=1,
        key="data_storage_duration"
    )
    st.info(f"현재 설정: {data_storage_duration}일 동안 데이터 저장")

    st.subheader("데이터 백업 설정")
    backup_enabled = st.checkbox("자동 데이터 백업 활성화", value=True, key="backup_enabled")
    if backup_enabled:
        backup_frequency = st.selectbox(
            "백업 주기:",
            options=["매일", "매주", "매월"],
            index=0,
            key="backup_frequency"
        )
        backup_path = st.text_input(
            "백업 경로:",
            value=os.path.join(os.getcwd(), "backup_data"),
            key="backup_path"
        )
        st.info(f"현재 설정: {backup_frequency} '{backup_path}'에 백업")
    else:
        st.info("자동 데이터 백업이 비활성화되었습니다.")

    st.markdown("---")

    # 3. 알림 설정
    st.header("알림 설정")
    
    st.subheader("알림 활성화/비활성화")
    alert_enabled = st.checkbox("이상 감지 시 알림 활성화", value=True, key="alert_enabled")

    st.subheader("알림 방식")
    alert_methods = st.multiselect(
        "알림 수신 방법을 선택하세요:",
        options=["데스크톱 알림", "이메일", "SMS"],
        default=["데스크톱 알림", "이메일"],
        key="alert_methods"
    )
    st.info(f"현재 설정: {', '.join(alert_methods)}으로 알림 수신")

    st.subheader("알림 수신자")
    alert_recipients = st.text_input(
        "알림을 받을 이메일 주소(쉼표로 구분):",
        value="example@example.com",
        key="alert_recipients"
    )
    st.info(f"현재 설정: {alert_recipients}로 알림 발송")

    st.markdown("---")

    # 4. 사용자 프로필 (선택 사항)
    st.header("사용자 프로필")

    user_name = st.text_input(
        "사용자 이름:",
        value="User Name",
        key="user_name"
    )
    user_email = st.text_input(
        "이메일 주소:",
        value="user@example.com",
        key="user_email"
    )
    st.info(f"프로필: {user_name} ({user_email})")

    st.markdown("---")

if __name__ == "__main__":
    main()