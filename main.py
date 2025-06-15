# 메인 페이지
# 프로젝트 소개
# 현재 시스템 상태 요약
# 주요 지표 대시보드
# 최근 이상 감지 기록

import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="생산시스템 모니터링",
    page_icon="🏭",
    layout="wide"
)

# 메인 페이지 내용
st.title("생산시스템 모니터링 대시보드")

st.header("프로젝트 소개")
st.write("""
이 프로젝트는 열풍건조 공정의 이상을 실시간으로 감지하고 모니터링하는 시스템입니다.
온도와 전류 데이터를 기반으로 이상을 탐지하며, 실시간 알림 기능을 제공합니다.
""")

# 현재 시스템 상태
st.header("현재 시스템 상태")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="시스템 상태", value="정상")
with col2:
    st.metric(label="마지막 업데이트", value="2024-03-21 15:30:00")
with col3:
    st.metric(label="데이터 포인트", value="1,234")

# 주요 지표
st.header("주요 지표")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="이상 감지율", value="95.5%")
with col2:
    st.metric(label="평균 응답 시간", value="0.12s")
with col3:
    st.metric(label="오탐지율", value="2.1%")
with col4:
    st.metric(label="시스템 가동률", value="99.9%")

# 최근 이상 감지 기록
st.header("최근 이상 감지 기록")
st.write("""
현재 시스템이 정상적으로 작동 중입니다.
마지막 이상 감지는 2024-03-20 14:30:00에 발생했습니다.
""")

if __name__ == "__main__":
    pass  # Streamlit이 자동으로 실행합니다