# 메인 페이지
# 프로젝트 소개
# 현재 시스템 상태 요약
# 주요 지표 대시보드
# 최근 이상 감지 기록

import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="생산시스템 모니터링 시스템",
    page_icon="🏭",
    layout="wide"
)

# --- 1. 히어로 섹션 ---
with st.container():
    st.markdown("""
    <style>
    .hero-title {
        font-size: 4.5em; /* 크게 */
        font-weight: 700;
        text-align: center;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.8em;
        text-align: center;
        color: #555;
        margin-bottom: 3em;
    }
    .section-header {
        font-size: 2.8em;
        font-weight: 600;
        text-align: center;
        margin-top: 2em;
        margin-bottom: 1em;
    }
    .section-text {
        font-size: 1.2em;
        line-height: 1.6;
        text-align: center;
        max-width: 800px; /* 텍스트 너비 제한 */
        margin-left: auto; /* 중앙 정렬 */
        margin-right: auto; /* 중앙 정렬 */
        margin-bottom: 2em;
    }
    .feature-point {
        font-size: 1.1em;
        line-height: 1.5;
        padding: 0.8em 0; /* 각 포인트별 패딩 */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='hero-title'>열풍건조 장비 이상감지</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>열풍건조 공정의 비정상 패턴을 AI 모델을 활용하여 실시간으로 감지하고 이상을 예측함으로써 생산 효율을 극대화합니다.</p>", unsafe_allow_html=True)

    st.image("images/training_history.png", caption="") # 예시 이미지
    st.markdown("---")

# --- 2. 문제 정의 섹션 ---
with st.container():
    st.markdown("<h2 class='section-header'>생산 현장의 보이지 않는 위협</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='section-text'>
    복잡한 생산 공정에서 발생하는 미세한 이상 징후는 생산성 저하, 품질 불량, 심지어는 치명적인 설비 고장으로 이어질 수 있습니다.
    전통적인 방식으로는 이러한 비정상 패턴을 사전에 감지하기 어렵고, 문제가 발생한 후에야 인지하는 경우가 대부분입니다.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- 3. 솔루션 섹션 ---
with st.container():
    st.markdown("<h2 class='section-header'>우리의 솔루션: AI 기반 실시간 모니터링</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='section-text'>
    우리의 시스템은 열풍건조 공정에서 발생하는 방대한 센서 데이터(전류, 온도 등)를 딥러닝 모델이 실시간으로 분석합니다.
    정상 범주를 벗어나는 미세한 변화까지 학습하여 이상 징후를 즉시 감지하고, 관리자에게 알림을 전송합니다.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-text'>
    <ul>
        <li class='feature-point'><b>실시간 이상 감지:</b> 딥러닝 모델이 연속적인 데이터를 분석하여 비정상 패턴 즉시 식별.</li>
        <li class='feature-point'><b>예지 보전 가능성:</b> 고장 전 징후를 포착하여 선제적인 유지보수 계획 수립.</li>
        <li class='feature-point'><b>직관적인 대시보드:</b> 전류, 온도 추이, 이상 확률 등 핵심 정보를 한눈에 파악.</li>
        <li class='feature-point'><b>맞춤형 알림 설정:</b> 이메일 등 다양한 방식으로 이상 발생 즉시 관리자에게 통보.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- 4. 핵심 기술 섹션 ---
with st.container():
    st.markdown("<h2 class='section-header'>견고한 시스템을 위한 핵심 기술</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='section-text'>
    최첨단 딥러닝 알고리즘과 현대적인 웹 프레임워크를 결합하여 안정적이고 효율적인 모니터링 시스템을 구축했습니다.
    </p>
    """, unsafe_allow_html=True)
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("""
        ### 딥러닝 모델 (LSTM)
        시계열 데이터의 장기 의존성을 학습하여 고도화된 이상 감지 수행.
        """)
    with tech_col2:
        st.markdown("""
        ### Streamlit (Python)
        빠르고 효율적인 웹 대시보드 구축 및 데이터 시각화.
        """)
    with tech_col3:
        st.markdown("""
        ### Plotly
        인터랙티브하고 미려한 데이터 시계열 및 분석 그래프 제공.
        """)
    st.markdown("---")

# --- 5. 주요 이점 섹션 ---
with st.container():
    st.markdown("<h2 class='section-header'>생산 효율을 한 단계 끌어올리다</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-text'>
    <ul>
        <li class='feature-point'><b>생산성 향상:</b> 이상 발생률 감소 및 예측을 통한 가동 시간 극대화.</li>
        <li class='feature-point'><b>품질 개선:</b> 잠재적 불량 요인 조기 발견으로 제품 품질 향상.</li>
        <li class='feature-point'><b>유지보수 비용 절감:</b> 계획되지 않은 설비 중단 방지 및 효율적인 자원 배분.</li>
        <li class='feature-point'><b>의사결정 지원:</b> 정확한 데이터 기반으로 신속하고 현명한 판단 지원.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- 6. 마무리 ---
with st.container():
    st.markdown("<h2 class='section-header'>지능형 생산의 미래, 지금 경험하세요.</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-text'>본 시스템에 대한 더 자세한 정보나 문의 사항이 있으시면 언제든지 연락 주십시오.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    pass # Streamlit이 자동으로 실행합니다.