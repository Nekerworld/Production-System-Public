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

# 사이드바 내용 추가
st.sidebar.header("2조 전자경영 팀")
st.sidebar.write("총괄팀장: 이인수")
st.sidebar.write("개발팀장: 김윤성")
st.sidebar.write("분석팀장: 최승환")
st.sidebar.write("조원: 이지원")
st.sidebar.markdown("---")

# --- 1. 히어로 섹션 ---
with st.container():
    st.markdown("""
    <style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css");
    @import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css");

    * {
        font-family: Pretendard;
        padding: 0;
        margin: 0;
    }

    .section {
        width: 100%;
        height: 100vh;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .section1 {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                    url('https://i.imgur.com/wVwCjnr.jpeg') no-repeat;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    .section2 {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                    url('https://i.imgur.com/cW4y4s3.jpeg') no-repeat;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    .section3 {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                    url('https://i.imgur.com/iUenTpM.jpeg') no-repeat;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    .section4 {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                    url('https://i.imgur.com/c5tnILg.jpeg') no-repeat;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    .section5 {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                    url('https://i.imgur.com/OHGBO4C.jpeg') no-repeat;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    .section6 {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                    url('https://i.imgur.com/bcps4wK.jpeg') no-repeat;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    .section-content {
        color: white;
        padding: 5rem;
        background: rgba(30, 57, 50, 0.8);
        border-radius: 10px;
        max-width: 1000px;
    }

    .hero-title {
        font-size: 5em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5em;
        line-height: 1.1;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .hero-subtitle {
        font-size: 1.8em;
        text-align: center;
        color: white;
        margin-bottom: 1em;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }

    .section-header {
        font-size: 2.8em;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1em;
        color: white;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }

    .section-text {
        font-size: 1.2em;
        line-height: 1.6;
        text-align: center;
        margin-bottom: 2em;
        color: white;
    }

    .feature-point {
        font-size: 1.1em;
        line-height: 1.5;
        padding: 0.8em 0;
        color: white;
    }

    .contact-info {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2em;
    }

    .contact-item {
        display: flex;
        align-items: center;
        margin: 0 1em;
    }

    .contact-item i {
        margin-right: 0.5em;
    }

    .contact-item a {
        color: white;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # 첫 번째 섹션
    st.markdown("""
    <div class='section section1'>
        <div class='section-content'>
            <h1 class='hero-title'>열풍건조 장비 이상감지</h1>
            <p class='hero-subtitle'>열풍건조 공정의 비정상 패턴을 AI 모델을 활용하여 실시간으로 감지하고 <br>이상을 예측함으로써 생산 효율을 극대화합니다.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 두 번째 섹션
    st.markdown("""
    <div class='section section2'>
        <div class='section-content'>
            <h2 class='section-header'>생산 현장의 보이지 않는 위협</h2>
            <p class='section-text'>
            복잡한 생산 공정에서 발생하는 미세한 이상 징후는 생산성 저하, 품질 불량, 심지어는 치명적인 설비 고장으로 <br>이어질 수 있습니다.
            전통적인 방식으로는 이러한 비정상 패턴을 사전에 감지하기 어렵고, <br>문제가 발생한 후에야 인지하는 경우가 대부분입니다.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 세 번째 섹션
    st.markdown("""
    <div class='section section3'>
        <div class='section-content'>
            <h2 class='section-header'>우리의 솔루션: AI 기반 실시간 모니터링</h2>
            <p class='section-text'>
            AI 모델 기반 모니터링 시스템은 열풍건조 공정에서 발생하는 전류와 온도 등의 방대한 센서 데이터를 <br>딥러닝 모델이 실시간으로 분석합니다.
            정상 범주를 벗어나는 미세한 변화까지 학습하여 <br>이상 징후를 즉시 감지하고, 관리자에게 알림을 전송합니다.
            </p>
            <ul class='section-text'>
                <li class='feature-point'><b>실시간 이상 감지:</b> 딥러닝 모델이 연속적인 데이터를 분석하여 비정상 패턴 즉시 식별.</li>
                <li class='feature-point'><b>예지 보전 가능성:</b> 고장 전 징후를 포착하여 선제적인 유지보수 계획 수립.</li>
                <li class='feature-point'><b>직관적인 대시보드:</b> 전류, 온도 추이, 이상 확률 등 핵심 정보를 한눈에 파악.</li>
                <li class='feature-point'><b>맞춤형 알림 설정:</b> 이메일 등 다양한 방식으로 이상 발생 즉시 관리자에게 통보.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 네 번째 섹션
    st.markdown("""
    <div class='section section4'>
        <div class='section-content'>
            <h2 class='section-header'>견고한 시스템을 위한 핵심 기술</h2>
            <p class='section-text'>
            최첨단 딥러닝 알고리즘과 현대적인 웹 프레임워크를 결합하여 <br>안정적이고 효율적인 모니터링 시스템을 구축했습니다.
            </p>
            <ul class='section-text'>
                <li class='feature-point'><b>LSTM 딥러닝 모델:</b> 시계열 데이터의 장기 의존성을 학습하여 고도화된 이상 감지 수행.</li>
                <li class='feature-point'><b>Streamlit 프레임워크:</b> 빠르고 효율적인 웹 대시보드 구축 및 데이터 시각화.</li>
                <li class='feature-point'><b>Plotly 라이브러리:</b> 인터랙티브하고 미려한 데이터 시계열 및 분석 그래프 제공.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 다섯 번째 섹션
    st.markdown("""
    <div class='section section5'>
        <div class='section-content'>
            <h2 class='section-header'>생산 효율을 한 단계 끌어올리다</h2>
            <p class='section-text'>
            우리의 시스템은 단순한 모니터링을 넘어, 실제 비즈니스 가치를 창출합니다.
            </p>
            <ul class='section-text'>
                <li class='feature-point'><b>생산성 향상:</b> 이상 발생률 감소 및 예측을 통한 가동 시간 극대화.</li>
                <li class='feature-point'><b>품질 개선:</b> 잠재적 불량 요인 조기 발견으로 제품 품질 향상.</li>
                <li class='feature-point'><b>유지보수 비용 절감:</b> 계획되지 않은 설비 중단 방지 및 효율적인 자원 배분.</li>
                <li class='feature-point'><b>의사결정 지원:</b> 정확한 데이터 기반으로 신속하고 현명한 판단 지원.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 여섯 번째 섹션
    st.markdown("""
    <div class='section section6'>
        <div class='section-content'>
            <h2 class='section-header'>지능형 생산의 미래, 지금 경험하세요</h2>
            <p class='section-text'>
            본 시스템에 대한 더 자세한 정보나 문의 사항이 있으시면 언제든지 연락 주십시오.
            </p>
            <div class="contact-info">
                <div class="contact-item">
                    <i class="fas fa-envelope"></i>
                    <a href="mailto:chrisabc94@gmail.com">chrisabc94@gmail.com</a>
                </div>
                <div class="contact-item">
                    <i class="fas fa-phone-alt"></i>
                    <span>+82 10-2204-4587</span>
                </div>
                <div class="contact-item">
                    <i class="fab fa-github"></i>
                    <a href="https://github.com/Nekerworld/Production-System-Public" target="_blank">GitHub Profile</a>
                </div>
                <div class="contact-item">
                    <i class="fas fa-map-marker-alt"></i>
                    <span>한국공학대학교</span>
                </div>
            </div>
            <p class='section-text' style='margin-top: 2em;'>
            지금 바로 시작하여 더 스마트하고 효율적인 생산 시스템을 구축하세요.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    pass # Streamlit이 자동으로 실행합니다.