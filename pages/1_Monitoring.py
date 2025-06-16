# 실시간 모니터링 페이지
# 현재 전류와 온도 데이터의 실시간 시각화
# 시계열 그래프로 데이터 추이 표시
# 현재 상태 표시 (정상/이상)
# 윈도우에 직접적으로 실시간 알림 띄우기

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from glob import glob

# 페이지 설정
st.set_page_config(
    page_title="실시간 모니터링",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        # 경로 설정
        DATA_DIR = os.path.join('data', '장비이상 조기탐지', '5공정_180sec')
        csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
                    'Error Lot list' not in os.path.basename(p)]
        error_df = pd.read_csv(os.path.join(DATA_DIR, 'Error Lot list.csv'))

        def mark_anomaly(df, err):
            df['is_anomaly'] = 0
            for _, row in err.iterrows():
                date = str(row.iloc[0]).strip()
                procs = set(row.iloc[1:].dropna().astype(int))
                if procs:
                    mask = (df['Date'] == date) & (df['Process'].isin(procs))
                    df.loc[mask, 'is_anomaly'] = 1
            return df

        def load_one(path):
            df = pd.read_csv(path)
            df['Time'] = (df['Time'].str.replace('오전', 'AM')
                                  .str.replace('오후', 'PM'))
            df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df['Index'] = df['Index'].astype(int)
            df = mark_anomaly(df, error_df)
            return df

        # 모든 데이터프레임 로드 및 병합
        dataframes = [load_one(p) for p in csv_paths]
        df = pd.concat(dataframes, ignore_index=True)
        df = df.sort_values('datetime')  # 시간순 정렬
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# 이상치 감지 함수
def detect_anomaly(row):
    # 실제 이상치 데이터 사용
    return row['is_anomaly'] == 1, "이상 감지" if row['is_anomaly'] == 1 else "정상"

def main():
    st.markdown(
        "<h1 style='text-align: center; font-size: 2.5rem; font-weight: bold;'>실시간 모니터링 대시보드</h1>",
        unsafe_allow_html=True
    )
    
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 불러올 수 없습니다.")
        return
    
    # 초기화
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # 모든 동적 콘텐츠를 포함할 단일 placeholder
    main_content_placeholder = st.empty()

    # 데이터포인트 수 조절 슬라이더
    st.sidebar.markdown("### 그래프 설정")
    num_points = st.sidebar.slider(
        "표시할 데이터포인트 수",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="그래프에 표시되는 최대 데이터포인트의 수를 조절하세요."
    )

    # 업데이트 속도 조절 슬라이더
    update_interval = st.sidebar.slider(
        "업데이트 속도 (초)",
        min_value=1,
        max_value=60,
        value=1,
        step=1,
        help="데이터 업데이트 간격을 초 단위로 조절하세요."
    )
    
    while True:
        # 현재 데이터 가져오기
        current_data = df.iloc[st.session_state.current_index]
        current = current_data['Current']
        temp = current_data['Temp']
        
        # 모든 동적 콘텐츠를 단일 컨테이너에 배치하여 매 업데이트마다 갱신
        with main_content_placeholder.container():
            # 그래프 업데이트
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('전류', '온도')
            )
            
            # 선택된 수만큼의 데이터 포인트만 표시
            start_idx = max(0, st.session_state.current_index - num_points)
            end_idx = st.session_state.current_index + 1
            
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[start_idx:end_idx]['datetime'], 
                    y=df.iloc[start_idx:end_idx]['Current'],
                    name='전류',
                    line=dict(color='royalblue', width=3)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[start_idx:end_idx]['datetime'], 
                    y=df.iloc[start_idx:end_idx]['Temp'],
                    name='온도',
                    line=dict(color='tomato', width=3)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='#222831',
                paper_bgcolor='#222831',
                font=dict(color='#EEEEEE', size=14),
                title_font=dict(size=22, color='#FFD369'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=30, r=30, t=60, b=30)
            )
            fig.update_xaxes(showgrid=True, gridcolor='#393E46', color='#EEEEEE')
            fig.update_yaxes(showgrid=True, gridcolor='#393E46', color='#EEEEEE')

            st.plotly_chart(fig, use_container_width=True)
            
            # 메트릭 및 상태 업데이트
            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.markdown(f"<div style='font-size:1.5rem; color:#FFD369;'>현재 전류</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:2.2rem; font-weight:bold; color:#00ADB5;'>{current:.2f}A</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='font-size:1.5rem; color:#FFD369;'>현재 온도</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:2.2rem; font-weight:bold; color:#FF6F61;'>{temp:.2f}°C</div>", unsafe_allow_html=True)
            with col3:
                is_anomaly, message = detect_anomaly(current_data)
                color = '#00FF00' if not is_anomaly else '#FF3333'
                icon = '✅' if not is_anomaly else '⚠️'
                st.markdown(
                    f"<div style='font-size:1.5rem; color:{color};'>상태: {icon} {message}</div>",
                    unsafe_allow_html=True
                )
        
        # 다음 데이터로 이동
        st.session_state.current_index = (st.session_state.current_index + 1) % len(df)
        
        # 선택된 업데이트 간격만큼 대기
        time.sleep(update_interval)

if __name__ == "__main__":
    main()