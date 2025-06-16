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
    st.title("실시간 모니터링 대시보드")
    
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 불러올 수 없습니다.")
        return
    
    # 초기화
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # 실시간 데이터 표시를 위한 placeholder
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # 메트릭 표시를 위한 placeholder
    metric_placeholder = st.empty()
    
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
    
    while True:
        # 현재 데이터 가져오기
        current_data = df.iloc[st.session_state.current_index]
        current = current_data['Current']
        temp = current_data['Temp']
        timestamp = current_data['datetime']
        
        # 메트릭 업데이트 (이전 메트릭은 자동으로 지워짐)
        with metric_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("현재 전류", f"{current:.2f}A")
            with col2:
                st.metric("현재 온도", f"{temp:.2f}°C")
        
        # 이상치 감지
        is_anomaly, message = detect_anomaly(current_data)
        
        # 상태 표시
        status_color = "red" if is_anomaly else "green"
        status_placeholder.markdown(
            f'<div style="color: {status_color}; font-size: 20px; text-align: center;">'
            f'상태: {"⚠️ " if is_anomaly else "✅ "}{message}</div>',
            unsafe_allow_html=True
        )
        
        # 그래프 업데이트
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('전류', '온도'))
        
        # 선택된 수만큼의 데이터 포인트만 표시
        start_idx = max(0, st.session_state.current_index - num_points)
        end_idx = st.session_state.current_index + 1
        
        fig.add_trace(
            go.Scatter(x=df.iloc[start_idx:end_idx]['datetime'], 
                      y=df.iloc[start_idx:end_idx]['Current'],
                      name='전류', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.iloc[start_idx:end_idx]['datetime'], 
                      y=df.iloc[start_idx:end_idx]['Temp'],
                      name='온도', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="센서 데이터"
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # 다음 데이터로 이동
        st.session_state.current_index = (st.session_state.current_index + 1) % len(df)
        
        # 1초 대기
        time.sleep(1)

if __name__ == "__main__":
    main()