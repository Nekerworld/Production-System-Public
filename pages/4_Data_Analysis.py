# 데이터 분석 페이지
# 과거 데이터 분석
# 이상 감지 패턴 분석
# 모델 성능 지표
# 데이터 통계 정보

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
from glob import glob

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석",
    page_icon="📊",
    layout="wide"
)

# 경로 설정
DATA_DIR = os.path.join('data', '장비이상 조기탐지', '5공정_180sec')
csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
             'Error Lot list' not in os.path.basename(p)]
error_df   = pd.read_csv(os.path.join(DATA_DIR, 'Error Lot list.csv'))

# 파라미터
WINDOW_WIDTH  = 3    # 한 번에 묶을 CSV 개수
SLIDE_STEP    = 1    # Stride
SEQ_LEN       = 10   # LSTM 시계열 길이
TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.1

def mark_anomaly(df, err):
    df['is_anomaly'] = 0
    for _, row in err.iterrows():
        date  = str(row.iloc[0]).strip()
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

dataframes = [load_one(p) for p in csv_paths]

def analyze_anomaly_patterns(df):
    """이상 감지 패턴을 분석하는 함수"""
    if df is None or 'is_anomaly' not in df.columns:
        return None
    
    anomaly_df = df[df['is_anomaly'] == 1]
    
    # 시간대별 이상 발생 빈도
    anomaly_df['hour'] = pd.to_datetime(anomaly_df['timestamp']).dt.hour
    hourly_pattern = anomaly_df.groupby('hour').size()
    
    # 이상치 발생 간격 분석
    anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'])
    anomaly_df = anomaly_df.sort_values('timestamp')
    anomaly_df['time_diff'] = anomaly_df['timestamp'].diff()
    
    return {
        'hourly_pattern': hourly_pattern,
        'time_diff': anomaly_df['time_diff']
    }

def calculate_model_metrics(df):
    """모델 성능 지표를 계산하는 함수"""
    if df is None or 'is_anomaly' not in df.columns or 'prediction' not in df.columns:
        return None
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {
        'precision': precision_score(df['is_anomaly'], df['prediction']),
        'recall': recall_score(df['is_anomaly'], df['prediction']),
        'f1': f1_score(df['is_anomaly'], df['prediction'])
    }
    
    return metrics

def main():
    st.title("📊 데이터 분석")
    
    # 데이터 로드
    df = dataframes
    
    if df is None:
        st.error("분석할 데이터가 없습니다.")
        return
    
    # 1. 데이터 통계 정보
    st.header("1. 데이터 통계 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 데이터 포인트", len(df))
    with col2:
        st.metric("이상치 수", len(df[df['is_anomaly'] == 1]))
    with col3:
        st.metric("정상 데이터 수", len(df[df['is_anomaly'] == 0]))
    
    # 2. 과거 데이터 분석
    st.header("2. 과거 데이터 분석")
    
    # 시간 범위 선택
    date_range = st.date_input(
        "분석할 기간을 선택하세요",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['timestamp'] >= pd.Timestamp(start_date)) & (df['timestamp'] <= pd.Timestamp(end_date))
        filtered_df = df[mask]
        
        # 시계열 그래프
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df['current'],
            name='전류',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df['temperature'],
            name='온도',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='전류 및 온도 시계열 데이터',
            xaxis_title='시간',
            yaxis_title='전류 (A)',
            yaxis2=dict(
                title='온도 (°C)',
                overlaying='y',
                side='right'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. 이상 감지 패턴 분석
    st.header("3. 이상 감지 패턴 분석")
    
    pattern_analysis = analyze_anomaly_patterns(df)
    if pattern_analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            # 시간대별 이상 발생 빈도
            fig = px.bar(
                x=pattern_analysis['hourly_pattern'].index,
                y=pattern_analysis['hourly_pattern'].values,
                title='시간대별 이상 발생 빈도',
                labels={'x': '시간', 'y': '이상 발생 횟수'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 이상치 발생 간격 히스토그램
            fig = px.histogram(
                x=pattern_analysis['time_diff'].dt.total_seconds() / 60,
                title='이상치 발생 간격 분포 (분)',
                labels={'x': '간격 (분)', 'y': '빈도'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 4. 모델 성능 지표
    st.header("4. 모델 성능 지표")
    
    metrics = calculate_model_metrics(df)
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("정밀도 (Precision)", f"{metrics['precision']:.3f}")
        with col2:
            st.metric("재현율 (Recall)", f"{metrics['recall']:.3f}")
        with col3:
            st.metric("F1 점수", f"{metrics['f1']:.3f}")
        
        # ROC 커브
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(df['is_anomaly'], df['prediction'])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'ROC 커브 (AUC = {roc_auc:.3f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='무작위 예측',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='ROC 커브',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()