# 데이터 분석 페이지
# 과거 데이터 분석
# 이상 감지 패턴 분석
# 모델 성능 지표
# 데이터 통계 정보

# 3번파일은 Alert Settings 였는데, 개발 도중 삭제하고 Settings로 통합하였습니다.

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

@st.cache_data # 데이터를 캐싱하여 재로드 방지
def load_historical_data():
    """과거 데이터를 로드하는 함수"""
    try:
        # 경로 설정 (함수 내부에서 정의)
        DATA_DIR = os.path.join('data', '장비이상 조기탐지', '5공정_180sec')
        csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
                    'Error Lot list' not in os.path.basename(p)]
        
        if not csv_paths:
            st.error("데이터 파일을 찾을 수 없습니다.")
            return None
        
        # 에러 데이터 로드 (함수 내부에서 정의)
        error_df = pd.read_csv(os.path.join(DATA_DIR, 'Error Lot list.csv'))
        
        # mark_anomaly 함수 (함수 내부에서 정의)
        def mark_anomaly(df, err):
            df['is_anomaly'] = 0
            for _, row in err.iterrows():
                date = str(row.iloc[0]).strip()
                procs = set(row.iloc[1:].dropna().astype(int))
                if procs:
                    mask = (df['Date'] == date) & (df['Process'].isin(procs))
                    df.loc[mask, 'is_anomaly'] = 1
            return df
        
        # load_one 함수 (함수 내부에서 정의)
        def load_one(path):
            df = pd.read_csv(path)
            df['Time'] = (df['Time'].str.replace('오전', 'AM')
                                  .str.replace('오후', 'PM'))
            df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df['Index'] = df['Index'].astype(int)
            df = mark_anomaly(df, error_df)

            # --- 데모를 위한 더미 'prediction' 컬럼 추가 ---
            np.random.seed(42)
            df['prediction'] = np.zeros(len(df))
            df.loc[df['is_anomaly'] == 1, 'prediction'] = np.random.uniform(0.6, 0.9, size=df['is_anomaly'].sum())
            df.loc[df['is_anomaly'] == 0, 'prediction'] = np.random.uniform(0.1, 0.4, size=(len(df) - df['is_anomaly'].sum()))
            # --- 더미 'prediction' 컬럼 추가 끝 ---

            return df
        
        # 모든 데이터 로드 및 병합
        dataframes = [load_one(p) for p in csv_paths]
        df = pd.concat(dataframes, ignore_index=True)
        
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

# 전역 파라미터 (이 페이지의 분석 로직에서는 직접 사용되지 않을 수 있습니다.)
WINDOW_WIDTH  = 3    
SLIDE_STEP    = 1    
SEQ_LEN       = 10   
TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.1

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
    
    # 예측 점수를 이진 클래스로 변환 (임계값 0.5 사용 예시)
    # 실제 모델에서는 이 임계값을 조정해야 할 수 있습니다.
    binary_prediction = (df['prediction'] >= 0.5).astype(int)

    metrics = {
        'precision': precision_score(df['is_anomaly'], binary_prediction),
        'recall': recall_score(df['is_anomaly'], binary_prediction),
        'f1': f1_score(df['is_anomaly'], binary_prediction)
    }
    
    return metrics

def get_total_data_points():
    """전체 데이터 포인트 수를 반환하는 함수"""
    df = load_historical_data()
    if df is None:
        return 0
    return len(df)

def main():
    st.title("📊 데이터 분석")
    
    # 데이터 로드
    df = load_historical_data()
    
    if df is None: # 데이터 로드 실패 시
        return
    
    # 1. 데이터 통계 정보
    st.header("데이터 통계 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 데이터 포인트", len(df))
    with col2:
        # 'is_anomaly' 컬럼이 없는 경우를 대비
        if 'is_anomaly' in df.columns:
            st.metric("이상치 수", int(df['is_anomaly'].sum()))
        else:
            st.metric("이상치 수", "N/A")
    with col3:
        if 'is_anomaly' in df.columns:
            st.metric("정상 데이터 수", int(len(df) - df['is_anomaly'].sum()))
        else:
            st.metric("정상 데이터 수", "N/A")
    
    # 2. 과거 데이터 분석
    st.header("과거 데이터 분석")
    
    # 'timestamp', 'current', 'temperature' 컬럼 존재 여부 확인
    if 'timestamp' in df.columns and 'current' in df.columns and 'temperature' in df.columns:
        # 시간 범위 선택
        # df의 최소/최대 timestamp를 기반으로 default value 설정
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.date_input(
            "분석할 기간을 선택하세요",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['timestamp'] >= pd.Timestamp(start_date)) & (df['timestamp'] <= pd.Timestamp(end_date))
            filtered_df = df[mask]
            
            if not filtered_df.empty:
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
                
                # 이상치 마킹
                anomaly_points = filtered_df[filtered_df['is_anomaly'] == 1]
                if not anomaly_points.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_points['timestamp'],
                        y=anomaly_points['current'],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='x'),
                        name='이상치 (전류)'
                    ))
                    fig.add_trace(go.Scatter(
                        x=anomaly_points['timestamp'],
                        y=anomaly_points['temperature'],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='x'),
                        name='이상치 (온도)',
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
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("선택된 기간에 해당하는 데이터가 없습니다.")
    else:
        st.warning("과거 데이터 분석을 위한 필수 컬럼 (timestamp, current, temperature)이 없습니다.")
    
    # 3. 이상 감지 패턴 분석
    st.header("이상 감지 패턴 분석")
    
    if 'is_anomaly' in df.columns and 'timestamp' in df.columns:
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
                # timedelta를 분 단위로 변환
                time_diff_minutes = pattern_analysis['time_diff'].dt.total_seconds() / 60
                fig = px.histogram(
                    time_diff_minutes,
                    title='이상치 발생 간격 분포 (분)',
                    labels={'value': '간격 (분)', 'count': '빈도'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("이상 감지 패턴을 분석할 데이터가 부족하거나, 필요한 컬럼이 없습니다.")
    else:
        st.warning("이상 감지 패턴 분석을 위한 필수 컬럼 (is_anomaly, timestamp)이 없습니다.")

    
    # 4. 모델 성능 지표
    st.header("모델 성능 지표")
    
    if 'is_anomaly' in df.columns and 'prediction' in df.columns:
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
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1], constrain='range'), # ROC 커브 범위 고정
                yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1, constrain='range') # ROC 커브 범위 고정
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("모델 성능 지표를 계산할 데이터가 부족하거나, 필요한 컬럼이 없습니다.")
    else:
        st.warning("모델 성능 지표를 위한 필수 컬럼 (is_anomaly, prediction)이 없습니다.")

if __name__ == "__main__":
    main()