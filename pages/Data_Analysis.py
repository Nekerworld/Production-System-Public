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

@st.cache_data
def load_historical_data(selected_file):
    """과거 데이터를 로드하는 함수"""
    try:
        # 경로 설정 (함수 내부에서 정의)
        DATA_DIR = os.path.join('data', '장비이상 조기탐지', '5공정_180sec')
        csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
                    'Error Lot list' not in os.path.basename(p)]
        
        if not csv_paths:
            st.error("데이터 파일을 찾을 수 없습니다.")
            return None
        
        # 선택된 파일의 전체 경로 찾기
        selected_path = next(p for p in csv_paths if os.path.basename(p) == selected_file)
        
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
        
        # 선택된 파일만 로드
        df = load_one(selected_path)
        
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def get_available_files():
    """사용 가능한 CSV 파일 목록을 반환하는 함수"""
    DATA_DIR = os.path.join('data', '장비이상 조기탐지', '5공정_180sec')
    csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
                'Error Lot list' not in os.path.basename(p)]
    return [os.path.basename(p) for p in csv_paths]

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

def get_anomaly_detection_rate():
    """이상 감지율을 계산하는 함수"""
    df = load_historical_data()
    if df is None or 'is_anomaly' not in df.columns or 'prediction' not in df.columns:
        return 0.0
    
    # 예측 점수를 이진 클래스로 변환 (임계값 0.5 사용)
    binary_prediction = (df['prediction'] >= 0.5).astype(int)
    
    # 실제 이상치 중에서 제대로 감지된 비율
    true_positives = ((df['is_anomaly'] == 1) & (binary_prediction == 1)).sum()
    total_anomalies = (df['is_anomaly'] == 1).sum()
    
    if total_anomalies == 0:
        return 0.0
    
    return (true_positives / total_anomalies) * 100

def get_false_detection_rate():
    """오탐지율을 계산하는 함수"""
    df = load_historical_data()
    if df is None or 'is_anomaly' not in df.columns or 'prediction' not in df.columns:
        return 0.0
    
    # 예측 점수를 이진 클래스로 변환 (임계값 0.5 사용)
    binary_prediction = (df['prediction'] >= 0.5).astype(int)
    
    # 정상 데이터 중에서 이상으로 잘못 감지된 비율
    false_positives = ((df['is_anomaly'] == 0) & (binary_prediction == 1)).sum()
    total_normal = (df['is_anomaly'] == 0).sum()
    
    if total_normal == 0:
        return 0.0
    
    return (false_positives / total_normal) * 100

def get_system_uptime():
    """시스템 가동률을 계산하는 함수"""
    # 실제 구현에서는 시스템 로그나 모니터링 데이터를 사용해야 합니다.
    # 현재는 데모를 위해 고정값 반환
    return 99.9

def get_average_response_time():
    """평균 응답 시간을 계산하는 함수"""
    # 실제 구현에서는 시스템 로그나 모니터링 데이터를 사용해야 합니다.
    # 현재는 데모를 위해 고정값 반환
    return 0.12

def main():
    st.title("📊 데이터 분석")
    
    # CSS 스타일 적용
    st.markdown("""
        <style>
        .stat-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-title {
            color: #666666;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .stat-value {
            color: #1f77b4;
            font-size: 1.8em;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 1. 데이터 통계 정보
    st.header("데이터 통계 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-title">총 데이터 포인트 수</div>
                <div class="stat-value">307,083</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-title">총 데이터 수</div>
                <div class="stat-value">51,084</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-title">정상 데이터 수</div>
                <div class="stat-value">47,088</div>
            </div>
        """, unsafe_allow_html=True)  
    with col4:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-title">이상치 수</div>
                <div class="stat-value">3,996</div>
            </div>
        """, unsafe_allow_html=True)
            
        # 사용 가능한 파일 목록 가져오기
    available_files = get_available_files()
    
    if not available_files:
        st.error("분석할 데이터 파일이 없습니다.")
        return
    
    st.markdown("---")
    st.subheader(f"개별 데이터 분석")
    # 파일 선택 위젯
    selected_file = st.selectbox("", available_files)
    
    # 데이터 로드
    df = load_historical_data(selected_file)
    
    if df is None: # 데이터 로드 실패 시
        return

    # 2. 데이터 상세 분석
    st.header("데이터 상세 분석")
    
    tab1, tab2, tab3 = st.tabs(["이상 데이터", "정상/이상 분포", "데이터 상관관계"])
    
    with tab1:
        if 'is_anomaly' in df.columns:
            # 항상 전체 데이터프레임 보여주기
            st.dataframe(df, use_container_width=True)
            anomaly_data = df[df['is_anomaly'] == 1]
            if not anomaly_data.empty:
                st.warning(f"이상 데이터 수: {len(anomaly_data)}")
            else:
                st.success("이상 데이터 없음")
        else:
            st.warning("이상 데이터 분석을 위한 'is_anomaly' 컬럼이 없습니다.")
    
    with tab2:
        if 'is_anomaly' in df.columns:
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                # 정상/이상 분포 파이 차트
                is_anomaly_exist = df['is_anomaly'].unique()
                if len(is_anomaly_exist) == 1:
                    # 정상만 있으면
                    if is_anomaly_exist[0] == 0:
                        values = [1]
                        names = ['정상']
                        colors = ['#1f77b4']
                    # 이상만 있으면
                    else:
                        values = [1]
                        names = ['이상']
                        colors = ['#ff7f0e']
                    fig = px.pie(
                        values=values,
                        names=names,
                        title='정상/이상 데이터 비율',
                        color_discrete_sequence=colors
                    )
                    fig.update_traces(textinfo='percent+label')
                else:
                    anomaly_counts = df['is_anomaly'].value_counts()
                    fig = px.pie(
                        values=anomaly_counts.values,
                        names=['정상', '이상'],
                        title='정상/이상 데이터 비율',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e']
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                is_anomaly_exist = df['is_anomaly'].unique()    # 1이면 이상치 없는거
                if len(is_anomaly_exist) == 1:
                    st.markdown(
                        """
                        <div style="
                            background-color: #f8d7da;
                            color: #721c24;
                            border-radius: 8px;
                            padding: 18px 10px;
                            margin: 10px 0 20px 0;
                            border: 1px solid #f5c6cb;
                            font-weight: bold;
                            text-align: center;
                            font-size: 1.1em;
                        ">
                            🚨 이 데이터에는 이상치가 없습니다.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # 온도 분포 (Plotly)
                    fig1 = px.histogram(
                        df,
                        x="Temp",
                        color="is_anomaly",
                        barmode="overlay",
                        nbins=50,
                        color_discrete_map={0: "#1f77b4", 1: "#ff7f0e"},
                        labels={"Temp": "온도", "is_anomaly": "Error"},
                    )
                    fig1.update_layout(
                        title="온도 분포",
                        legend_title_text="Error",
                        legend=dict(
                            itemsizing='constant',
                            title_font=dict(size=12),
                            font=dict(size=12),
                            traceorder="normal",
                            itemclick="toggleothers"
                        )
                    )
                    fig1.update_traces(opacity=0.5)
                    st.plotly_chart(fig1, use_container_width=True)

            with col3:
                is_anomaly_exist = df['is_anomaly'].unique()    # 1이면 이상치 없는거
                if len(is_anomaly_exist) == 1:
                    st.markdown(
                        """
                        <div style="
                            background-color: #f8d7da;
                            color: #721c24;
                            border-radius: 8px;
                            padding: 18px 10px;
                            margin: 10px 0 20px 0;
                            border: 1px solid #f5c6cb;
                            font-weight: bold;
                            text-align: center;
                            font-size: 1.1em;
                        ">
                            🚨 이 데이터에는 이상치가 없습니다.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # 전류 분포 (Plotly)
                    fig2 = px.histogram(
                        df,
                        x="Current",
                        color="is_anomaly",
                        barmode="overlay",
                        nbins=50,
                        color_discrete_map={0: "#1f77b4", 1: "#ff7f0e"},
                        labels={"Current": "전류", "is_anomaly": "Error"}
                    )
                    fig2.update_layout(
                        title="전류 분포",
                        legend_title_text="Error",
                        legend=dict(
                            itemsizing='constant',
                            title_font=dict(size=12),
                            font=dict(size=12),
                            traceorder="normal",
                            itemclick="toggleothers"
                        )
                    )
                    fig2.update_traces(opacity=0.5)
                    st.plotly_chart(fig2, use_container_width=True)
                
        else:
            st.warning("정상/이상 분포 분석을 위한 'is_anomaly' 컬럼이 없습니다.")
    
    with tab3:
        if 'is_anomaly' in df.columns:
            col1, col2 = st.columns([2, 1])
            with col1:
                is_anomaly_exist = df['is_anomaly'].unique()    # 1이면 이상치 없는거
                if len(is_anomaly_exist) == 1:
                    st.markdown(
                        """
                        <div style="
                            background-color: #f8d7da;
                            color: #721c24;
                            border-radius: 8px;
                            padding: 18px 10px;
                            margin: 10px 0 20px 0;
                            border: 1px solid #f5c6cb;
                            font-weight: bold;
                            text-align: center;
                            font-size: 1.1em;
                        ">
                            🚨 이 데이터에는 이상치가 없습니다.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # 정상/이상 모두 있을 때
                    fig = px.scatter(
                        df,
                        x='timestamp',
                        y='prediction',
                        color='is_anomaly',
                        title='시간에 따른 예측값 분포',
                        labels={'prediction': '예측값', 'timestamp': '시간'}
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 이상치가 없는 경우 처리
                if df['is_anomaly'].sum() == 0:
                    st.info("이 데이터에는 이상치가 없습니다.")
                else:
                    fig_correlation = px.scatter(
                        df,
                        x='Temp',
                        y='Current',
                        color='is_anomaly',
                        title='온도와 전류의 관계',
                        labels={'Temp': '온도', 'Current': '전류'},
                        color_discrete_map={0: 'blue', 1: 'red'}
                    )
                    st.plotly_chart(fig_correlation, use_container_width=True)
        else:
            st.warning("데이터 상관관계 분석을 위한 'is_anomaly' 컬럼이 없습니다.")
   
    is_anomaly_exist = df['is_anomaly'].unique()    # 1이면 이상치 없는거
    if len(is_anomaly_exist) == 1:
        st.markdown(
            """
            <div style="
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 8px;
                padding: 18px 10px;
                margin: 10px 0 20px 0;
                border: 1px solid #f5c6cb;
                font-weight: bold;
                text-align: center;
                font-size: 1.1em;
            ">
                🚨 이 데이터에는 이상치가 없습니다.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
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