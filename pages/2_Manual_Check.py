# 수동 검사 페이지
# 사용자가 직접 전류와 온도 값을 입력하여 검사
# 입력된 데이터에 대한 정상/이상 확률 계산
# 입력 데이터의 시각화
# 검사 결과 히스토리

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from glob import glob
from collections import deque

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상수 정의
SEQ_LEN = 10  # 시퀀스 길이
DATA_DIR = os.path.join('data', '장비이상 조기탐지', '5공정_180sec')  # 데이터 디렉토리

# 페이지 설정
st.set_page_config(
    page_title="수동 검사",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class PredictionResult:
    """예측 결과를 저장하는 데이터 클래스"""
    timestamp: str
    anomaly_probability: float
    is_anomaly: bool
    last_sequence: Dict[str, Any]
    prediction_history: Dict[str, List[Any]]
    visualization: go.Figure

class AnomalyPredictor:
    """이상치 예측을 위한 클래스"""
    
    def __init__(self, model_dir: Union[str, Path] = 'models'):
        """
        Args:
            model_dir (Union[str, Path]): 모델과 스케일러가 저장된 디렉토리 경로
        """
        self.model_dir = Path(model_dir)
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.seq_len = SEQ_LEN
        self.threshold = 0.5
        self.historical_data: Optional[pd.DataFrame] = None
        self.window_buffer: deque[Dict[str, Any]] = deque(maxlen=SEQ_LEN)
        
    def load_historical_data(self) -> None:
        """기존 데이터를 로드합니다."""
        try:
            # 에러 데이터 로드
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
            
            # CSV 파일 경로 목록 가져오기
            csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
                        'Error Lot list' not in os.path.basename(p)]
            
            # 모든 데이터프레임 로드 및 병합
            dataframes = [load_one(p) for p in csv_paths]
            self.historical_data = pd.concat(dataframes, ignore_index=True)
            self.historical_data = self.historical_data.sort_values('datetime')  # 시간순 정렬
            
            logger.info("기존 데이터 로드 완료")
        except Exception as e:
            logger.error(f"기존 데이터 로드 실패: {str(e)}")
            raise
        
    def load_models(self) -> None:
        """모델과 스케일러를 로드합니다."""
        try:
            # 모델 로드
            model_path = self.model_dir / 'final_model.keras'
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            self.model = keras.models.load_model(model_path)
            logger.info("모델 로드 완료")
            
            # 스케일러 로드
            scaler_files = list(self.model_dir.glob('global_scaler.pkl'))
            if not scaler_files:
                raise FileNotFoundError("스케일러 파일을 찾을 수 없습니다.")
            
            self.scaler = joblib.load(scaler_files[0])
            logger.info("스케일러 로드 완료")
            
            # 기존 데이터 로드
            self.load_historical_data()
            
        except Exception as e:
            logger.error(f"모델/스케일러 로드 실패: {str(e)}")
            raise
    
    def prepare_single_input(self, current: float, temp: float) -> pd.DataFrame:
        """
        단일 입력값을 시퀀스 데이터로 변환합니다.
        
        Args:
            current (float): 전류 값
            temp (float): 온도 값
            
        Returns:
            pd.DataFrame: 시퀀스 데이터
        """
        try:
            if self.historical_data is None:
                raise ValueError("기존 데이터가 로드되지 않았습니다.")
            
            # 현재 시간 기준으로 시퀀스 생성
            current_time = datetime.now()
            
            # 기존 데이터에서 가장 유사한 패턴 찾기
            ref_data = self.historical_data[['Current', 'Temp']].values
            input_data = np.array([[current, temp]])
            
            # 유사도 계산 (유클리드 거리)
            distances = np.linalg.norm(ref_data - input_data, axis=1)
            most_similar_idx = np.argmin(distances)
            
            # 가장 유사한 시퀀스의 이전 데이터 사용
            start_idx = max(0, most_similar_idx - self.seq_len + 1)
            sequence = self.historical_data.iloc[start_idx:start_idx + self.seq_len - 1].copy()
            
            # 마지막 데이터 포인트 추가
            new_point = pd.DataFrame({
                'Date': [current_time.strftime('%Y-%m-%d')],
                'Time': [current_time.strftime('%p %I:%M:%S.%f')],
                'Current': [current],
                'Temp': [temp],
                'Process': [1],
                'datetime': [current_time],
                'Index': [sequence['Index'].iloc[-1] + 1],
                'is_anomaly': [0]
            })
            
            sequence = pd.concat([sequence, new_point], ignore_index=True)
            return sequence
            
        except Exception as e:
            logger.error(f"단일 입력 데이터 준비 실패: {str(e)}")
            raise
    
    def prepare_sequence(self, data: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
        """
        데이터를 시퀀스 형태로 변환합니다.
        
        Args:
            data (pd.DataFrame): 입력 데이터
            scaler (StandardScaler): 스케일러 객체
            
        Returns:
            np.ndarray: 시퀀스 데이터
        """
        try:
            if len(data) < self.seq_len:
                raise ValueError(f"데이터 길이가 {self.seq_len}보다 작습니다.")
            
            features = data[['Temp', 'Current']].copy()
            scaled_data = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
            
            X = np.array([
                scaled_data.iloc[i:i+self.seq_len].values
                for i in range(len(data) - self.seq_len + 1)
            ])
            return X
            
        except Exception as e:
            logger.error(f"시퀀스 데이터 준비 실패: {str(e)}")
            raise
    
    def format_prediction_result(self, 
                               predictions: np.ndarray, 
                               data: pd.DataFrame,
                               last_probability: float) -> PredictionResult:
        """
        예측 결과를 포맷팅합니다.
        
        Args:
            predictions (np.ndarray): 예측 확률 배열
            data (pd.DataFrame): 원본 데이터
            last_probability (float): 마지막 시퀀스의 예측 확률
            
        Returns:
            PredictionResult: 포맷팅된 예측 결과
        """
        try:
            last_sequence = data.iloc[-self.seq_len:]
            start_time = pd.to_datetime(last_sequence['datetime'].iloc[0])
            end_time = pd.to_datetime(last_sequence['datetime'].iloc[-1])
            
            # 단일 그래프로 변경
            fig = go.Figure()
            
            if self.historical_data is not None:
                # 정상 및 이상 데이터 분류
                normal_data = self.historical_data[self.historical_data['is_anomaly'] == 0]
                anomaly_data = self.historical_data[self.historical_data['is_anomaly'] == 1]
                
                # 각 데이터에서 1%만 샘플링
                normal_sample = normal_data.sample(frac=0.01, random_state=42) if not normal_data.empty else pd.DataFrame()
                anomaly_sample = anomaly_data.sample(frac=0.01, random_state=42) if not anomaly_data.empty else pd.DataFrame()
                
                # 정상 데이터 시각화
                if not normal_sample.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=normal_sample['Temp'],
                            y=normal_sample['Current'],
                            mode='markers',
                            name='Normal Data',
                            marker=dict(color='blue', size=5, opacity=0.6)
                        )
                    )
                
                # 이상 데이터 시각화
                if not anomaly_sample.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_sample['Temp'],
                            y=anomaly_sample['Current'],
                            mode='markers',
                            name='Anomaly Data',
                            marker=dict(color='red', size=7, symbol='x')
                        )
                    )

            # 현재 입력된 데이터를 눈에 띄는 점으로 추가
            current_temp = data['Temp'].iloc[-1]
            current_current = data['Current'].iloc[-1]
            
            fig.add_trace(
                go.Scatter(
                    x=[current_temp],
                    y=[current_current],
                    mode='markers',
                    name='Input Point',
                    marker=dict(color='orange', size=12, symbol='star')
                )
            )

            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Current vs. Temperature Anomaly Detection",
                xaxis_title="Temperature",
                yaxis_title="Current"
            )
            
            return PredictionResult(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                anomaly_probability=float(last_probability),
                is_anomaly=bool(last_probability >= self.threshold),
                last_sequence={
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'avg_temperature': float(last_sequence['Temp'].mean()),
                    'avg_current': float(last_sequence['Current'].mean()),
                    'process_id': int(last_sequence['Process'].iloc[-1])
                },
                prediction_history={
                    'timestamps': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
                    'probabilities': [float(p[0]) for p in predictions]
                },
                visualization=fig
            )
            
        except Exception as e:
            logger.error(f"예측 결과 포맷팅 실패: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, float, PredictionResult]:
        """
        새로운 데이터에 대한 이상치 확률을 예측합니다.
        
        Args:
            data (pd.DataFrame): 예측할 데이터
            
        Returns:
            Tuple[np.ndarray, float, PredictionResult]: 
                (전체 시퀀스에 대한 예측 확률, 마지막 시퀀스의 예측 확률, 포맷팅된 결과)
        """
        try:
            if self.model is None or self.scaler is None:
                self.load_models()
            
            if 'datetime' not in data.columns:
                data['Time'] = (data['Time'].str.replace('오전', 'AM')
                                          .str.replace('오후', 'PM'))
                data['Time'] = pd.to_datetime(data['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
                data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
            
            X = self.prepare_sequence(data, self.scaler)
            predictions = self.model.predict(X, verbose=0)
            last_probability = predictions[-1][0]
            
            formatted_result = self.format_prediction_result(predictions, data, last_probability)
            
            logger.info(f"예측 완료: 마지막 시퀀스의 이상치 확률 = {last_probability*100:.2f}%")
            return predictions, last_probability, formatted_result
            
        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            raise

@st.cache_resource
def get_predictor() -> Optional[AnomalyPredictor]:
    """AnomalyPredictor 인스턴스를 캐싱하여 모델 로드를 한 번만 수행"""
    predictor = AnomalyPredictor()
    try:
        predictor.load_models()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}. 'models' 디렉토리에 final_model.keras와 global_scaler.pkl 파일들이 있는지 확인해주세요.")
        return None
    return predictor

def main() -> None:
    st.title("수동 검사 대시보드")
    st.markdown("---")

    predictor = get_predictor()
    if predictor is None:
        return

    st.subheader("새로운 데이터 입력")
    col1, col2 = st.columns(2)

    with col1:
        current_input = st.number_input("전류 값 (Current)", min_value=0.0, value=1.6, step=0.01)
    with col2:
        temp_input = st.number_input("온도 값 (Temperature)", min_value=0.0, value=70.0, step=0.01)

    if st.button("이상치 검사 실행"):
        try:
            # 단일 입력값으로 시퀀스 데이터 생성
            sequence_data = predictor.prepare_single_input(current_input, temp_input)
            
            # 예측 수행
            _, last_prob, result = predictor.predict(sequence_data)
            
            st.subheader("검사 결과")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("이상치 확률", f"{result.anomaly_probability*100:.2f}%")
            with col2:
                status_text = "이상 감지" if result.is_anomaly else "정상"
                status_color = "red" if result.is_anomaly else "green"
                st.markdown(
                    f"<div style='font-size:20px; color:{status_color}; font-weight:bold;'>상태: {status_text}</div>",
                    unsafe_allow_html=True
                )
            
            st.info(f"**임계값:** {predictor.threshold:.2f}, **신뢰도:** {'높음' if result.anomaly_probability > 0.8 else '중간' if result.anomaly_probability > 0.5 else '낮음'}")

            st.subheader("입력 데이터 시각화")
            st.plotly_chart(result.visualization, use_container_width=True)
            
            if 'manual_check_history' not in st.session_state:
                st.session_state.manual_check_history = []
            
            st.session_state.manual_check_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current': current_input,
                'temperature': temp_input,
                'probability': result.anomaly_probability * 100,
                'status': status_text
            })
            
            st.subheader("검사 결과 히스토리")
            display_history = pd.DataFrame(st.session_state.manual_check_history).tail(10)
            if not display_history.empty:
                st.dataframe(display_history.set_index('timestamp'))
            else:
                st.info("아직 검사 결과가 없습니다.")

        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")

    st.markdown("---")
    st.markdown("**참고:** 입력된 값은 기존 데이터와 비교하여 가장 유사한 패턴을 찾아 예측을 수행합니다.")

if __name__ == "__main__":
    main()