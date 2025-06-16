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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상수 정의
SEQ_LEN = 10  # 시퀀스 길이

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
        self.scalers: List[StandardScaler] = []
        self.seq_len = SEQ_LEN
        self.threshold = 0.5
        
    def load_models(self) -> None:
        """모델과 스케일러를 로드합니다."""
        try:
            # 모델 로드
            model_path = self.model_dir / 'prediction_model.h5'
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            self.model = keras.models.load_model(model_path)
            logger.info("모델 로드 완료")
            
            # 스케일러 로드
            scaler_files = list(self.model_dir.glob('*_scaler.pkl'))
            if not scaler_files:
                raise FileNotFoundError("스케일러 파일을 찾을 수 없습니다.")
            
            self.scalers = [joblib.load(scaler_file) for scaler_file in scaler_files]
            logger.info(f"{len(self.scalers)}개의 스케일러 로드 완료")
            
        except Exception as e:
            logger.error(f"모델/스케일러 로드 실패: {str(e)}")
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
            
            timestamps = pd.to_datetime(data['datetime'].iloc[-len(predictions):])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Temperature', 'Current')
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data['Temp'].iloc[-len(predictions):],
                    name='Temperature',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data['Current'].iloc[-len(predictions):],
                    name='Current',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Sensor Data with Predictions"
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
                    'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
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
            if self.model is None or not self.scalers:
                self.load_models()
            
            if 'datetime' not in data.columns:
                data['Time'] = (data['Time'].str.replace('오전', 'AM')
                                          .str.replace('오후', 'PM'))
                data['Time'] = pd.to_datetime(data['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
                data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
            
            all_predictions = []
            for scaler in self.scalers:
                X = self.prepare_sequence(data, scaler)
                predictions = self.model.predict(X, verbose=0)
                all_predictions.append(predictions)
            
            final_predictions = np.mean(all_predictions, axis=0)
            last_probability = final_predictions[-1][0]
            
            formatted_result = self.format_prediction_result(final_predictions, data, last_probability)
            
            logger.info(f"예측 완료: 마지막 시퀀스의 이상치 확률 = {last_probability*100:.2f}%")
            return final_predictions, last_probability, formatted_result
            
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
        st.error(f"모델 로드 실패: {e}. 'models' 디렉토리에 prediction_model.h5와 scaler 파일들이 있는지 확인해주세요.")
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
        current_input = st.number_input("전류 값 (Current)", min_value=0.0, value=1.0, step=0.01)
    with col2:
        temp_input = st.number_input("온도 값 (Temperature)", min_value=0.0, value=25.0, step=0.01)

    if 'manual_check_data_history' not in st.session_state:
        st.session_state.manual_check_data_history = pd.DataFrame(
            columns=['Date', 'Time', 'Current', 'Temp', 'Process', 'datetime']
        )
    
    if st.button("이상치 검사 실행"):
        if len(st.session_state.manual_check_data_history) < (SEQ_LEN - 1):
            st.warning(f"과거 데이터가 부족합니다. 최소 {SEQ_LEN-1}개의 데이터가 필요합니다.")
            st.info(f"현재 {len(st.session_state.manual_check_data_history)}개 데이터만 있습니다. 여러 번 입력해주세요.")
            
            for _ in range((SEQ_LEN - 1) - len(st.session_state.manual_check_data_history)):
                dummy_time = datetime.now() - timedelta(seconds=(SEQ_LEN - 1 - _))
                st.session_state.manual_check_data_history = pd.concat([
                    st.session_state.manual_check_data_history,
                    pd.DataFrame({
                        'Date': [dummy_time.strftime('%Y-%m-%d')],
                        'Time': [dummy_time.strftime('%p %I:%M:%S.%f')],
                        'Current': [1.0],
                        'Temp': [25.0],
                        'Process': [1],
                        'datetime': [dummy_time]
                    })
                ], ignore_index=True)

        current_time = datetime.now()
        new_data_point = pd.DataFrame({
            'Date': [current_time.strftime('%Y-%m-%d')],
            'Time': [current_time.strftime('%p %I:%M:%S.%f')],
            'Current': [current_input],
            'Temp': [temp_input],
            'Process': [1],
            'datetime': [current_time]
        })
        
        st.session_state.manual_check_data_history = pd.concat(
            [st.session_state.manual_check_data_history, new_data_point],
            ignore_index=True
        )
        
        if len(st.session_state.manual_check_data_history) > SEQ_LEN:
            st.session_state.manual_check_data_history = st.session_state.manual_check_data_history.tail(SEQ_LEN).reset_index(drop=True)
        
        try:
            if len(st.session_state.manual_check_data_history) < SEQ_LEN:
                st.error(f"데이터 시퀀스 길이가 부족하여 예측할 수 없습니다. 최소 {SEQ_LEN}개의 데이터가 필요합니다.")
                st.stop()

            data_for_prediction = st.session_state.manual_check_data_history.tail(SEQ_LEN).copy()
            _, last_prob, result = predictor.predict(data_for_prediction)
            
            st.subheader("검사 결과")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("이상치 확률", f"{result.anomaly_probability*100:.2f}%")
            with col2:
                status_text = "정상" if not result.is_anomaly else "이상 감지"
                status_color = "green" if not result.is_anomaly else "red"
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
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
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

        except ValueError as ve:
            st.error(f"데이터 오류: {ve}")
            st.warning("예측을 위해서는 최소 10개의 연속된 데이터 포인트가 필요합니다. 몇 번 더 입력해보세요.")
        except Exception as e:
            st.error(f"예측 중 알 수 없는 오류 발생: {e}")

    st.markdown("---")
    st.markdown("**참고:** 예측 모델은 최소 10개의 연속된 데이터 포인트(시퀀스)를 기반으로 작동합니다. 첫 몇 번의 입력 시에는 '과거 데이터 부족' 메시지가 나타날 수 있습니다.")

if __name__ == "__main__":
    main()