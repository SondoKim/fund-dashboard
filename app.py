import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from streamlit_gsheets import GSheetsConnection  # 이 줄이 추가됨

# 1. 페이지 설정
st.set_page_config(page_title="泰 투자자문 성과분석", layout="wide")

st.title("📈 泰 투자자문 펀드 현황")
st.markdown("---")

# ---------------------------------------------------------
# [수정된 부분] 2. 구글 시트 데이터 가져오기
# ---------------------------------------------------------
try:
    # Secrets에 설정한 gsheets 연결 객체 생성
    conn = st.connection("gsheets", type=GSheetsConnection)

    # 데이터 읽기 (캐시 시간 0으로 설정하면 매번 최신 데이터를 가져옴. 너무 자주 하면 느리니 600초(10분) 추천)
    df = conn.read(worksheet="Holdings", ttl=0) 
    
    # 필수 컬럼이 있는지 확인
    required_cols = ['Ticker', 'Name', 'Quantity', 'AvgPrice', 'EntryDate']
    if not all(col in df.columns for col in required_cols):
        st.error("구글 시트의 컬럼명이 올바르지 않습니다. (Ticker, Name, Quantity, AvgPrice, EntryDate 확인 필요)")
        st.stop()

    # 데이터가 비어있는 경우 처리
    if df.empty:
        st.warning("보유 종목 데이터가 없습니다. 구글 시트에 데이터를 입력해주세요.")
        st.stop()

except Exception as e:
    st.error(f"데이터베이스 연결 오류: {e}")
    st.stop()
# ---------------------------------------------------------

# 3. 야후 파이낸스에서 현재가 및 과거 데이터 수집
# (이하 로직은 이전 코드와 동일하지만, df 변수를 위에서 받아온 것으로 사용)
tickers = df['Ticker'].tolist()

if len(tickers) > 0:
    # ... (이전 코드의 3번 ~ 7번 내용 그대로 사용) ...
    
    # (중략: 환율, yfinance 다운로드, 각종 지표 계산 로직)
    # 아래는 테스트용으로 필요한 부분만 짧게 요약했습니다. 실제로는 이전 코드의 로직을 그대로 이어 붙이세요.
    
    # 예시: 현재가 가져오는 부분
    try:
        hist_data = yf.download(tickers, period="1y")['Close']
        if hist_data.empty:
             st.error("주가 데이터를 가져오는데 실패했습니다. 티커를 확인해주세요.")
             st.stop()
        
        # 데이터프레임 구조에 따라 처리 (단일 종목일 경우 Series로 반환될 수 있음)
        if isinstance(hist_data, pd.Series):
             hist_data = hist_data.to_frame()

        current_prices = hist_data.iloc[-1]
        
        # ... (이후 성과 계산 및 차트 그리기 로직 동일) ...
        
        # 간단한 확인용 출력 (코드 통합 시 삭제)
        st.write("데이터 로드 성공!", df.head())

    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")

else:
    st.write("보유 종목이 없습니다.")
