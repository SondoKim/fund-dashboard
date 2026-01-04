import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="OO투자자문 성과분석", layout="wide")

st.title("📈 OO투자자문 펀드 현황")
st.markdown("---")

# 2. 구글 시트 데이터 가져오기 (여기서는 예시 데이터를 직접 넣지만, 실제론 연동 코드가 들어갑니다)
# 실제 배포 시에는 st.connection을 사용하여 구글 시트와 연결합니다.
# 지금은 테스트를 위해 위에서 만든 엑셀 데이터를 그대로 DataFrame으로 만듭니다.
data = {
    'Ticker': ['005930.KS', '000660.KS', 'AAPL'],
    'Name': ['삼성전자', 'SK하이닉스', 'Apple'],
    'Quantity': [100, 50, 20],
    'AvgPrice': [72000, 135000, 185], # 원화 환산 가정 필요하거나 통화 구분 필요
    'EntryDate': ['2024-01-15', '2024-02-01', '2023-11-20']
}
df = pd.DataFrame(data)

# 3. 야후 파이낸스에서 현재가 및 과거 데이터 수집
tickers = df['Ticker'].tolist()
if len(tickers) > 0:
    # 환율 정보 (간략화: 1달러 = 1350원 고정, 실제론 환율 API 연동 추천)
    usd_krw = 1350.0 
    
    # 1년치 데이터 다운로드
    hist_data = yf.download(tickers, period="1y")['Close']
    
    # 현재가 가져오기 (가장 최근 종가)
    current_prices = hist_data.iloc[-1]

    # 데이터 프레임에 현재가 추가 및 평가액 계산
    def get_current_val(row):
        price = current_prices[row['Ticker']]
        # 미국 주식이면 환율 적용 (간이 로직)
        if row['Ticker'].isalpha(): 
            return price * usd_krw
        return price

    df['CurrentPrice'] = df.apply(get_current_val, axis=1)
    df['Valuation'] = df['CurrentPrice'] * df['Quantity'] # 평가금액
    df['Invested'] = df['AvgPrice'] * df['Quantity']      # 투자원금
    df['PnL'] = df['Valuation'] - df['Invested']          # 손익
    df['Return(%)'] = (df['PnL'] / df['Invested']) * 100  # 수익률
    
    # 4. 전체 포트폴리오 지표 계산
    total_asset = df['Valuation'].sum()
    total_invested = df['Invested'].sum()
    total_return = ((total_asset - total_invested) / total_invested) * 100

    # 5. 화면 상단 메트릭 표시
    col1, col2, col3 = st.columns(3)
    col1.metric("총 운용 자산 (AUM)", f"{total_asset:,.0f} 원")
    col2.metric("총 수익률", f"{total_return:.2f}%")
    col3.metric("평가 손익", f"{total_asset - total_invested:,.0f} 원")

    # 6. 포트폴리오 차트 (가상 백테스팅: 현재 비중대로 1년 전부터 보유했다고 가정)
    # 각 종목의 일별 변동폭에 비중을 곱해 포트폴리오 지수 산출
    normalized = hist_data / hist_data.iloc[0] # 시작일 기준 1로 정규화
    
    # 포트폴리오 가치 변화 시뮬레이션
    weights = df['Valuation'] / total_asset
    # 단순화를 위해 환율 효과 제외하고 종목 변동성만 반영
    portfolio_nav = pd.DataFrame()
    for ticker in tickers:
        portfolio_nav[ticker] = normalized[ticker] * weights[df[df['Ticker']==ticker].index[0]]
    
    portfolio_curve = portfolio_nav.sum(axis=1) * total_invested # 원금 기준 변화

    # MDD 계산
    peak = portfolio_curve.cummax()
    drawdown = (portfolio_curve - peak) / peak
    mdd = drawdown.min() * 100

    # Sharpe Ratio 계산 (무위험이자율 2% 가정)
    daily_ret = portfolio_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252))

    # 추가 메트릭
    col1, col2 = st.columns(2)
    col1.metric("MDD (최대 낙폭)", f"{mdd:.2f}%")
    col2.metric("Sharpe Ratio (샤프 지수)", f"{sharpe:.2f}")

    # 차트 그리기
    st.subheader("📊 포트폴리오 성과 추이 (NAV)")
    st.line_chart(portfolio_curve)

    # 7. 보유 종목 상세 표
    st.subheader("📋 펀드 보유 종목 (Holdings)")
    st.dataframe(df[['Name', 'Ticker', 'Quantity', 'AvgPrice', 'CurrentPrice', 'Return(%)', 'Valuation']].style.format({
        'AvgPrice': "{:,.0f}",
        'CurrentPrice': "{:,.0f}",
        'Return(%)': "{:.2f}%",
        'Valuation': "{:,.0f}"
    }))

else:
    st.write("보유 종목 데이터가 없습니다.")
