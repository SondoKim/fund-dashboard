import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, date
from streamlit_gsheets import GSheetsConnection

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="泰투자자문 포트폴리오",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 泰투자자문 펀드 운용 현황")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="Holdings", ttl=0) # ttl=0 : 캐시 없이 즉시 로딩
    
    # 필수 컬럼 체크
    required_cols = ['Ticker', 'Name', 'Quantity', 'AvgPrice', 'EntryDate']
    if not all(col in df.columns for col in required_cols):
        st.error("구글 시트 컬럼 부족. (Ticker, Name, Quantity, AvgPrice, EntryDate, ExitDate 확인 필요)")
        st.stop()

    # ExitDate 컬럼이 아예 없으면 생성 (에러 방지)
    if 'ExitDate' not in df.columns:
        df['ExitDate'] = pd.NaT

    # 날짜 형식 변환
    df['EntryDate'] = pd.to_datetime(df['EntryDate'])
    df['ExitDate'] = pd.to_datetime(df['ExitDate'])
    
    # 현재 보유중인 종목 (ExitDate가 비어있거나 미래인 경우)
    today = pd.Timestamp(date.today())
    df['IsHeld'] = df['ExitDate'].isna() | (df['ExitDate'] > today)

except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 시장 데이터 수집 및 시계열 분석 엔진
# -----------------------------------------------------------------------------
with st.spinner('전체 기간 데이터를 분석 중입니다...'):
    tickers = df['Ticker'].unique().tolist()
    USD_KRW = 1450.0  # 환율 설정
    
    if len(tickers) > 0:
        # 1. 전체 기간(가장 빠른 편입일 ~ 오늘) 데이터 가져오기
        start_date = df['EntryDate'].min()
        hist_data = yf.download(tickers, start=start_date, end=date.today())['Close']
        
        # 단일 종목일 경우 Series -> DataFrame 변환
        if isinstance(hist_data, pd.Series):
            hist_data = hist_data.to_frame(name=tickers[0])
            
        # 결측치 보간 (휴장일 등)
        hist_data = hist_data.ffill().bfill()
        
        # ---------------------------------------------------------
        # [핵심 로직] 일별 포트폴리오 가치 산출 (History Curve)
        # ---------------------------------------------------------
        # 날짜별 총 자산 가치를 담을 0으로 된 시리즈 생성
        portfolio_series = pd.Series(0.0, index=hist_data.index)
        
        # 각 종목(행)별로 루프를 돌며 자산 가치를 더함
        for idx, row in df.iterrows():
            ticker = row['Ticker']
            qty = row['Quantity']
            entry = row['EntryDate']
            exit_d = row['ExitDate']
            
            if ticker not in hist_data.columns:
                continue # 티커 데이터가 없으면 스킵

            # 해당 종목의 전체 가격 데이터
            price_series = hist_data[ticker].copy()
            
            # 환율 적용 (국내 주식이 아니면)
            if ".KS" not in ticker and ".KQ" not in ticker:
                price_series = price_series * USD_KRW

            # 유효 보유 기간 마스크 생성 (Entry <= Date <= Exit)
            # ExitDate가 없으면(NaT) 오늘까지 보유한 것으로 처리
            if pd.isna(exit_d):
                mask = (price_series.index >= entry)
            else:
                mask = (price_series.index >= entry) & (price_series.index <= exit_d)
            
            # 보유 기간 동안의 가치 = 가격 * 수량
            asset_value = price_series[mask] * qty
            
            # 전체 포트폴리오에 합산 (인덱스 매칭되어 날짜별로 더해짐)
            portfolio_series = portfolio_series.add(asset_value, fill_value=0)

        # ---------------------------------------------------------
        # 4. 현황 지표 계산 (현재 시점)
        # ---------------------------------------------------------
        
        # A. 현재 운용 자산 (AUM): 현재 보유 중인 종목들의 평가액 합
        # (마지막 날짜 기준 가격으로 계산)
        current_prices = hist_data.iloc[-1]
        
        total_aum = 0
        total_invested_active = 0 # 현재 보유분의 투자원금
        
        for idx, row in df[df['IsHeld']].iterrows():
            ticker = row['Ticker']
            if ticker in current_prices:
                price = current_prices[ticker]
                exchange = 1.0 if (".KS" in ticker or ".KQ" in ticker) else USD_KRW
                val = price * row['Quantity'] * exchange
                
                total_aum += val
                total_invested_active += (row['AvgPrice'] * row['Quantity'])

        # B. 실현 손익 (Realized PnL): 이미 매도한 종목들의 확정 손익
        realized_pnl = 0
        for idx, row in df[~df['IsHeld']].iterrows():
            # 매도일의 가격 찾기
            exit_date_lookup = row['ExitDate']
            # 매도일이 데이터 범위 내에 있는지 확인 (휴장일이면 직전 평일 찾기)
            if row['Ticker'] in hist_data.columns:
                try:
                    # asof: 해당 날짜 혹은 그 전 가장 가까운 날짜의 가격
                    exit_price = hist_data[row['Ticker']].asof(exit_date_lookup)
                    exchange = 1.0 if (".KS" in row['Ticker'] or ".KQ" in row['Ticker']) else USD_KRW
                    
                    sell_amt = exit_price * row['Quantity'] * exchange
                    buy_amt = row['AvgPrice'] * row['Quantity']
                    realized_pnl += (sell_amt - buy_amt)
                except:
                    pass # 데이터 매칭 실패 시 스킵

        # C. 미실현 손익 (Unrealized PnL)
        unrealized_pnl = total_aum - total_invested_active
        
        # D. 총 수익금 (실현 + 미실현)
        total_profit = realized_pnl + unrealized_pnl

        # MDD, Sharpe 계산
        mdd_val = 0
        sharpe_val = 0
        
        if not portfolio_series.empty and portfolio_series.max() > 0:
            # MDD
            rolling_max = portfolio_series.cummax()
            drawdown = (portfolio_series - rolling_max) / rolling_max
            mdd_val = drawdown.min() * 100
            
            # Sharpe
            daily_ret = portfolio_series.pct_change().dropna()
            if daily_ret.std() != 0:
                sharpe_val = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252))

        # ---------------------------------------------------------
        # 5. 화면 출력
        # ---------------------------------------------------------
        
        # 메트릭 표시
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("현재 운용 자산 (AUM)", f"{total_aum:,.0f} 원")
        col2.metric("총 실현 손익 (Realized)", f"{realized_pnl:,.0f} 원", 
                    delta_color="normal" if realized_pnl >=0 else "inverse")
        col3.metric("현재 평가 손익 (Unrealized)", f"{unrealized_pnl:,.0f} 원",
                    delta_color="normal" if unrealized_pnl >=0 else "inverse")
        col4.metric("MDD (History)", f"{mdd_val:.2f} %")
        col5.metric("Sharpe Ratio", f"{sharpe_val:.2f}")

        # 차트
        st.subheader("📊 펀드 전체 자산 추이 (History)")
        st.line_chart(portfolio_series, color="#FF4B4B")

        # 테이블 1: 현재 보유 종목
        st.subheader("🔵 현재 보유 포트폴리오")
        if not df[df['IsHeld']].empty:
            active_df = df[df['IsHeld']].copy()
            # 현재가 매핑
            active_df['CurPrice'] = active_df['Ticker'].map(lambda x: current_prices.get(x, 0))
            active_df['Valuation'] = active_df.apply(
                lambda r: r['CurPrice'] * r['Quantity'] * (1.0 if ".KS" in r['Ticker'] or ".KQ" in r['Ticker'] else USD_KRW), axis=1
            )
            active_df['Return'] = (active_df['Valuation'] - (active_df['AvgPrice']*active_df['Quantity'])) / (active_df['AvgPrice']*active_df['Quantity']) * 100
            
            st.dataframe(active_df[['Name', 'Ticker', 'EntryDate', 'Quantity', 'AvgPrice', 'Valuation', 'Return']].style.format({
                'AvgPrice': "{:,.0f}", 'Valuation': "{:,.0f}", 'Return': "{:+.2f}%", 'EntryDate': "{:%Y-%m-%d}"
            }))
        else:
            st.info("현재 보유 중인 종목이 없습니다.")

        # 테이블 2: 청산 완료 종목
        st.subheader("⚪️ 청산(매도) 완료 내역")
        if not df[~df['IsHeld']].empty:
            exited_df = df[~df['IsHeld']].copy()
            # 매도 당시 가격 추정 로직 (단순화를 위해 마지막 날짜 기준이 아닌, ExitDate 기준)
            # 여기서는 편의상 리스트엔 표시만 함
            st.dataframe(exited_df[['Name', 'Ticker', 'EntryDate', 'ExitDate', 'Quantity', 'AvgPrice']].style.format({
                'AvgPrice': "{:,.0f}", 'EntryDate': "{:%Y-%m-%d}", 'ExitDate': "{:%Y-%m-%d}"
            }))
        else:
            st.info("청산된 종목 내역이 없습니다.")

    else:
        st.warning("종목 데이터가 없습니다.")
