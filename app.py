import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, date
from streamlit_gsheets import GSheetsConnection

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 함수 정의
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="투자자문 성과 비교 분석",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 성과 지표 계산 함수
def calculate_metrics(daily_series, risk_free_rate=0.02):
    if daily_series.empty:
        return 0, 0, 0
    
    total_ret = ((daily_series.iloc[-1] - daily_series.iloc[0]) / daily_series.iloc[0]) * 100
    
    rolling_max = daily_series.cummax()
    drawdown = (daily_series - rolling_max) / rolling_max
    mdd = drawdown.min() * 100
    
    daily_pct = daily_series.pct_change().dropna()
    if daily_pct.std() != 0:
        sharpe = (daily_pct.mean() * 252 - risk_free_rate) / (daily_pct.std() * np.sqrt(252))
    else:
        sharpe = 0
        
    return total_ret, mdd, sharpe

st.title("📈 펀드 성과 vs 벤치마크 비교")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (내 포트폴리오, 벤치마크, 환율)
# -----------------------------------------------------------------------------
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # 1) 내 포트폴리오
    df_port = conn.read(worksheet="Holdings", ttl=0)
    
    # 2) 커스텀 벤치마크
    try:
        df_bm_custom = conn.read(worksheet="Benchmark", ttl=0)
    except:
        df_bm_custom = pd.DataFrame()

    # 3) [NEW] 환율 데이터 로드
    try:
        df_exchange = conn.read(worksheet="ExchangeRate", ttl=0)
        # 컬럼 확인 및 전처리
        if 'Date' in df_exchange.columns and 'USD_KRW' in df_exchange.columns:
            df_exchange['Date'] = pd.to_datetime(df_exchange['Date'])
            df_exchange = df_exchange.set_index('Date').sort_index()
            # 중복 날짜 제거 (혹시 모를 오류 방지)
            df_exchange = df_exchange[~df_exchange.index.duplicated(keep='last')]
        else:
            st.warning("'ExchangeRate' 시트의 컬럼명은 Date, USD_KRW 여야 합니다. (기본값 1450원 적용)")
            df_exchange = pd.DataFrame()
    except Exception:
        st.warning("환율 시트(ExchangeRate)를 찾을 수 없습니다. (기본값 1450원 적용)")
        df_exchange = pd.DataFrame()

    # 필수 컬럼 체크
    required_cols = ['Ticker', 'Name', 'Quantity', 'AvgPrice', 'EntryDate']
    if not all(col in df_port.columns for col in required_cols):
        st.error("Holdings 시트에 필수 컬럼이 누락되었습니다.")
        st.stop()
        
    # 날짜 전처리
    df_port['EntryDate'] = pd.to_datetime(df_port['EntryDate'])
    if 'ExitDate' not in df_port.columns: df_port['ExitDate'] = pd.NaT
    df_port['ExitDate'] = pd.to_datetime(df_port['ExitDate'])
    
    today = pd.Timestamp(date.today())
    df_port['IsHeld'] = df_port['ExitDate'].isna() | (df_port['ExitDate'] > today)

except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바: 벤치마크 설정
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 벤치마크 설정")

market_indices = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "KOSPI": "^KS11",
    "KOSPI 200": "^KS200"
}

selected_indices = st.sidebar.multiselect(
    "시장 지수 비교",
    options=list(market_indices.keys()),
    default=["S&P 500", "KOSPI"]
)

use_custom_bm = False
if not df_bm_custom.empty and 'Ticker' in df_bm_custom.columns and 'Weight' in df_bm_custom.columns:
    use_custom_bm = st.sidebar.checkbox("커스텀 벤치마크 포함", value=True)

# -----------------------------------------------------------------------------
# 4. 데이터 수집 및 분석 엔진
# -----------------------------------------------------------------------------
with st.spinner('시장 데이터 및 환율 정보를 분석 중...'):
    port_tickers = df_port['Ticker'].unique().tolist()
    bm_tickers = [market_indices[name] for name in selected_indices]
    if use_custom_bm:
        bm_tickers += df_bm_custom['Ticker'].unique().tolist()
    
    all_tickers = list(set(port_tickers + bm_tickers))
    
    if len(all_tickers) > 0:
        start_date = df_port['EntryDate'].min()
        
        # 주가 데이터 다운로드
        raw_data = yf.download(all_tickers, start=start_date, end=date.today())['Close']
        if isinstance(raw_data, pd.Series): raw_data = raw_data.to_frame(name=all_tickers[0])
        raw_data = raw_data.ffill().bfill() # 주가 결측치 채움
        
        # [NEW] 환율 데이터 동기화 (주가 데이터 날짜에 맞춤)
        # 환율 시트 데이터가 있으면 사용, 없으면 고정값(1450)
        if not df_exchange.empty:
            # 주가 데이터 인덱스(날짜)에 맞춰 환율 데이터 재정렬 (빈 날짜는 직전 환율로 채움 ffill)
            exchange_series = df_exchange['USD_KRW'].reindex(raw_data.index, method='ffill').fillna(1450.0)
        else:
            exchange_series = pd.Series(1450.0, index=raw_data.index)
            
        # 최신 환율 (현재가 계산용)
        current_exchange_rate = exchange_series.iloc[-1]

        # -----------------------------------------------------
        # (1) 내 펀드 NAV 계산 (Time Series)
        # -----------------------------------------------------
        my_nav_series = pd.Series(0.0, index=raw_data.index)
        
        for idx, row in df_port.iterrows():
            ticker = row['Ticker']
            if ticker not in raw_data.columns: continue
            
            price_s = raw_data[ticker].copy()
            
            # [NEW] 환율 적용 로직
            # 한국 주식이 아니면 날짜별 환율 곱하기
            if ".KS" not in ticker and ".KQ" not in ticker:
                price_s = price_s * exchange_series
            
            entry, exit_d = row['EntryDate'], row['ExitDate']
            
            # 보유 기간 마스킹
            if pd.isna(exit_d):
                mask = (price_s.index >= entry)
            else:
                mask = (price_s.index >= entry) & (price_s.index <= exit_d)
            
            # 가치 합산 (가격 * 환율 * 수량)
            my_nav_series = my_nav_series.add(price_s[mask] * row['Quantity'], fill_value=0)
            
        my_nav_series = my_nav_series[my_nav_series > 0]
        
        if my_nav_series.empty:
            st.warning("데이터 부족으로 차트를 그릴 수 없습니다.")
            st.stop()
            
        common_start_date = my_nav_series.index[0]
        
        # 차트용 DF 생성
        chart_df = pd.DataFrame()
        my_return_curve = (my_nav_series / my_nav_series.iloc[0]) - 1
        chart_df['My Fund'] = my_return_curve * 100
        
        metrics_summary = []
        ret, mdd, sharpe = calculate_metrics(my_nav_series)
        metrics_summary.append(["My Fund", ret, mdd, sharpe])

        # -----------------------------------------------------
        # (2) 벤치마크 계산 (시장 지수)
        # -----------------------------------------------------
        # 시장 지수는 이미 해당 통화(USD/KRW) 기준이므로 환율 곱할 필요 없음 (수익률 비교이므로)
        for name in selected_indices:
            ticker = market_indices[name]
            if ticker in raw_data.columns:
                bm_series = raw_data[ticker][common_start_date:]
                if not bm_series.empty:
                    bm_curve = (bm_series / bm_series.iloc[0]) - 1
                    chart_df[name] = bm_curve * 100
                    ret, mdd, sharpe = calculate_metrics(bm_series)
                    metrics_summary.append([name, ret, mdd, sharpe])

        # -----------------------------------------------------
        # (3) 커스텀 벤치마크 계산
        # -----------------------------------------------------
        if use_custom_bm:
            custom_bm_series = pd.Series(0.0, index=raw_data.index)
            valid_weight = 0
            for idx, row in df_bm_custom.iterrows():
                t, w = row['Ticker'], row['Weight']
                if t in raw_data.columns:
                    # 커스텀 벤치마크는 '지수' 개념이므로 환율 변동을 굳이 태우지 않고 원화 수익률 관점에서 봅니다.
                    # (만약 벤치마크도 환헤지 안 된 달러 자산이라면 환율 곱해야 하지만, 여기선 단순화)
                    normalized = raw_data[t] / raw_data[t].iloc[0] * 100
                    custom_bm_series += (normalized * w)
                    valid_weight += w
            
            if valid_weight > 0:
                custom_bm_series = custom_bm_series[common_start_date:]
                bm_curve = (custom_bm_series / custom_bm_series.iloc[0]) - 1
                chart_df['Custom BM'] = bm_curve * 100
                ret, mdd, sharpe = calculate_metrics(custom_bm_series)
                metrics_summary.append(["Custom BM", ret, mdd, sharpe])

        # -----------------------------------------------------
        # 5. 시각화 (상단)
        # -----------------------------------------------------
        st.subheader("📊 성과 요약")
        metrics_df = pd.DataFrame(metrics_summary, columns=["구분", "총 수익률(%)", "MDD(%)", "Sharpe"])
        st.dataframe(metrics_df.style.format({
            "총 수익률(%)": "{:+.2f}%", "MDD(%)": "{:.2f}%", "Sharpe": "{:.2f}"
        }).background_gradient(subset=['총 수익률(%)'], cmap='RdYlGn'), hide_index=True, use_container_width=True)
        
        st.subheader("📈 누적 수익률 추이 비교")
        st.line_chart(chart_df, color=["#FF0000"] + ["#AAAAAA"]*(len(chart_df.columns)-1))

        st.markdown("---")

        # =========================================================================
        # 상세 종목 내역 (환율 적용)
        # =========================================================================
        
        # 현재가 가져오기
        current_prices = raw_data.iloc[-1]
        
        # 1. 현재 보유 포트폴리오
        st.subheader(f"🔵 현재 보유 자산 (적용 환율: {current_exchange_rate:,.1f}원)")
        
        if not df_port[df_port['IsHeld']].empty:
            active_df = df_port[df_port['IsHeld']].copy()
            
            # 계산 로직
            def calc_active_stats(row):
                ticker = row['Ticker']
                curr_price = current_prices.get(ticker, 0)
                
                # [NEW] 현재 시점 환율 적용
                is_kr_stock = ".KS" in ticker or ".KQ" in ticker
                exchange = 1.0 if is_kr_stock else current_exchange_rate
                
                valuation = curr_price * row['Quantity'] * exchange
                invested = row['AvgPrice'] * row['Quantity'] 
                ret_pct = ((valuation - invested) / invested) * 100 if invested != 0 else 0
                return pd.Series([curr_price, valuation, ret_pct])

            active_df[['CurrentPrice', 'Valuation', 'Return(%)']] = active_df.apply(calc_active_stats, axis=1)
            
            st.dataframe(
                active_df[['Name', 'Ticker', 'EntryDate', 'Quantity', 'AvgPrice', 'CurrentPrice', 'Valuation', 'Return(%)']].style.format({
                    'AvgPrice': "{:,.0f}", 
                    'CurrentPrice': "{:,.2f}", 
                    'Valuation': "{:,.0f}", 
                    'Return(%)': "{:+.2f}%",
                    'EntryDate': "{:%Y-%m-%d}"
                }).background_gradient(subset=['Return(%)'], cmap='RdYlGn', vmin=-30, vmax=30),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("현재 보유 중인 종목이 없습니다.")

        # 2. 청산(매도) 완료 내역
        st.subheader("⚪️ 실현 손익 내역")
        
        if not df_port[~df_port['IsHeld']].empty:
            exited_df = df_port[~df_port['IsHeld']].copy()
            
            def calc_realized_stats(row):
                ticker = row['Ticker']
                exit_date = row['ExitDate']
                
                # 매도일 당시 가격
                if ticker in raw_data.columns:
                    exit_price = raw_data[ticker].asof(exit_date)
                    if pd.isna(exit_price): exit_price = 0
                else:
                    exit_price = 0
                
                # [NEW] 매도일 당시 환율
                is_kr_stock = ".KS" in ticker or ".KQ" in ticker
                if is_kr_stock:
                    exchange = 1.0
                else:
                    # 매도일(exit_date) 시점의 환율 가져오기 (asof)
                    exchange = exchange_series.asof(exit_date)
                    if pd.isna(exchange): exchange = 1450.0 # 예외처리
                
                sell_amt = exit_price * row['Quantity'] * exchange
                buy_amt = row['AvgPrice'] * row['Quantity']
                pnl = sell_amt - buy_amt
                ret_pct = (pnl / buy_amt) * 100 if buy_amt != 0 else 0
                
                return pd.Series([exit_price, pnl, ret_pct])

            exited_df[['ExitPrice', 'PnL', 'Return(%)']] = exited_df.apply(calc_realized_stats, axis=1)

            st.dataframe(
                exited_df[['Name', 'Ticker', 'EntryDate', 'ExitDate', 'AvgPrice', 'ExitPrice', 'PnL', 'Return(%)']].style.format({
                    'AvgPrice': "{:,.0f}", 
                    'ExitPrice': "{:,.2f}", 
                    'PnL': "{:,.0f}", 
                    'Return(%)': "{:+.2f}%",
                    'EntryDate': "{:%Y-%m-%d}",
                    'ExitDate': "{:%Y-%m-%d}"
                }).background_gradient(subset=['Return(%)'], cmap='RdYlGn', vmin=-30, vmax=30),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("청산된 내역이 없습니다.")
            
        st.markdown("---")

        # 상관관계 분석
        with st.expander("📊 상관관계 분석 (Correlation) 보기"):
            corr_matrix = chart_df.pct_change().corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))

    else:
        st.warning("데이터를 불러올 수 없습니다.")
