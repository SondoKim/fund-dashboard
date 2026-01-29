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
    page_title="APKA 투자자문 성과 분석",
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

st.title("📈 펀드 운용 성과 대시보드")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (구글 시트 연동)
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

    # 3) 환율 데이터
    try:
        df_exchange = conn.read(worksheet="ExchangeRate", ttl=0)
        if 'Date' in df_exchange.columns and 'USD_KRW' in df_exchange.columns:
            df_exchange['Date'] = pd.to_datetime(df_exchange['Date'])
            df_exchange = df_exchange.set_index('Date').sort_index()
            df_exchange = df_exchange[~df_exchange.index.duplicated(keep='last')]
        else:
            df_exchange = pd.DataFrame()
    except Exception:
        df_exchange = pd.DataFrame()

    # 필수 컬럼 체크
    required_cols = ['Ticker', 'Name', 'Quantity', 'AvgPrice', 'EntryDate']
    if not all(col in df_port.columns for col in required_cols):
        st.error("Holdings 시트의 필수 컬럼(Ticker, Name, Quantity, AvgPrice, EntryDate)을 확인해주세요.")
        st.stop()
        
    # 데이터 전처리
    df_port['EntryDate'] = pd.to_datetime(df_port['EntryDate'])
    if 'ExitDate' not in df_port.columns: df_port['ExitDate'] = pd.NaT
    df_port['ExitDate'] = pd.to_datetime(df_port['ExitDate'])
    
    today = pd.Timestamp(date.today())
    df_port['IsHeld'] = df_port['ExitDate'].isna() | (df_port['ExitDate'] > today)

except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바: 벤치마크 설정
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 분석 설정")

market_indices = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "KOSPI": "^KS11",
    "KOSPI 200": "^KS200"
}

selected_indices = st.sidebar.multiselect(
    "벤치마크 지수 선택",
    options=list(market_indices.keys()),
    default=["S&P 500", "KOSPI"]
)

use_custom_bm = False
if not df_bm_custom.empty and 'Ticker' in df_bm_custom.columns and 'Weight' in df_bm_custom.columns:
    use_custom_bm = st.sidebar.checkbox("커스텀 벤치마크 포함", value=True)

# -----------------------------------------------------------------------------
# 4. 분석 엔진 및 시각화
# -----------------------------------------------------------------------------
with st.spinner('데이터 수집 및 성과 분석 중입니다...'):
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
        raw_data = raw_data.ffill().bfill()
        
        # 환율 데이터 동기화
        if not df_exchange.empty:
            exchange_series = df_exchange['USD_KRW'].reindex(raw_data.index, method='ffill').fillna(1450.0)
        else:
            exchange_series = pd.Series(1450.0, index=raw_data.index)
        
        current_exchange_rate = exchange_series.iloc[-1]
        current_prices = raw_data.iloc[-1] # 현재가 미리 추출

        # -----------------------------------------------------
        # (1) 내 펀드 NAV 계산 (Time Series)
        # -----------------------------------------------------
        my_nav_series = pd.Series(0.0, index=raw_data.index)
        
        for idx, row in df_port.iterrows():
            ticker = row['Ticker']
            if ticker not in raw_data.columns: continue
            
            price_s = raw_data[ticker].copy()
            # 해외 주식 환율 적용
            if ".KS" not in ticker and ".KQ" not in ticker:
                price_s = price_s * exchange_series
            
            # 보유 기간 적용
            entry, exit_d = row['EntryDate'], row['ExitDate']
            if pd.isna(exit_d):
                mask = (price_s.index >= entry)
            else:
                mask = (price_s.index >= entry) & (price_s.index <= exit_d)
            
            my_nav_series = my_nav_series.add(price_s[mask] * row['Quantity'], fill_value=0)
            
        my_nav_series = my_nav_series[my_nav_series > 0]
        
        if my_nav_series.empty:
            st.warning("표시할 데이터가 없습니다.")
            st.stop()

        # -----------------------------------------------------
        # (2) [NEW] 상단 핵심 메트릭 (AUM, 수익률, MDD, Sharpe)
        # -----------------------------------------------------
        # A. 현재 운용 규모 (AUM) 계산
        current_aum = 0
        for idx, row in df_port[df_port['IsHeld']].iterrows():
            t = row['Ticker']
            p = current_prices.get(t, 0)
            ex = 1.0 if (".KS" in t or ".KQ" in t) else current_exchange_rate
            current_aum += (p * row['Quantity'] * ex)

        # B. 전체 성과 지표 (내 펀드)
        my_ret, my_mdd, my_sharpe = calculate_metrics(my_nav_series)

        # 화면 출력
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 운용 자산 (AUM)", f"{current_aum:,.0f} 원")
        m2.metric("총 누적 수익률", f"{my_ret:+.2f}%", delta_color="normal")
        m3.metric("최대 낙폭 (MDD)", f"{my_mdd:.2f}%", delta_color="inverse")
        m4.metric("샤프 지수 (Sharpe)", f"{my_sharpe:.2f}")
        
        st.markdown("---")

        # -----------------------------------------------------
        # (3) 벤치마크 및 비교 분석
        # -----------------------------------------------------
        common_start_date = my_nav_series.index[0]
        chart_df = pd.DataFrame()
        
        # 내 펀드 추가
        chart_df['My Fund'] = (my_nav_series / my_nav_series.iloc[0]) - 1
        metrics_summary = [["My Fund", my_ret, my_mdd, my_sharpe]] # 위에서 계산한 값 재사용

        # 시장 지수 추가
        for name in selected_indices:
            ticker = market_indices[name]
            if ticker in raw_data.columns:
                bm_series = raw_data[ticker][common_start_date:]
                if not bm_series.empty:
                    # 시장지수는 이미 통화 기준이므로 환율 곱할 필요 없음 (수익률 비교)
                    chart_df[name] = (bm_series / bm_series.iloc[0]) - 1
                    r, m, s = calculate_metrics(bm_series)
                    metrics_summary.append([name, r, m, s])

        # 커스텀 벤치마크 추가
        if use_custom_bm:
            custom_bm_series = pd.Series(0.0, index=raw_data.index)
            w_sum = 0
            for idx, row in df_bm_custom.iterrows():
                t, w = row['Ticker'], row['Weight']
                if t in raw_data.columns:
                    norm_p = raw_data[t] / raw_data[t].iloc[0] * 100
                    custom_bm_series += (norm_p * w)
                    w_sum += w
            
            if w_sum > 0:
                custom_bm_series = custom_bm_series[common_start_date:]
                chart_df['Custom BM'] = (custom_bm_series / custom_bm_series.iloc[0]) - 1
                r, m, s = calculate_metrics(custom_bm_series)
                metrics_summary.append(["Custom BM", r, m, s])

        # -----------------------------------------------------
        # (4) 시각화: 성과 요약표 및 차트
        # -----------------------------------------------------
        st.subheader("📊 성과 비교 요약")
        metrics_df = pd.DataFrame(metrics_summary, columns=["구분", "총 수익률(%)", "MDD(%)", "Sharpe"])
        
        # 수익률 기준 정렬 (옵션)
        # metrics_df = metrics_df.sort_values(by="총 수익률(%)", ascending=False)
        
        st.dataframe(
            metrics_df.style.format({
                "총 수익률(%)": "{:+.2f}%",
                "MDD(%)": "{:.2f}%",
                "Sharpe": "{:.2f}"
            }).background_gradient(subset=['총 수익률(%)'], cmap='RdYlGn'),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("📈 누적 수익률 추이 (Benchmark Comparison)")
        # 수익률 퍼센트 변환
        st.line_chart(chart_df * 100, color=["#FF0000"] + ["#AAAAAA"]*(len(chart_df.columns)-1))
        
        st.markdown("---")

        # -----------------------------------------------------
        # (5) 상세 종목 내역
        # -----------------------------------------------------
        # 1. 현재 보유
        st.subheader(f"🔵 현재 보유 자산 상세 (적용 환율: {current_exchange_rate:,.1f}원)")
        if not df_port[df_port['IsHeld']].empty:
            active_df = df_port[df_port['IsHeld']].copy()
            
            def calc_active(row):
                t = row['Ticker']
                p = current_prices.get(t, 0)
                ex = 1.0 if (".KS" in t or ".KQ" in t) else current_exchange_rate
                val = p * row['Quantity'] * ex
                inv = row['AvgPrice'] * row['Quantity']
                ret = ((val - inv)/inv)*100 if inv!=0 else 0
                return pd.Series([p, val, ret])
            
            active_df[['CurrentPrice', 'Valuation', 'Return(%)']] = active_df.apply(calc_active, axis=1)
            
            st.dataframe(
                active_df[['Name', 'Ticker', 'EntryDate', 'Quantity', 'AvgPrice', 'CurrentPrice', 'Valuation', 'Return(%)']].style.format({
                    'AvgPrice': "{:,.0f}", 'CurrentPrice': "{:,.2f}", 'Valuation': "{:,.0f}", 'Return(%)': "{:+.2f}%", 'EntryDate': "{:%Y-%m-%d}"
                }).background_gradient(subset=['Return(%)'], cmap='RdYlGn', vmin=-30, vmax=30),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("보유 종목이 없습니다.")

        # 2. 청산 내역
        st.subheader("⚪️ 실현 손익 내역 (청산 완료)")
        if not df_port[~df_port['IsHeld']].empty:
            exited_df = df_port[~df_port['IsHeld']].copy()
            
            def calc_exit(row):
                t = row['Ticker']
                exit_d = row['ExitDate']
                if t in raw_data.columns:
                    p = raw_data[t].asof(exit_d)
                    if pd.isna(p): p=0
                else: p=0
                
                is_kr = ".KS" in t or ".KQ" in t
                ex = 1.0 if is_kr else exchange_series.asof(exit_d)
                if pd.isna(ex): ex=1450.0
                
                sell_amt = p * row['Quantity'] * ex
                buy_amt = row['AvgPrice'] * row['Quantity']
                pnl = sell_amt - buy_amt
                ret = (pnl/buy_amt)*100 if buy_amt!=0 else 0
                return pd.Series([p, pnl, ret])
            
            exited_df[['ExitPrice', 'PnL', 'Return(%)']] = exited_df.apply(calc_exit, axis=1)
            
            st.dataframe(
                exited_df[['Name', 'Ticker', 'EntryDate', 'ExitDate', 'AvgPrice', 'ExitPrice', 'PnL', 'Return(%)']].style.format({
                    'AvgPrice': "{:,.0f}", 'ExitPrice': "{:,.2f}", 'PnL': "{:,.0f}", 'Return(%)': "{:+.2f}%", 'EntryDate': "{:%Y-%m-%d}", 'ExitDate': "{:%Y-%m-%d}"
                }).background_gradient(subset=['Return(%)'], cmap='RdYlGn', vmin=-30, vmax=30),
                use_container_width=True, hide_index=True
            )
            
        st.markdown("---")
        
        # 상관관계
        with st.expander("📊 상관관계 분석 보기"):
            st.dataframe(chart_df.pct_change().corr().style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))

    else:
        st.warning("데이터를 불러올 수 없습니다.")
