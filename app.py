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

# 성과 지표 계산 함수 (재사용을 위해 함수로 분리)
def calculate_metrics(daily_series, risk_free_rate=0.02):
    if daily_series.empty:
        return 0, 0, 0
    
    # 총 수익률
    total_ret = ((daily_series.iloc[-1] - daily_series.iloc[0]) / daily_series.iloc[0]) * 100
    
    # MDD
    rolling_max = daily_series.cummax()
    drawdown = (daily_series - rolling_max) / rolling_max
    mdd = drawdown.min() * 100
    
    # Sharpe
    daily_pct = daily_series.pct_change().dropna()
    if daily_pct.std() != 0:
        sharpe = (daily_pct.mean() * 252 - risk_free_rate) / (daily_pct.std() * np.sqrt(252))
    else:
        sharpe = 0
        
    return total_ret, mdd, sharpe

st.title("📈 펀드 성과 vs 벤치마크 비교")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (내 포트폴리오 & 커스텀 벤치마크)
# -----------------------------------------------------------------------------
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # 1) 내 포트폴리오 (Holdings)
    df_port = conn.read(worksheet="Holdings", ttl=0)
    
    # 2) 커스텀 벤치마크 (Benchmark)
    # 시트가 없을 수도 있으므로 예외처리
    try:
        df_bm_custom = conn.read(worksheet="Benchmark", ttl=0)
    except:
        df_bm_custom = pd.DataFrame()

    # 필수 컬럼 체크
    if 'EntryDate' not in df_port.columns:
        st.error("Holdings 시트에 EntryDate 컬럼이 필요합니다.")
        st.stop()
        
    # 날짜 및 데이터 전처리
    df_port['EntryDate'] = pd.to_datetime(df_port['EntryDate'])
    if 'ExitDate' not in df_port.columns: df_port['ExitDate'] = pd.NaT
    df_port['ExitDate'] = pd.to_datetime(df_port['ExitDate'])
    
    today = pd.Timestamp(date.today())
    df_port['IsHeld'] = df_port['ExitDate'].isna() | (df_port['ExitDate'] > today)

except Exception as e:
    st.error(f"구글 시트 로드 실패: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바: 벤치마크 선택
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 벤치마크 설정")

# 시장 표준 벤치마크 정의
market_indices = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "KOSPI": "^KS11",
    "KOSPI 200": "^KS200"
}

# 1. 시장 지수 선택
selected_indices = st.sidebar.multiselect(
    "시장 지수 비교",
    options=list(market_indices.keys()),
    default=["S&P 500", "KOSPI"]
)

# 2. 커스텀 벤치마크 활성화 여부
use_custom_bm = False
if not df_bm_custom.empty and 'Ticker' in df_bm_custom.columns and 'Weight' in df_bm_custom.columns:
    use_custom_bm = st.sidebar.checkbox("커스텀 벤치마크(Sheet) 포함", value=True)

# -----------------------------------------------------------------------------
# 4. 데이터 수집 및 분석 엔진
# -----------------------------------------------------------------------------
with st.spinner('시장 데이터 수집 및 비교 분석 중...'):
    # A. 내 펀드 데이터 수집
    port_tickers = df_port['Ticker'].unique().tolist()
    
    # B. 벤치마크용 티커 수집
    bm_tickers = [market_indices[name] for name in selected_indices]
    if use_custom_bm:
        bm_tickers += df_bm_custom['Ticker'].unique().tolist()
    
    # 전체 티커 합치기 (중복 제거)
    all_tickers = list(set(port_tickers + bm_tickers))
    USD_KRW = 1450.0 # 환율

    if len(all_tickers) > 0:
        # 데이터 시작일: 내 펀드 최초 편입일
        start_date = df_port['EntryDate'].min()
        
        # 야후 파이낸스 다운로드
        raw_data = yf.download(all_tickers, start=start_date, end=date.today())['Close']
        if isinstance(raw_data, pd.Series): raw_data = raw_data.to_frame(name=all_tickers[0])
        raw_data = raw_data.ffill().bfill()
        
        # -----------------------------------------------------
        # (1) 내 펀드 NAV 계산 (이전 로직과 동일)
        # -----------------------------------------------------
        my_nav_series = pd.Series(0.0, index=raw_data.index)
        
        for idx, row in df_port.iterrows():
            ticker = row['Ticker']
            if ticker not in raw_data.columns: continue
            
            # 가격 데이터
            price_s = raw_data[ticker].copy()
            if ".KS" not in ticker and ".KQ" not in ticker:
                price_s = price_s * USD_KRW
            
            # 보유 기간 마스킹
            entry, exit_d = row['EntryDate'], row['ExitDate']
            if pd.isna(exit_d):
                mask = (price_s.index >= entry)
            else:
                mask = (price_s.index >= entry) & (price_s.index <= exit_d)
            
            my_nav_series = my_nav_series.add(price_s[mask] * row['Quantity'], fill_value=0)
            
        # 0인 구간(투자 전) 제거
        my_nav_series = my_nav_series[my_nav_series > 0]
        if my_nav_series.empty:
            st.warning("표시할 펀드 데이터가 없습니다.")
            st.stop()
            
        # 비교를 위해 "누적 수익률(%)"로 변환 (시작일 = 0%)
        # 내 펀드의 시작 날짜를 기준으로 모든 벤치마크를 자름
        common_start_date = my_nav_series.index[0]
        
        # DataFrame for Plotting (모든 라인을 여기 담음)
        chart_df = pd.DataFrame()
        
        # 1. 내 펀드 추가
        my_return_curve = (my_nav_series / my_nav_series.iloc[0]) - 1
        chart_df['My Fund'] = my_return_curve * 100
        
        # 메트릭 저장용 리스트
        metrics_summary = []
        ret, mdd, sharpe = calculate_metrics(my_nav_series)
        metrics_summary.append(["My Fund", ret, mdd, sharpe])

        # -----------------------------------------------------
        # (2) 시장 벤치마크 계산
        # -----------------------------------------------------
        for name in selected_indices:
            ticker = market_indices[name]
            if ticker in raw_data.columns:
                # 내 펀드 시작일부터 슬라이싱
                bm_series = raw_data[ticker][common_start_date:]
                # 정규화
                bm_curve = (bm_series / bm_series.iloc[0]) - 1
                chart_df[name] = bm_curve * 100
                
                # 메트릭 계산
                ret, mdd, sharpe = calculate_metrics(bm_series)
                metrics_summary.append([name, ret, mdd, sharpe])

        # -----------------------------------------------------
        # (3) 커스텀 벤치마크 계산
        # -----------------------------------------------------
        if use_custom_bm:
            # 100으로 시작하는 지수 산출 (Weighted Sum)
            custom_bm_series = pd.Series(0.0, index=raw_data.index)
            valid_weight = 0
            
            for idx, row in df_bm_custom.iterrows():
                t, w = row['Ticker'], row['Weight']
                if t in raw_data.columns:
                    # 정규화된 가격(시작일=100)에 비중을 곱함 -> 리밸런싱 없는 고정비중 바스켓 가정
                    normalized_price = raw_data[t] / raw_data[t].iloc[0] * 100
                    custom_bm_series += (normalized_price * w)
                    valid_weight += w
            
            if valid_weight > 0:
                # 내 펀드 기간과 맞춤
                custom_bm_series = custom_bm_series[common_start_date:]
                bm_curve = (custom_bm_series / custom_bm_series.iloc[0]) - 1
                chart_df['Custom BM'] = bm_curve * 100
                
                ret, mdd, sharpe = calculate_metrics(custom_bm_series)
                metrics_summary.append(["Custom BM", ret, mdd, sharpe])

        # -----------------------------------------------------
        # 5. 시각화 및 표출
        # -----------------------------------------------------
        
        # A. 성과 요약 테이블
        st.subheader("📊 성과 비교 요약")
        metrics_df = pd.DataFrame(metrics_summary, columns=["구분", "총 수익률(%)", "MDD(%)", "Sharpe"])
        
        # 스타일링 (수익률 높고, MDD 낮은 순으로 강조하면 좋겠지만 단순 표출)
        st.dataframe(
            metrics_df.style.format({
                "총 수익률(%)": "{:+.2f}%",
                "MDD(%)": "{:.2f}%",
                "Sharpe": "{:.2f}"
            }).background_gradient(subset=['총 수익률(%)'], cmap='RdYlGn'),
            hide_index=True,
            use_container_width=True
        )
        
        # B. 비교 차트
        st.subheader("📈 누적 수익률 추이 비교")
        # 색상 지정 (내 펀드는 빨강, 나머지는 자동)
        st.line_chart(chart_df, color=["#FF0000"] + ["#AAAAAA"]*(len(chart_df.columns)-1))
        
        # C. (옵션) 상관관계 분석
        with st.expander("상관관계 분석 (Correlation) 보기"):
            st.write("내 펀드와 벤치마크 간의 움직임이 얼마나 비슷한지(0~1) 보여줍니다.")
            corr_matrix = chart_df.pct_change().corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))

    else:
        st.warning("데이터를 불러올 수 없습니다.")
