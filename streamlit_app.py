"""
Streamlit Dashboard for Neural Operator Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Neural Operator", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("neural_operator_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🧠 P2Quant Neural Operator</div>', unsafe_allow_html=True)
st.markdown('<div>Fourier Neural Operator – Learning Cross‑Asset Exchange Option Prices</div>', unsafe_allow_html=True)

with st.expander("📘 How It Works", expanded=False):
    st.markdown("""
    The Neural Operator learns the mapping from the covariance surface to Margrabe exchange option prices.
    - **Fourier Neural Operator (FNO)**: Learns function-to-function mappings in the frequency domain.
    - **Margrabe Exchange Option**: Prices the right to exchange one ETF for another.
    ETFs are ranked by their average exchange option price when used as the first asset (higher = better).
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key, [])
        universe_data = universes.get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            score = pick['score']
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🧠 TOP PICK (Highest Avg Exchange Price)</div>
                <div class="hero-ticker">{ticker}</div>
                <div style="font-size: 1.5rem;">Score: {score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Picks")
            rows = []
            for p in top:
                rows.append({"Ticker": p['ticker'], "Score": f"{p['score']:.4f}"})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, s in universe_data.items():
                all_rows.append({"Ticker": t, "Score": f"{s:.4f}"})
            df_all = pd.DataFrame(all_rows).sort_values("Score", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
