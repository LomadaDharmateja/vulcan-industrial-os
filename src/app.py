import streamlit as st
import pandas as pd
from main import IndustrialAI
from tools.data_tools import calculate_risk_scores
from tools.search_tools import get_live_market_news


# --- PAGE CONFIG ---
st.set_page_config(page_title="VULCAN AI | Industrial OS", layout="wide", page_icon="⚡")

# --- CUSTOM MODERN CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fix for the "White Box" issue in Predictive Analytics */
    [data-testid="stMetric"] {
        background-color: #1E293B !important; /* Deep Slate Blue */
        border: 1px solid #334155 !important;
        padding: 15px !important;
        border-radius: 12px !important;
        color: #F8F9FA !important;
    }

    /* Ensure metric labels are visible */
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important; /* Soft Grey */
        font-size: 0.9rem !important;
    }

    /* Ensure metric values (numbers) are bright white */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #F8F9FA;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION ---
if "agent" not in st.session_state:
    st.session_state.agent = IndustrialAI()

def clean_response(output):
    if isinstance(output, list):
        return " ".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in output])
    return str(output)

# --- TOP NAVIGATION / HEADER ---
st.markdown('<h1 class="main-header">⚡ VULCAN Industrial OS</h1>', unsafe_allow_html=True)
st.caption("Intelligence Layer for Predictive Maintenance & Global Supply Chain")

# --- TABS SYSTEM ---
tab_dashboard, tab_predictive, tab_chat = st.tabs(["📊 Live Dashboard", "🔮 Predictive Analytics", "💬 AI Intelligence"])

with tab_dashboard:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown("### Fleet Telemetry")
        maint_df = pd.read_csv('data/maintenance.csv').tail(10)
        st.dataframe(maint_df, use_container_width=True)
        
    with col_b:
        st.markdown("### Market Indices")
        comm_df = pd.read_csv('data/commodity.csv').tail(20)
        st.line_chart(comm_df.set_index('date')['price_nominal_usd'])
        
    st.markdown("---")
    if st.button("🚀 Run Executive Audit", type="primary"):
        with st.spinner("Synthesizing Multi-Source Intelligence..."):
            report = st.session_state.agent.run_analysis()
            st.success("Audit Complete")
            st.info(report)
            st.download_button("📩 Export Audit", report, file_name="executive_audit.txt")

with tab_predictive:
    st.markdown("### Proactive Health Monitoring")
    risk_df = calculate_risk_scores()
    
    # Modern Metric Row
    cols = st.columns(len(risk_df))
    for i, (index, row) in enumerate(risk_df.iterrows()):
        score = int(row['health_score'])
        with cols[i % 5]:
            st.metric(
                label=f"Unit {row['Product ID'][-5:]}", 
                value=f"{score}%", 
                delta=f"{score-100}%" if score < 100 else "Stable"
            )
            if score < 75:
                st.error("Action Required")

with tab_chat:
    st.markdown("### Consult System Engineer")
    
    # Chat container
    chat_container = st.container()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query the knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Accessing SQL & Technical Manuals..."):
                response = st.session_state.agent.executor.invoke({"input": prompt})
                output = clean_response(response["output"])
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092063.png", width=100)
    st.title("System Controls")
    
    st.subheader("📡 Global Alerts")
    if st.checkbox("Enable Live Feed"):
        alerts = get_live_market_news("Industrial supply chain news 2026")
        st.caption(alerts[:400] + "...")
    
    st.markdown("---")
    st.info(f"**Compute:** NVIDIA RTX 3050 Ti\n\n**Brain:** Gemini 3 Flash")