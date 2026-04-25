import streamlit as st
import pandas as pd
import os
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

    [data-testid="stMetric"] {
        background-color: #1E293B !important; 
        border: 1px solid #334155 !important;
        padding: 15px !important;
        border-radius: 12px !important;
        color: #F8F9FA !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important; 
        font-size: 0.9rem !important;
    }

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

# Pre-calculate telemetry so the Audit isn't empty on startup
if "telemetry_results" not in st.session_state:
    with st.spinner("Initializing Fleet Telemetry..."):
        risk_df = calculate_risk_scores()
        # Store as dictionary for the Audit function to read easily
        st.session_state.telemetry_results = risk_df.to_dict('records')

def clean_response(output):
    if isinstance(output, list):
        return " ".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in output])
    return str(output)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092063.png", width=100)
    st.title("System Controls")
    
    st.subheader("📡 Global Alerts")
    if st.checkbox("Enable Live Feed"):
        alerts = get_live_market_news("Industrial supply chain news 2026")
        st.caption(alerts[:400] + "...")
    
    st.markdown("---")
    # Updated Brain label to reflect your switch to 2.5 Flash Lite
    st.info(f"**Compute:** Cloud Distributed\n\n**Brain:** Gemini 2.5 Flash-Lite")

# --- AUTONOMOUS AUDIT FRAGMENT ---
@st.fragment(run_every="1d")
def autonomous_daily_audit():
    st.markdown("### 📅 Daily Autonomous Audit")
    
    if "telemetry_results" in st.session_state:
        results = st.session_state.telemetry_results
        
        # Identify the "Hero" (Machine with the lowest health score)
        top_issue = min(results, key=lambda x: x['health_score'])
        
        # Display the Hero Card if health is below a threshold (e.g., 80%)
        if top_issue['health_score'] < 80:
            st.error(f"🚨 **CRITICAL ATTENTION REQUIRED: Unit {top_issue['Product ID'][-5:]}**")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Health Score", f"{int(top_issue['health_score'])}%", delta="-CRITICAL")
            with col2:
                st.write("**AI Analysis:** Machine is showing signatures of imminent failure.")
                if st.button("Generate Autonomous Repair Plan", type="primary"):
                    with st.spinner("Agent is consulting technical manuals & market data..."):
                        # Query the agent specifically about this unit's ID
                        prompt = f"Provide a repair plan and part procurement strategy for Unit {top_issue['Product ID']}"
                        response = st.session_state.agent.executor.invoke({"input": prompt})
                        st.success("Plan Generated Successfully")
                        st.markdown(clean_response(response["output"]))
        
        # Collapsible for the rest of the fleet
        with st.expander("🔍 View Fleet Status Overview"):
            for issue in results:
                if issue != top_issue:
                    st.write(f"✅ **Unit {issue['Product ID'][-5:]}**: {int(issue['health_score'])}% Health (Stable)")
    else:
        st.info("Gathering initial telemetry data...")

# Render the Audit at the top of the app
autonomous_daily_audit()

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
    if st.button("🚀 Run Full Executive Audit", type="secondary"):
        with st.spinner("Synthesizing Multi-Source Intelligence..."):
            report = st.session_state.agent.run_analysis()
            st.success("Audit Complete")
            st.info(report)

with tab_predictive:
    st.markdown("### Proactive Health Monitoring")
    # Pulling from the results we initialized at the top
    risk_df = pd.DataFrame(st.session_state.telemetry_results)
    
    cols = st.columns(5) # Show 5 metrics per row
    for i, (index, row) in enumerate(risk_df.iterrows()):
        score = int(row['health_score'])
        with cols[i % 5]:
            st.metric(
                label=f"Unit {row['Product ID'][-5:]}", 
                value=f"{score}%", 
                delta=f"{score-100}%" if score < 100 else "Stable"
            )

with tab_chat:
    st.markdown("### Consult System Engineer")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about maintenance, manuals, or parts..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Orchestrating Agents..."):
                response = st.session_state.agent.executor.invoke({"input": prompt})
                output = clean_response(response["output"])
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})