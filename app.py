# app.py
import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title="Flow Regime Visual Twin",
    page_icon="🌊",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("🌟 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Classify Flow Regime", "How to use ", "Example 3"]
)

# --- Home Page ---
if page == "Home":
    st.title("Multiphase Flow Regime Dashboard")
    st.markdown("""
    This dashboard provides insights into multiphase flow regimes and their impact on engineering systems.
    """)
    
    # --- Section 1: About Multiphase Flow Regime ---
    st.subheader("1️⃣ About Multiphase Flow Regime")
    st.markdown("""
    Multiphase flow refers to the simultaneous flow of materials with different phases (gas, liquid, and/or solid) 
    within pipelines or process systems. Understanding the flow regime (such as bubbly, slug, annular, or dispersed flows) is critical because it affects pressure drop, heat transfer, and mass transport efficiency. Flow regimes are typically visualized using velocity, volume fraction, or imaging techniques.
    """)

    # --- Section 2: Impact of Flow Regimes in Multiphase Systems ---
    st.subheader("2️⃣ Impact of Flow Regimes in Multiphase Systems")
    st.markdown("""
    Different flow regimes have a significant impact on system performance and safety:
    - **Pressure Drop:** Certain regimes (like slug flow) can cause large fluctuations in pressure.
    - **Separation Efficiency:** Flow regime affects the performance of separators and separators.
    - **Equipment Design:** Correct prediction of flow regime is essential for pumps, pipelines, and reactors.
    - **Operational Safety:** Unstable flow regimes can lead to erosion, vibration, and operational hazards.
    """)

    # --- Image ---
    st.image("assets/flow-regime.png", caption="Common Multiphase Flow Regimes", width=400)

# --- Example Tabs ---
elif page == "Example 1":
    st.title("Example 1: Analytics")
    st.markdown("This section can contain charts, metrics, or tables.")
    
elif page == "Example 2":
    st.title("Example 2: Model Prediction")
    st.markdown("This section can have input forms, sliders, or model results.")

elif page == "Example 3":
    st.title("Example 3: Data Exploration")
    st.markdown("This section can have interactive data visualizations.")
