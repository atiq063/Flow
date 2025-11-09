import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Flow Regime Visualizer", layout="wide")

# App title
st.title("Flow Regime Visualizer 🌊")

# Tabs
tab1, tab2, tab3 = st.tabs(["How to Use", "Classify Flow Regime", "Visualization"])

# -------------------------------
# Tab 1: How to Use
# -------------------------------
with tab1:
    st.header("How to Use This App")
    st.markdown("""
    1. Go to **Classify Flow Regime** tab to upload your dataset and classify the flow regime.
    2. Go to **Visualization** tab to explore visual insights of the flow data.
    3. Follow on-screen instructions and explore your data.
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Water_flowing_in_a_river.jpg/320px-Water_flowing_in_a_river.jpg",
        caption="Flow Visualization Example"
    )

# -------------------------------
# Tab 2: Classify Flow Regime
# -------------------------------
with tab2:
    st.header("Classify Flow Regime")
    
    uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
        
        if st.button("Classify Flow"):
            # Placeholder for ML model
            st.success("Flow regime classified as: **Slug Flow**")  # Example output