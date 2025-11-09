import streamlit as st

# Set page configuration
st.set_page_config(page_title="Flow Regime App", layout="wide")

# Sidebar with tabs
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["About Flow Regime", "Classify", "Visualize the Flow Regime"])

# --- About Flow Regime Tab ---
if tab == "About Flow Regime":
    st.title("About Flow Regime")
    st.markdown("""
    Flow regimes describe the distribution of phases (gas, liquid) in multiphase flow systems.
    They are critical for designing pipelines and ensuring safe and efficient operation.
    
    Common flow regimes include:
    - **Dispersed Bubble Flow**
    - **Slug Flow**
    - **Stratified Flow**
    - **Annular Flow**
    """)
    
# --- Classify Tab ---
elif tab == "Classify":
    st.title("Classify Flow Regime")
    st.markdown("Upload your sensor data or video to classify the flow regime.")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "mp4"])
    
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        # Placeholder for classification result
        st.info("Classification result will appear here.")

# --- Visualize Flow Regime Tab ---
elif tab == "Visualize the Flow Regime":
    st.title("Visualize Flow Regime")
    st.markdown("Visualize the flow regime from sensor data or video.")
    
    uploaded_file_viz = st.file_uploader("Upload file for visualization", type=["csv", "mp4"], key="viz")
    
    if uploaded_file_viz is not None:
        st.success(f"File {uploaded_file_viz.name} uploaded successfully!")
        # Placeholder for visualization
        st.info("Visualization will appear here.")
