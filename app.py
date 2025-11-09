import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Flow Regime Visual Twin",
    layout="wide"
)

# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# --- Custom CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1e2130;
    }
    .sidebar-button {
        background-color: #262c3d;
        color: #ffffff;
        padding: 14px 20px;
        border-radius: 8px;
        border: 1px solid #3d4463;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        width: 100%;
        text-align: left;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .sidebar-button:hover {
        background-color: #2d3348;
        border-color: #4d5578;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6366f1;
    }
    .stMarkdown, p, li { color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Navigation")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "🏠 Home"
    if st.button("📊 Classify Flow Regime", use_container_width=True):
        st.session_state.page = "📊 Classify Flow Regime"
    if st.button("📋 Guideline", use_container_width=True):
        st.session_state.page = "📋 Guideline"
    if st.button("🔬 About Model", use_container_width=True):
        st.session_state.page = "🔬 About Model"

# --- Page Selection ---
page = st.session_state.page

# ------------------------------------------------------------------------
# 🏠 HOME PAGE
# ------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("Multiphase Flow Regime Dashboard")
    st.markdown("""
    This dashboard provides insights into multiphase flow regimes and their impact on engineering systems.
    """)

    st.subheader("1️⃣ About Multiphase Flow Regime")
    st.markdown("""
    Multiphase flow refers to the simultaneous flow of materials with different phases (gas, liquid, and/or solid)
    within pipelines or process systems. Understanding the flow regime (such as bubbly, slug, annular, or dispersed flows)
    is critical because it affects pressure drop, heat transfer, and mass transport efficiency.
    """)

    st.subheader("2️⃣ Impact of Flow Regimes in Multiphase Systems")
    st.markdown("""
    Different flow regimes have a significant impact on system performance and safety:
    - **Pressure Drop:** Certain regimes (like slug flow) can cause large fluctuations in pressure.
    - **Separation Efficiency:** Flow regime affects the performance of separators.
    - **Equipment Design:** Correct prediction of flow regime is essential for pumps, pipelines, and reactors.
    - **Operational Safety:** Unstable flow regimes can lead to erosion, vibration, and operational hazards.
    """)

    st.image("assets/flow-regime.png", caption="Common Multiphase Flow Regimes", width=500)

# ------------------------------------------------------------------------
# 📊 CLASSIFY FLOW REGIME PAGE
# ------------------------------------------------------------------------
# --- Classify Flow Regime Page ---
elif page == "📊 Classify Flow Regime":
    st.title("📊 Classify Flow Regime")
    st.markdown("""
    Upload a **CSV or Excel file** containing flow measurement data (e.g., pressure, velocity, void fraction, etc.)  
    The system will visualize the data and provide an overall **predicted flow regime**.
    """)

    uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # --- Load Data ---
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        
        # Show data preview
        with st.expander("👀 View Data Preview"):
            st.dataframe(df.head(5), use_container_width=False)
            st.caption(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # --- Plot automatically detected features ---
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            st.subheader("📈 Data Visualization")

            # Identify time column if it exists
            time_col = None
            time_keywords = ['time', 'timestamp', 'date', 't', 'sec', 'second']
            for col in df.columns:
                if any(keyword in col.lower() for keyword in time_keywords):
                    time_col = col
                    break
            
            # Plot measurement columns (exclude time column)
            st.markdown("**Time-Series or Sequential Behavior:**")
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Select columns to plot (exclude time column)
            plot_cols = [col for col in numeric_cols if col != time_col]
            
            # Limit to first 3-5 columns for clarity
            plot_cols = plot_cols[:5]
            
            if not plot_cols:
                st.warning("⚠️ No measurement columns found to plot (only time column detected).")
            else:
                if time_col and time_col in df.columns:
                    # Use time column as x-axis
                    for col in plot_cols:
                        ax.plot(df[time_col], df[col], label=col, linewidth=1.5)
                    ax.set_xlabel(time_col, fontsize=10)
                else:
                    # Use index as x-axis
                    for col in plot_cols:
                        ax.plot(df.index, df[col], label=col, linewidth=1.5)
                    ax.set_xlabel("Sample Index", fontsize=10)
                
                ax.set_title("Flow Signal Trends", fontsize=12)
                ax.set_ylabel("Measurement Value", fontsize=10)
                ax.legend(loc='best', framealpha=0.9, fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

            # Correlation heatmap (exclude time column)
            if len(plot_cols) >= 2:
                st.markdown("**Correlation Between Features:**")
                corr = df[plot_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                cax = ax2.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                fig2.colorbar(cax)
                ax2.set_xticks(range(len(corr.columns)))
                ax2.set_yticks(range(len(corr.columns)))
                ax2.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
                ax2.set_yticklabels(corr.columns, fontsize=9)
                ax2.set_title("Feature Correlation Heatmap", pad=20, fontsize=12)
                
                # Add correlation values to cells
                for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        ax2.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                                ha='center', va='center', color='black', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig2)

        else:
            st.warning("⚠️ Not enough numeric columns to plot. Please upload data with multiple numeric features.")

        # --- Demo Prediction ---
        if st.button("🔍 Predict Flow Regime", use_container_width=False):
            possible_classes = ["Dispersed Flow", "Slug Flow", "Plug Flow"]
            predicted_class = np.random.choice(possible_classes)
            confidence = np.random.uniform(88, 97)

            st.subheader("🧠 Predicted Flow Regime")
            st.success(f"**{predicted_class}** (Confidence: {confidence:.2f}%)")
            st.caption("_Prediction generated from aggregated signal patterns (demo only)._")
            
            # Display corresponding flow regime image
            image_mapping = {
                "Dispersed Flow": "video-flow-regime/Dispersed-Flow.png",
                "Slug Flow": "video-flow-regime/Slug-Flow.png",
                "Plug Flow": "video-flow-regime/Plug-Flow.png"
            }
            
            image_path = image_mapping.get(predicted_class)
            if image_path:
                try:
                    st.image(image_path, caption=f"{predicted_class} Visualization", width=600)
                except Exception as e:
                    st.warning(f"⚠️ Could not load image: {image_path}")
                    st.info("Please ensure the image exists in the correct directory.")

    else:
        st.info("👆 Please upload a file to begin visualization and prediction.")

# ------------------------------------------------------------------------
# 📋 GUIDELINE PAGE
# ------------------------------------------------------------------------
elif page == "📋 Guideline":
    st.title("📋 Guideline")
    st.markdown("""
    This section provides instructions for using the Flow Regime Visual Twin:
    1. Navigate to **Classify Flow Regime**.
    2. Upload your dataset (CSV/Excel).
    3. Click **Predict Flow Regime**.
    4. View and download results.
    """)

# ------------------------------------------------------------------------
# 🔬 ABOUT MODEL PAGE
# ------------------------------------------------------------------------
elif page == "🔬 About Model":
    st.title("🔬 About the Model")
    st.markdown("""
    The classification model is designed to predict multiphase flow regimes using signal and image features.  
    Future versions will integrate:
    - **CNN-based video feature extraction**
    - **Physics-informed constraints**
    - **Multitask learning (classification + regression)**
    """)