"""
Streamlit Application for Flow Regime Classification and Video Retrieval
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from model_utils import FlowRegimePredictor
from visualization_utils import (
    plot_pressure_signal,
    plot_classification_results,
    plot_velocity_predictions,
    plot_retrieved_videos,
    display_video_info
)

# Page configuration
st.set_page_config(
    page_title="Flow Regime Classifier",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/water.png", width=80)
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    # Model selection
    st.subheader("📁 Model Configuration")
    model_path = st.text_input(
        "Model Path",
        value="models/best_multitask_pinn_fold_1.pth",
        help="Path to the trained model checkpoint"
    )
    
    scalers_path = st.text_input(
        "Scalers Path",
        value="models/best_model_scalers.pkl",
        help="Path to the saved scalers file"
    )
    
    # Load model button
    if st.button("🚀 Load Model", use_container_width=True):
        with st.spinner("Loading model and scalers..."):
            try:
                st.session_state.predictor = FlowRegimePredictor(
                    model_path=model_path,
                    scalers_path=scalers_path
                )
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    st.markdown("---")
    
    # Video retrieval settings
    st.subheader("🎬 Video Retrieval")
    video_library_path = st.text_input(
        "Video Library CSV",
        value="video_library.csv",
        help="Path to video library CSV file"
    )
    
    top_k = st.slider(
        "Number of videos to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Top K similar videos to retrieve"
    )
    
    similarity_metric = st.selectbox(
        "Similarity Metric",
        options=["cosine", "euclidean"],
        index=0,
        help="Metric for computing similarity"
    )
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Flow Regime Classifier v1.0**
        
        A multi-task deep learning system for:
        - Flow regime classification
        - Velocity regression (Vsg, Vsl)
        - Video retrieval
        
        Built with Physics-Informed Neural Networks (PINNs)
        """)

# Main content
st.markdown('<p class="main-header">🌊 Multiphase Flow Regime Classification System</p>', 
            unsafe_allow_html=True)

# Check if model is loaded
if st.session_state.predictor is None:
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.warning("⚠️ Please load the model from the sidebar to start!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display instructions
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📋 Quick Start Guide
    
    1. **Load Model**: Enter model and scalers paths in the sidebar and click "Load Model"
    2. **Upload Data**: Upload your pressure data file (Excel format)
    3. **Analyze**: View predictions, velocity estimates, and retrieved similar videos
    4. **Export**: Download results and visualizations
    
    ### 📁 Supported File Formats
    - Excel files (.xlsx) with pressure measurements
    - Filename should contain velocities: `FlowType-Vsg=X.XX-Vsl=Y.YY.xlsx`
    - Required column: `Pressure (barA)` or `Pressure/bar`
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sample data format
    st.subheader("📊 Sample Data Format")
    sample_data = pd.DataFrame({
        'Time (s)': [0.0, 0.05, 0.10, 0.15, 0.20],
        'Pressure (barA)': [1.013, 1.015, 1.012, 1.014, 1.013],
    })
    st.dataframe(sample_data, use_container_width=True)
    
    st.stop()

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single File Analysis", 
    "📁 Batch Processing", 
    "🎬 Video Library", 
    "📈 Model Info"
])

# TAB 1: Single File Analysis
with tab1:
    st.markdown('<p class="sub-header">Single File Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Pressure Data File",
            type=['xlsx', 'xls'],
            help="Upload Excel file containing pressure measurements",
            key="single_file"
        )
    
    with col2:
        # Manual velocity input (optional)
        st.subheader("Manual Velocity Input")
        use_manual = st.checkbox("Enter velocities manually", value=False)
        
        if use_manual:
            manual_vsg = st.number_input("Vsg (m/s)", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
            manual_vsl = st.number_input("Vsl (m/s)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
        else:
            manual_vsg = None
            manual_vsl = None
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and display data
        try:
            df = pd.read_excel(temp_file_path)
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Analyze button
            if st.button("🔍 Analyze", use_container_width=True, type="primary"):
                with st.spinner("🔮 Running inference..."):
                    try:
                        # Run prediction
                        results = st.session_state.predictor.predict_from_file(
                            str(temp_file_path),
                            manual_vsg=manual_vsg,
                            manual_vsl=manual_vsl
                        )
                        
                        st.session_state.prediction_results = results
                        st.session_state.uploaded_data = df
                        
                        st.success("✅ Analysis completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display results
            if st.session_state.prediction_results is not None:
                results = st.session_state.prediction_results
                
                st.markdown("---")
                st.markdown('<p class="sub-header">📈 Prediction Results</p>', unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Predicted Flow Regime",
                        value=results['predicted_class_name'],
                        help="Predicted flow pattern"
                    )
                
                with col2:
                    confidence = results['class_probabilities'][results['predicted_class']]
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1%}",
                        help="Classification confidence"
                    )
                
                with col3:
                    st.metric(
                        label="Predicted Vsg",
                        value=f"{results['predicted_vsg']:.3f} m/s",
                        delta=f"{results['predicted_vsg'] - results.get('vsg_true', results['predicted_vsg']):.3f}" if 'vsg_true' in results else None,
                        help="Superficial gas velocity"
                    )
                
                with col4:
                    st.metric(
                        label="Predicted Vsl",
                        value=f"{results['predicted_vsl']:.3f} m/s",
                        delta=f"{results['predicted_vsl'] - results.get('vsl_true', results['predicted_vsl']):.3f}" if 'vsl_true' in results else None,
                        help="Superficial liquid velocity"
                    )
                
                # Visualizations
                st.markdown("---")
                
                # Row 1: Pressure signal and classification
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Pressure Signal")
                    pressure_col = None
                    for col in ['Pressure/bar', 'Pressure (barA)', 'Pressure']:
                        if col in st.session_state.uploaded_data.columns:
                            pressure_col = col
                            break
                    
                    if pressure_col:
                        fig = plot_pressure_signal(st.session_state.uploaded_data[pressure_col].values)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Pressure column not found in data")
                
                with col2:
                    st.subheader("🎯 Classification Probabilities")
                    fig = plot_classification_results(results)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2: Velocity predictions
                st.markdown("---")
                st.subheader("🚀 Velocity Predictions")
                fig = plot_velocity_predictions(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Video retrieval (if library available)
                if os.path.exists(video_library_path):
                    st.markdown("---")
                    st.markdown('<p class="sub-header">🎬 Similar Videos</p>', unsafe_allow_html=True)
                    
                    if st.button("🔍 Retrieve Similar Videos", use_container_width=True):
                        with st.spinner("Searching video library..."):
                            try:
                                retrieved_videos = st.session_state.predictor.retrieve_videos(
                                    results['embedding'],
                                    results['predicted_class'],
                                    video_library_path,
                                    top_k=top_k,
                                    similarity_metric=similarity_metric
                                )
                                
                                if len(retrieved_videos) > 0:
                                    st.success(f"✅ Found {len(retrieved_videos)} similar videos")
                                    
                                    # Display videos
                                    for idx, (_, video) in enumerate(retrieved_videos.iterrows(), 1):
                                        with st.expander(f"📹 Video {idx}: {video['filename']}", expanded=(idx==1)):
                                            display_video_info(video, idx)
                                    
                                    # Plot similarity scores
                                    fig = plot_retrieved_videos(retrieved_videos)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No similar videos found")
                                    
                            except Exception as e:
                                st.error(f"Error retrieving videos: {str(e)}")
                
                # Export results
                st.markdown("---")
                st.subheader("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export as JSON
                    import json
                    results_dict = {
                        'filename': uploaded_file.name,
                        'predicted_class': results['predicted_class_name'],
                        'confidence': float(results['class_probabilities'][results['predicted_class']]),
                        'predicted_vsg': float(results['predicted_vsg']),
                        'predicted_vsl': float(results['predicted_vsl']),
                        'class_probabilities': {
                            'Dispersed Flow': float(results['class_probabilities'][0]),
                            'Plug Flow': float(results['class_probabilities'][1]),
                            'Slug Flow': float(results['class_probabilities'][2])
                        }
                    }
                    
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(results_dict, indent=2),
                        file_name=f"results_{uploaded_file.name.replace('.xlsx', '.json')}",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    # Export as CSV
                    results_df = pd.DataFrame([results_dict])
                    csv = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"results_{uploaded_file.name.replace('.xlsx', '.csv')}",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# TAB 2: Batch Processing
with tab2:
    st.markdown('<p class="sub-header">Batch File Processing</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("📁 Upload multiple files to process them in batch mode")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Multiple Pressure Data Files",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        key="batch_files"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} files uploaded")
        
        if st.button("🚀 Process All Files", use_container_width=True, type="primary"):
            # Create temp directory
            temp_dir = Path("temp_batch")
            temp_dir.mkdir(exist_ok=True)
            
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Save file temporarily
                temp_file_path = temp_dir / uploaded_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Run prediction
                    results = st.session_state.predictor.predict_from_file(str(temp_file_path))
                    
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'predicted_class': results['predicted_class_name'],
                        'confidence': results['class_probabilities'][results['predicted_class']],
                        'predicted_vsg': results['predicted_vsg'],
                        'predicted_vsl': results['predicted_vsl'],
                        'vsg_true': results.get('vsg_true', None),
                        'vsl_true': results.get('vsl_true', None)
                    })
                
                except Exception as e:
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'predicted_class': 'ERROR',
                        'confidence': 0,
                        'predicted_vsg': 0,
                        'predicted_vsl': 0,
                        'error': str(e)
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing completed!")
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Batch Results")
            
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Files", len(uploaded_files))
            
            with col2:
                success_count = len([r for r in batch_results if r['predicted_class'] != 'ERROR'])
                st.metric("Successful", success_count)
            
            with col3:
                error_count = len([r for r in batch_results if r['predicted_class'] == 'ERROR'])
                st.metric("Errors", error_count)
            
            # Distribution plot
            if success_count > 0:
                st.markdown("---")
                st.subheader("📈 Flow Regime Distribution")
                
                class_counts = results_df[results_df['predicted_class'] != 'ERROR']['predicted_class'].value_counts()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=class_counts.index,
                        y=class_counts.values,
                        marker_color=['#3498db', '#e74c3c', '#2ecc71'],
                        text=class_counts.values,
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Distribution of Predicted Flow Regimes",
                    xaxis_title="Flow Regime",
                    yaxis_title="Count",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export batch results
            st.markdown("---")
            st.subheader("💾 Export Batch Results")
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Batch Results (CSV)",
                data=csv,
                file_name="batch_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# TAB 3: Video Library
with tab3:
    st.markdown('<p class="sub-header">Video Library Management</p>', unsafe_allow_html=True)
    
    if os.path.exists(video_library_path):
        try:
            video_df = pd.read_csv(video_library_path)
            
            st.success(f"✅ Video library loaded: {len(video_df)} videos")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Videos", len(video_df))
            
            with col2:
                st.metric("Flow Regimes", video_df['flow_regime'].nunique())
            
            with col3:
                avg_vsg = video_df['vsg'].mean()
                st.metric("Avg Vsg", f"{avg_vsg:.3f} m/s")
            
            # Flow regime distribution
            st.markdown("---")
            st.subheader("📊 Library Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Flow regime distribution
                regime_counts = video_df['flow_regime'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=regime_counts.index,
                        values=regime_counts.values,
                        hole=0.4,
                        marker_colors=['#3498db', '#e74c3c', '#2ecc71']
                    )
                ])
                
                fig.update_layout(
                    title="Flow Regime Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Velocity scatter plot
                fig = go.Figure()
                
                for regime in video_df['flow_regime'].unique():
                    regime_data = video_df[video_df['flow_regime'] == regime]
                    
                    fig.add_trace(go.Scatter(
                        x=regime_data['vsg'],
                        y=regime_data['vsl'],
                        mode='markers',
                        name=regime,
                        marker=dict(size=10, opacity=0.7)
                    ))
                
                fig.update_layout(
                    title="Velocity Space Coverage",
                    xaxis_title="Vsg (m/s)",
                    yaxis_title="Vsl (m/s)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Video table
            st.markdown("---")
            st.subheader("📋 Video Library")
            
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                selected_regime = st.multiselect(
                    "Filter by Flow Regime",
                    options=video_df['flow_regime'].unique(),
                    default=video_df['flow_regime'].unique()
                )
            
            with col2:
                vsg_range = st.slider(
                    "Vsg Range (m/s)",
                    min_value=float(video_df['vsg'].min()),
                    max_value=float(video_df['vsg'].max()),
                    value=(float(video_df['vsg'].min()), float(video_df['vsg'].max()))
                )
            
            # Apply filters
            filtered_df = video_df[
                (video_df['flow_regime'].isin(selected_regime)) &
                (video_df['vsg'] >= vsg_range[0]) &
                (video_df['vsg'] <= vsg_range[1])
            ]
            
            st.dataframe(filtered_df, use_container_width=True)
            st.info(f"Showing {len(filtered_df)} of {len(video_df)} videos")
            
        except Exception as e:
            st.error(f"Error loading video library: {str(e)}")
    else:
        st.warning(f"⚠️ Video library not found at: {video_library_path}")
        st.info("Please upload a video library CSV file or update the path in the sidebar")

# TAB 4: Model Info
with tab4:
    st.markdown('<p class="sub-header">Model Information</p>', unsafe_allow_html=True)
    
    if st.session_state.predictor is not None:
        predictor = st.session_state.predictor
        
        # Model architecture
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏗️ Model Architecture")
            
            st.markdown("""
            **Multi-Task Physics-Informed Neural Network (PINN)**
            
            The model consists of:
            - **Shared CNN Backbone**: Processes pressure signal windows
            - **Feature Processing Branch**: Statistical and frequency features
            - **Velocity Processing Branch**: Superficial velocities
            - **Shared Representation Layer**: 128 hidden units
            
            **Task-Specific Heads:**
            1. Classification Head (3 classes)
            2. Velocity Regression Head (Vsg, Vsl)
            3. Physics-Based Pressure Prediction
            """)
        
        with col2:
            st.subheader("📊 Model Metrics")
            
            # Display model info if available
            if hasattr(predictor, 'model_info'):
                st.metric("Validation Accuracy", f"{predictor.model_info.get('val_acc', 'N/A')}")
                st.metric("Velocity MAE", f"{predictor.model_info.get('val_velocity_mae', 'N/A'):.4f}")
            else:
                st.info("Model metrics not available")
        
        # Model parameters
        st.markdown("---")
        st.subheader("⚙️ Hyperparameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Signal Processing:**
            - Window Size: 40
            - Stride: 20
            - Hidden Size: 128
            """)
        
        with col2:
            st.markdown("""
            **Training:**
            - Batch Size: 16
            - Epochs: 200
            - Learning Rate: 0.0005
            """)
        
        with col3:
            st.markdown("""
            **Loss Weights:**
            - Classification: 0.7
            - Velocity: 0.7
            - Physics: 0.3
            """)
        
        # Scaler information
        st.markdown("---")
        st.subheader("📐 Data Scalers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Feature Scalers:**
            - Pressure: StandardScaler
            - Features: StandardScaler
            """)
        
        with col2:
            st.markdown("""
            **Velocity Scalers:**
            - Vsg: RobustScaler
            - Vsl: RobustScaler
            """)
        
        # Flow regimes
        st.markdown("---")
        st.subheader("🌊 Flow Regimes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Dispersed Flow (Class 0)**
            
            Gas bubbles dispersed in continuous liquid phase
            
            Characteristics:
            - Small gas bubbles
            - Homogeneous distribution
            - Low Vsg, high Vsl
            """)
        
        with col2:
            st.markdown("""
            **Plug Flow (Class 1)**
            
            Alternating plugs of gas and liquid
            
            Characteristics:
            - Bullet-shaped bubbles
            - Periodic pattern
            - Medium Vsg, medium Vsl
            """)
        
        with col3:
            st.markdown("""
            **Slug Flow (Class 2)**
            
            Large gas pockets with liquid slugs
            
            Characteristics:
            - Taylor bubbles
            - Liquid slugs between
            - High Vsg, variable Vsl
            """)
        
        # System info
        st.markdown("---")
        st.subheader("💻 System Information")
        
        import torch
        import platform
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Environment:**
            - Python: {platform.python_version()}
            - PyTorch: {torch.__version__}
            - Device: {predictor.device}
            """)
        
        with col2:
            st.markdown(f"""
            **Platform:**
            - OS: {platform.system()}
            - Processor: {platform.processor()}
            - CUDA Available: {torch.cuda.is_available()}
            """)
    
    else:
        st.warning("⚠️ Please load the model first to view detailed information")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p>Flow Regime Classification System v1.0</p>
    <p>Built with ❤️ using Streamlit and PyTorch</p>
</div>
""", unsafe_allow_html=True)