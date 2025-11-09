"""
Visualization utilities for Streamlit app
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np


def plot_pressure_signal(pressure_data, max_points=1000):
    """Plot pressure signal"""
    # Downsample if too many points
    if len(pressure_data) > max_points:
        indices = np.linspace(0, len(pressure_data)-1, max_points, dtype=int)
        pressure_data = pressure_data[indices]
        time = indices * 0.05  # Assuming 20 Hz sampling
    else:
        time = np.arange(len(pressure_data)) * 0.05
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=pressure_data,
        mode='lines',
        name='Pressure',
        line=dict(color='#3498db', width=2)
    ))
    
    fig.update_layout(
        title="Pressure Signal Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Pressure (bar)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_classification_results(results):
    """Plot classification probabilities"""
    class_names = ["Dispersed Flow", "Plug Flow", "Slug Flow"]
    probabilities = results['class_probabilities']
    predicted_class = results['predicted_class']
    
    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(3)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=class_names,
        y=probabilities,
        marker_color=colors,
        text=[f'{p:.1%}' for p in probabilities],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Flow Regime Probabilities",
        xaxis_title="Flow Regime",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    return fig


def plot_velocity_predictions(results):
    """Plot velocity predictions"""
    velocities = ['Vsg', 'Vsl']
    predicted = [results['predicted_vsg'], results['predicted_vsl']]
    
    # Create subplot with predicted and true values if available
    if 'vsg_true' in results and results['vsg_true'] is not None:
        true_values = [results['vsg_true'], results['vsl_true']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Predicted',
            x=velocities,
            y=predicted,
            marker_color='#3498db',
            text=[f'{v:.3f}' for v in predicted],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='True',
            x=velocities,
            y=true_values,
            marker_color='#e74c3c',
            text=[f'{v:.3f}' for v in true_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Velocity Predictions vs True Values",
            xaxis_title="Velocity Component",
            yaxis_title="Velocity (m/s)",
            barmode='group',
            height=400
        )
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=velocities,
            y=predicted,
            marker_color=['#e74c3c', '#9b59b6'],
            text=[f'{v:.3f}' for v in predicted],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Predicted Superficial Velocities",
            xaxis_title="Velocity Component",
            yaxis_title="Velocity (m/s)",
            showlegend=False,
            height=400
        )
    
    return fig


def plot_retrieved_videos(retrieved_videos):
    """Plot similarity scores of retrieved videos"""
    fig = go.Figure()
    
    video_labels = [f"Video {i+1}" for i in range(len(retrieved_videos))]
    scores = retrieved_videos['similarity_score'].values
    
    fig.add_trace(go.Bar(
        x=scores,
        y=video_labels,
        orientation='h',
        marker_color='#1abc9c',
        text=[f'{s:.4f}' for s in scores],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Similarity: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Video Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis_title="",
        height=max(300, len(retrieved_videos) * 50),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def display_video_info(video, index):
    """Display video information card"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        **Rank:** #{index}  
        **Similarity:** {video['similarity_score']:.4f}
        """)
    
    with col2:
        st.markdown(f"""
        **Velocities:**  
        Vsg = {video['vsg']:.3f} m/s  
        Vsl = {video['vsl']:.3f} m/s  
        
        **File:** `{video['filename']}`
        """)
    
    # Display video if path exists
    if 'video_path' in video and os.path.exists(video['video_path']):
        try:
            st.video(video['video_path'])
        except:
            st.info("Video preview not available")


def create_batch_summary_plot(batch_results):
    """Create summary visualization for batch results"""
    df = pd.DataFrame(batch_results)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Flow Regime Distribution', 'Average Confidence by Regime'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Flow regime counts
    regime_counts = df['predicted_class'].value_counts()
    fig.add_trace(
        go.Bar(x=regime_counts.index, y=regime_counts.values, 
               marker_color='#3498db', name='Count'),
        row=1, col=1
    )
    
    # Average confidence by regime
    avg_confidence = df.groupby('predicted_class')['confidence'].mean()
    fig.add_trace(
        go.Bar(x=avg_confidence.index, y=avg_confidence.values,
               marker_color='#2ecc71', name='Avg Confidence'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig