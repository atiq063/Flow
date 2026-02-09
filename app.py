import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Flow Regime Visual Twin",
    layout="wide"
)

# Model configuration
WINDOW_SIZE = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ["Dispersed Flow", "Plug Flow", "Slug Flow"]

# --- Model Definition ---
# --- Model Definition ---
class MultiTaskPINN(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=3):
        super(MultiTaskPINN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.4)
        
        self.fc_features = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fc_velocities = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        conv_output_size = 64 * (WINDOW_SIZE // 2)
        combined_size = conv_output_size + 64 + 32
        
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.velocity_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        self.physics_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, pressure_window, features, velocities):
        x = pressure_window.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        
        feat = self.fc_features(features)
        vel = self.fc_velocities(velocities)
        
        combined = torch.cat([x, feat, vel], dim=1)
        shared_repr = self.shared_layer(combined)
        
        class_output = self.classifier(shared_repr)
        velocity_output = self.velocity_regressor(shared_repr)
        physics_output = self.physics_net(shared_repr)
        
        return class_output, velocity_output, physics_output
# --- Helper Functions ---
@st.cache_resource
def load_model_and_scalers():
    """Load the trained model and scalers"""
    try:
        # Load model
        model_path = "models/best_multitask_pinn_fold_5.pth"
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        model = MultiTaskPINN(input_size=WINDOW_SIZE, hidden_size=128, num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        # Try to load scalers from checkpoint first
        scalers = checkpoint.get('scalers', None)
        
        # If not in checkpoint, try loading from pickle file
        if scalers is None:
            try:
                scalers_path = "training_scalers.pkl"
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
            except FileNotFoundError:
                st.error("❌ Scalers file not found at: training_scalers.pkl")
                return model, None
            except Exception as e:
                st.error(f"❌ Error loading scalers: {str(e)}")
                return model, None
        
        # Validate that all required scalers are present
        required_scalers = ['scaler_pressure', 'scaler_features', 'scaler_vsg', 'scaler_vsl']
        missing_scalers = [s for s in required_scalers if s not in scalers]
        
        if missing_scalers:
            st.warning(f"⚠️ Missing scalers: {', '.join(missing_scalers)}")
            st.info(f"Available scalers: {list(scalers.keys())}")
            return model, None
        
        return model, scalers
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: models/best_multitask_pinn_fold_5.pth")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        return None, None

def extract_features(pressure_window):
    """Extract features from pressure window"""
    features = []
    features.append(np.mean(pressure_window))
    features.append(np.std(pressure_window))
    features.append(np.max(pressure_window) - np.min(pressure_window))
    
    gradient = np.gradient(pressure_window)
    features.append(np.mean(gradient))
    features.append(np.std(gradient))
    features.append(np.max(np.abs(gradient)))
    
    if len(pressure_window) > 4:
        freqs = fftfreq(len(pressure_window), d=0.05)
        fft_vals = np.abs(fft(pressure_window))
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]
        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            features.append(positive_freqs[dominant_freq_idx])
            features.append(positive_fft[dominant_freq_idx])
        else:
            features.append(0.0)
            features.append(0.0)
    else:
        features.append(0.0)
        features.append(0.0)
    
    return np.array(features)

def predict_flow_regime(model, scalers, pressure_data, vsg=0.5, vsl=0.5):
    """Make prediction using the trained model"""
    try:
        # Check if scalers are available
        if scalers is None:
            st.error("❌ Scalers not available. The model checkpoint may not contain scalers.")
            st.info("💡 Please retrain the model and save scalers in the checkpoint, or provide them separately.")
            return None
        
        stride = 20
        windows = []
        for i in range(0, len(pressure_data) - WINDOW_SIZE + 1, stride):
            window = pressure_data[i:i+WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                windows.append(window)
        
        if len(windows) == 0:
            st.warning(f"⚠️ Not enough data points. Need at least {WINDOW_SIZE} points, but got {len(pressure_data)}.")
            return None
        
        features_list = [extract_features(w) for w in windows]
        
        pressure_windows = np.array(windows)
        features_array = np.array(features_list)
        
        # Apply scaling with error handling
        try:
            pressure_scaled = scalers['scaler_pressure'].transform(pressure_windows)
            features_scaled = scalers['scaler_features'].transform(features_array)
            vsg_scaled = scalers['scaler_vsg'].transform([[vsg]] * len(windows))
            vsl_scaled = scalers['scaler_vsl'].transform([[vsl]] * len(windows))
            velocities_scaled = np.hstack([vsg_scaled, vsl_scaled])
        except KeyError as e:
            st.error(f"❌ Missing scaler: {e}")
            st.info(f"Available scalers: {list(scalers.keys())}")
            return None
        except Exception as e:
            st.error(f"❌ Error during scaling: {str(e)}")
            return None
        
        pressure_tensor = torch.FloatTensor(pressure_scaled).to(DEVICE)
        features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
        velocities_tensor = torch.FloatTensor(velocities_scaled).to(DEVICE)
        
        with torch.no_grad():
            class_output, velocity_output, _ = model(pressure_tensor, features_tensor, velocities_tensor)
            
            probabilities = torch.softmax(class_output, dim=1)
            avg_probabilities = probabilities.mean(dim=0).cpu().numpy()
            
            predicted_class = torch.argmax(probabilities.mean(dim=0)).item()
            confidence = avg_probabilities[predicted_class] * 100
            
            velocity_pred = velocity_output.mean(dim=0).cpu().numpy()
            vsg_pred = scalers['scaler_vsg'].inverse_transform([[velocity_pred[0]]])[0][0]
            vsl_pred = scalers['scaler_vsl'].inverse_transform([[velocity_pred[1]]])[0][0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': avg_probabilities,
            'vsg_predicted': vsg_pred,
            'vsl_predicted': vsl_pred,
            'num_windows': len(windows)
        }
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.exception(e)
        return None

def render_media_card(media_path, title, caption):
    """Render an image or GIF inside a styled card."""
    try:
        with open(media_path, "rb") as f:
            media_bytes = f.read()
        media_b64 = base64.b64encode(media_bytes).decode("ascii")
        extension = os.path.splitext(media_path)[1].lower()
        mime = "image/gif" if extension == ".gif" else "image/png"
        st.markdown(
            f"""
            <div class="media-card">
                <div class="media-title">{title}</div>
                <img src="data:{mime};base64,{media_b64}" alt="{caption}">
                <div class="media-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"⚠️ Unable to render visualization card: {str(e)}")

def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator

def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return "n/a"
    return f"{value:.6g}"

def build_output_table(rows, vsg_pred=None, vsl_pred=None, vsg_input=None, vsl_input=None):
    rows_formatted = [(name, format_value(value), units) for name, value, units in rows]
    midpoint = (len(rows_formatted) + 1) // 2
    left_rows = rows_formatted[:midpoint]
    right_rows = rows_formatted[midpoint:]

    html_rows = []
    max_len = max(len(left_rows), len(right_rows))
    for idx in range(max_len):
        left = left_rows[idx] if idx < len(left_rows) else ("", "", "")
        right = right_rows[idx] if idx < len(right_rows) else ("", "", "")
        html_rows.append(
            "<tr>"
            f"<td>{left[0]}</td><td>{left[1]}</td><td>{left[2]}</td>"
            f"<td>{right[0]}</td><td>{right[1]}</td><td>{right[2]}</td>"
            "</tr>"
        )

    metrics_html = ""
    if vsg_pred is not None and vsl_pred is not None:
        vsg_delta = None if vsg_input is None else vsg_pred - vsg_input
        vsl_delta = None if vsl_input is None else vsl_pred - vsl_input
        vsg_delta_text = "" if vsg_delta is None else f"Delta: {format_value(vsg_delta)} m/s"
        vsl_delta_text = "" if vsl_delta is None else f"Delta: {format_value(vsl_delta)} m/s"
        metrics_html = (
            '<div class="derived-metrics">'
            '<div class="derived-metric">'
            '<div class="metric-label">Vsg (Predicted)</div>'
            f'<div class="metric-value">{format_value(vsg_pred)} m/s</div>'
            f'<div class="metric-delta">{vsg_delta_text}</div>'
            '</div>'
            '<div class="derived-metric">'
            '<div class="metric-label">Vsl (Predicted)</div>'
            f'<div class="metric-value">{format_value(vsl_pred)} m/s</div>'
            f'<div class="metric-delta">{vsl_delta_text}</div>'
            '</div>'
            '</div>'
        )

    html = (
        '<div class="derived-card">'
        '<div class="derived-title">Derived Flow Calculations</div>'
        '<table class="derived-table">'
        '<thead>'
        '<tr>'
        '<th>Parameter</th><th>Value</th><th>Units</th>'
        '<th>Parameter</th><th>Value</th><th>Units</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        f"{''.join(html_rows)}"
        '</tbody>'
        '</table>'
        f"{metrics_html}"
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def compute_mass_flow_outputs(inputs):
    mdot_l = inputs["mdot_l"]
    mdot_g = inputs["mdot_g"]
    p_in = inputs["p_in"]
    t = inputs["t"]
    z = inputs["z"]
    rho_l = inputs["rho_l"]
    mu_l = inputs["mu_l"]
    mu_g = inputs["mu_g"]
    d = inputs["d"]
    r_g = inputs["r_g"]

    area = np.pi * d**2 / 4
    rho_g = safe_divide(p_in, z * r_g * t)
    q_l = safe_divide(mdot_l, rho_l)
    q_g = safe_divide(mdot_g, rho_g)
    q_t = q_l + q_g
    alpha_in = safe_divide(q_g, q_t)
    u_sl = safe_divide(q_l, area)
    u_sg = safe_divide(q_g, area)
    u_m = u_sl + u_sg
    rho_m = alpha_in * rho_g + (1 - alpha_in) * rho_l
    mu_m = alpha_in * mu_g + (1 - alpha_in) * mu_l
    re_m = safe_divide(rho_m * u_m * d, mu_m)

    outputs = [
        ("A", area, "m^2"),
        ("rho_G", rho_g, "kg/m^3"),
        ("q_L", q_l, "m^3/s"),
        ("q_G", q_g, "m^3/s"),
        ("q_T", q_t, "m^3/s"),
        ("alpha_in", alpha_in, "-"),
        ("U_sl", u_sl, "m/s"),
        ("U_sg", u_sg, "m/s"),
        ("U_m", u_m, "m/s"),
        ("rho_m", rho_m, "kg/m^3"),
        ("mu_m", mu_m, "Pa*s"),
        ("Re_m", re_m, "-"),
    ]

    return outputs, u_sg, u_sl

def compute_volume_flow_outputs(inputs):
    q_l = inputs["q_l"]
    q_g = inputs["q_g"]
    u_sl_input = inputs["u_sl"]
    u_sg_input = inputs["u_sg"]
    p_in = inputs["p_in"]
    t = inputs["t"]
    z = inputs["z"]
    rho_l = inputs["rho_l"]
    mu_l = inputs["mu_l"]
    mu_g = inputs["mu_g"]
    d = inputs["d"]
    r_g = inputs["r_g"]
    u_l = inputs.get("u_l")
    u_g = inputs.get("u_g")

    area = np.pi * d**2 / 4
    rho_g = safe_divide(p_in, z * r_g * t)
    u_sl = safe_divide(q_l, area)
    u_sg = safe_divide(q_g, area)
    mdot_l = rho_l * q_l
    mdot_g = rho_g * q_g
    q_t = q_l + q_g
    alpha_in_q = safe_divide(q_g, q_t)
    u_m = u_sl + u_sg

    a_l = None
    a_g = None
    alpha_in_area = None
    s = None
    if u_l and u_g:
        a_l = safe_divide(q_l, u_l)
        a_g = safe_divide(q_g, u_g)
        alpha_in_area = safe_divide(a_g, a_g + a_l)
        s = safe_divide(u_g, u_l)

    alpha_used = alpha_in_area if alpha_in_area is not None else alpha_in_q
    rho_m = alpha_used * rho_g + (1 - alpha_used) * rho_l
    mu_m = alpha_used * mu_g + (1 - alpha_used) * mu_l
    re_m = safe_divide(rho_m * u_m * d, mu_m)

    outputs = [
        ("A", area, "m^2"),
        ("rho_G", rho_g, "kg/m^3"),
        ("q_L", q_l, "m^3/s"),
        ("q_G", q_g, "m^3/s"),
        ("mdot_L", mdot_l, "kg/s"),
        ("mdot_G", mdot_g, "kg/s"),
        ("q_T", q_t, "m^3/s"),
        ("alpha_in (from q)", alpha_in_q, "-"),
        ("U_m", u_m, "m/s"),
    ]

    if a_l is not None and a_g is not None:
        outputs.extend([
            ("A_L", a_l, "m^2"),
            ("A_G", a_g, "m^2"),
            ("alpha_in (from areas)", alpha_in_area, "-"),
            ("S", s, "-"),
        ])

    outputs.extend([
        ("rho_m", rho_m, "kg/m^3"),
        ("mu_m", mu_m, "Pa*s"),
        ("Re_m", re_m, "-"),
    ])

    return outputs, u_sl_input, u_sg_input, u_sl, u_sg, u_sg, u_sl

# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# --- Custom CSS ---
st.markdown("""
<style>
    :root {
        --hbku-maroon: #8a1538;
        --hbku-maroon-dark: #6f0f2c;
        --hbku-gold: #c6a15b;
        --hbku-sand: #f7f3ee;
        --hbku-ink: #1f1f24;
        --hbku-slate: #4b5563;
        --hbku-mist: #eef0f3;
    }

    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(198, 161, 91, 0.18), transparent 60%),
            radial-gradient(900px 500px at 95% 10%, rgba(138, 21, 56, 0.12), transparent 55%),
            var(--hbku-sand);
        color: var(--hbku-ink);
    }

    html, body, [class*="css"] {
        font-family: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f3f4f7 100%);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--hbku-ink);
    }

    .stMarkdown, p, li {
        color: var(--hbku-slate);
        font-size: 1rem;
    }

    h1, h2, h3 {
        color: var(--hbku-ink);
        font-family: "Crimson Pro", "Times New Roman", serif;
        letter-spacing: 0.2px;
    }

    h1 {
        font-size: 2.4rem;
    }

    h2 {
        font-size: 1.8rem;
    }

    h3 {
        font-size: 1.4rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--hbku-maroon) 0%, var(--hbku-maroon-dark) 100%);
        color: #ffffff;
        padding: 0.7rem 1.1rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 8px 18px rgba(138, 21, 56, 0.22);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(138, 21, 56, 0.28);
    }

    .stButton > button:focus {
        outline: 2px solid rgba(138, 21, 56, 0.35);
        outline-offset: 2px;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, var(--hbku-maroon) 0%, var(--hbku-maroon-dark) 100%);
        color: #ffffff;
    }

    [data-testid="stSidebar"] .stButton > button p {
        color: #ffffff;
        font-weight: 600;
    }

    .stButton > button * {
        color: #ffffff !important;
    }

    .cta-card {
        background: #ffffff;
        border: 1px solid rgba(138, 21, 56, 0.18);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 12px 20px rgba(31, 41, 55, 0.08);
        margin: 16px 0 8px 0;
    }

    .cta-title {
        font-family: "Crimson Pro", "Times New Roman", serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--hbku-maroon);
        margin-bottom: 6px;
    }

    .cta-subtitle {
        color: var(--hbku-slate);
        font-size: 1rem;
    }

    .media-card {
        background: #ffffff;
        border: 1px solid rgba(138, 21, 56, 0.15);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 12px 20px rgba(31, 41, 55, 0.08);
        margin: 12px 0 20px 0;
        min-height: 520px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        text-align: center;
    }

    .media-card img {
        width: 100%;
        height: 380px;
        object-fit: contain;
        border-radius: 12px;
    }

    .media-title {
        font-family: "Crimson Pro", "Times New Roman", serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--hbku-maroon);
        margin-bottom: 10px;
    }

    .media-caption {
        color: var(--hbku-slate);
        font-size: 0.95rem;
        margin-top: 10px;
    }

    .derived-card {
        background: #ffffff;
        border: 1px solid rgba(138, 21, 56, 0.15);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 12px 20px rgba(31, 41, 55, 0.08);
        margin: 12px 0 20px 0;
        min-height: 520px;
        display: flex;
        flex-direction: column;
    }

    .derived-title {
        font-family: "Crimson Pro", "Times New Roman", serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--hbku-maroon);
        margin-bottom: 10px;
    }

    .derived-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
        margin-bottom: 4px;
    }

    .derived-table th {
        text-align: left;
        color: var(--hbku-ink);
        font-weight: 600;
        padding: 8px 6px;
        border-bottom: 1px solid rgba(15, 23, 42, 0.12);
    }

    .derived-table td {
        padding: 6px 6px;
        color: var(--hbku-slate);
        border-bottom: 1px solid rgba(15, 23, 42, 0.06);
        vertical-align: top;
    }

    .derived-table tr:last-child td {
        border-bottom: line;
    }

    .derived-metrics {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid rgba(15, 23, 42, 0.08);
    }

    .derived-metric {
        background: linear-gradient(135deg, rgba(138, 21, 56, 0.08), rgba(198, 161, 91, 0.08));
        border-radius: 12px;
        padding: 10px 12px;
    }

    .metric-label {
        color: var(--hbku-slate);
        font-size: 0.85rem;
    }

    .metric-value {
        color: var(--hbku-ink);
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 2px;
    }

    .metric-delta {
        color: var(--hbku-slate);
        font-size: 0.8rem;
        margin-top: 2px;
    }

    .stFileUploader, .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #ffffff;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: var(--hbku-ink);
    }

    [data-testid="stMetricLabel"] {
        color: var(--hbku-slate);
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(138, 21, 56, 0.15);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 10px 18px rgba(31, 41, 55, 0.08);
    }

    .regime-card [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(138, 21, 56, 0.14), rgba(198, 161, 91, 0.18));
        border: 1px solid rgba(138, 21, 56, 0.2);
    }

    .stDataFrame {
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 12px;
        overflow: hidden;
    }

    .info-card {
        background:
            linear-gradient(120deg, rgba(138, 21, 56, 0.95) 0%, rgba(198, 161, 91, 0.9) 100%);
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 14px 28px rgba(138, 21, 56, 0.2);
        margin: 24px 0;
        color: #ffffff;
    }
    .card-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 10px;
        text-align: center;
    }
    .card-subtitle {
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
        opacity: 0.95;
    }
    .card-footer {
        font-size: 14px;
        text-align: center;
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        font-style: italic;
    }
    .justified-text {
        text-align: justify;
        line-height: 1.8;
        margin-bottom: 20px;
    }
    .justified-text ul {
        text-align: justify;
        line-height: 1.8;
    }
    .justified-text li {
        margin-bottom: 10px;
    }

    .hbku-brand {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 18px;
        border: 1px solid rgba(138, 21, 56, 0.15);
        box-shadow: 0 10px 18px rgba(31, 41, 55, 0.08);
    }

    .hbku-brand .title {
        font-family: "Crimson Pro", "Times New Roman", serif;
        font-size: 1.35rem;
        color: var(--hbku-maroon);
        font-weight: 700;
        margin-bottom: 4px;
    }

    .hbku-brand .subtitle {
        font-size: 0.95rem;
        color: var(--hbku-slate);
    }

    .stExpander {
        border-radius: 12px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div class="hbku-brand">
        <div class="title">Hamad Bin Khalifa University</div>
        <div class="subtitle">Flow Regime Visual Twin</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")

    if st.button("Home", width="stretch"):
        st.session_state["page"] = "Home"
    if st.button("Classify Flow Regime", width="stretch"):
        st.session_state["page"] = "Classify Flow Regime"
    if st.button("User Guideline", width="stretch"):
        st.session_state["page"] = "User Guideline"
    if st.button("Privacy and Policy", width="stretch"):
        st.session_state["page"] = "Privacy and Policy"

# --- Page Selection ---
page = st.session_state["page"]

# ------------------------------------------------------------------------
# 🏠 HOME PAGE
# ------------------------------------------------------------------------
if page == "Home":
    st.title("MTPINN for Multiphase Flow Regime")
    
    st.markdown("""
    <div class="info-card">
        <div class="card-title">MTPINN</div>
        <div class="card-subtitle">
            Multi-Task Physics-Informed Neural Network for<br>
            Advanced Flow Regime Classification
        </div>
        <div class="card-footer">
            This research is led by Dr. Amith Khandakar and Dr. Mohammad Azizur Rahman
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("About Multiphase Flow Regime")
    st.markdown("""
    <div class="justified-text">
    Multiphase flow refers to the simultaneous flow of materials with different phases (gas, liquid, and/or solid)
    within pipelines or process systems. Understanding the flow regime (such as bubbly, slug, annular, or dispersed flows)
    is critical because it affects pressure drop, heat transfer, and mass transport efficiency.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Impact of Flow Regimes in Multiphase Systems")
    st.markdown("""
    <div class="justified-text">
    Different flow regimes have a significant impact on system performance and safety:
    <ul>
    <li><strong>Pressure Drop:</strong> Certain regimes (like slug flow) can cause large fluctuations in pressure.</li>
    <li><strong>Separation Efficiency:</strong> Flow regime affects the performance of separators.</li>
    <li><strong>Equipment Design:</strong> Correct prediction of flow regime is essential for pumps, pipelines, and reactors.</li>
    <li><strong>Operational Safety:</strong> Unstable flow regimes can lead to erosion, vibration, and operational hazards.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/flow-regime.png", caption="Common Multiphase Flow Regimes", width="stretch")

    st.subheader("How the MTPINN Works")
    st.markdown("""
    <div class="justified-text">
    The Multi-Task Physics-Informed Neural Network (MTPINN) combines deep learning with physics-based constraints 
    to accurately classify multiphase flow regimes. The system processes experimental video data through a sophisticated 
    pipeline that extracts temporal and spatial features, integrates physical laws of fluid dynamics, and performs 
    multi-task learning to simultaneously predict flow patterns and estimate key flow parameters.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.image("assets/Methodoloy-Fig.svg", caption="MTPINN Methodology and Architecture", width="stretch")

# ------------------------------------------------------------------------
# 📊 EVALUATE MODEL PAGE
# ------------------------------------------------------------------------
elif page == "Classify Flow Regime":
    st.title("Classify Flow Regime")
    st.markdown("""
    Upload a **CSV or Excel file** containing flow measurement data with a **'Pressure (barA)'** column.  
    The system will visualize the data and predict the **flow regime** using the trained MTPINN model.
    """)

    # Load model
    model, scalers = load_model_and_scalers()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if the model file exists at 'models/best_multitask_pinn_fold_5.pth'")
    else:
        st.subheader("1. Upload and Inspect Data")
        uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # --- Load Data ---
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ File uploaded successfully!")

            # Check for pressure column
            pressure_col = None
            for col in df.columns:
                if 'pressure' in col.lower():
                    pressure_col = col
                    break

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            preview_tab, viz_tab = st.tabs(["Data Preview", "Visualizations"])
            with preview_tab:
                st.dataframe(df.head(10), width="content")
                st.caption(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            with viz_tab:
                if pressure_col is None:
                    st.info("Upload a dataset with a pressure column to view visualizations.")
                elif len(numeric_cols) >= 1:
                    st.subheader("📈 Data Visualization")

                    # Identify time column
                    time_col = None
                    time_keywords = ['time', 'timestamp', 'date', 't', 'sec', 'second']
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in time_keywords):
                            time_col = col
                            break
                    
                    # Plot pressure signal
                    st.markdown("**Pressure Signal:**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    if time_col and time_col in df.columns:
                        ax.plot(df[time_col], df[pressure_col], linewidth=1.5, color='#8a1538')
                        ax.set_xlabel(time_col, fontsize=10)
                    else:
                        ax.plot(df.index, df[pressure_col], linewidth=1.5, color='#8a1538')
                        ax.set_xlabel("Sample Index", fontsize=10)
                    
                    ax.set_title("Pressure Signal Over Time", fontsize=12)
                    ax.set_ylabel("Pressure (barA)", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Correlation heatmap if multiple numeric columns
                    plot_cols = [col for col in numeric_cols if col != time_col]
                    if len(plot_cols) >= 2:
                        st.markdown("**Correlation Between Features:**")
                        corr = df[plot_cols[:5]].corr()
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        cax = ax2.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                        fig2.colorbar(cax)
                        ax2.set_xticks(range(len(corr.columns)))
                        ax2.set_yticks(range(len(corr.columns)))
                        ax2.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
                        ax2.set_yticklabels(corr.columns, fontsize=9)
                        ax2.set_title("Feature Correlation Heatmap", pad=20, fontsize=12)
                        
                        for i in range(len(corr.columns)):
                            for j in range(len(corr.columns)):
                                ax2.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                                        ha='center', va='center', color='black', fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig2)

            if pressure_col is None:
                st.error("❌ No 'Pressure' column found in the dataset. Please ensure your file contains a pressure column.")
            else:
                st.divider()
                # --- Flow Parameter Input ---
                st.subheader("2. Define Flow Parameters")
                input_mode = st.radio(
                    "Select input type",
                    ["Mass flow rates", "Volume flow rates", "Direct Vsg/Vsl"],
                    horizontal=True
                )

                if input_mode == "Mass flow rates":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mdot_l = st.number_input("m_dot_L (kg/s)", min_value=0.0, value=0.5, step=0.1)
                        p_in = st.number_input("P_in (Pa)", min_value=0.0, value=101325.0, step=100.0)
                        rho_l = st.number_input("rho_L (kg/m^3)", min_value=0.0, value=1000.0, step=10.0)
                        mu_l = st.number_input("mu_L (Pa*s)", min_value=0.0, value=0.001, step=0.0001, format="%.6f")
                    with col2:
                        mdot_g = st.number_input("m_dot_G (kg/s)", min_value=0.0, value=0.1, step=0.05)
                        t = st.number_input("T (K)", min_value=0.0, value=298.0, step=1.0)
                        mu_g = st.number_input("mu_G (Pa*s)", min_value=0.0, value=1.8e-5, step=1e-6, format="%.7f")
                        d = st.number_input("D (m)", min_value=0.0, value=0.05, step=0.005, format="%.4f")
                    with col3:
                        z = st.number_input("Z (-)", min_value=0.0, value=1.0, step=0.01)
                        r_g = st.number_input(
                            "R_g (J*kg^-1*K^-1)",
                            min_value=0.0,
                            value=287.0,
                            step=1.0,
                            disabled=True
                        )

                elif input_mode == "Volume flow rates":
                    default_d = 0.05
                    default_u_sl = 0.5
                    default_u_sg = 0.5
                    default_area = np.pi * default_d**2 / 4
                    default_q_l = default_u_sl * default_area
                    default_q_g = default_u_sg * default_area

                    st.caption("U_sl and U_sg are used as reference checks. Computations use q_L and q_G.")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        q_l = st.number_input("q_L (m^3/s)", min_value=0.0, value=default_q_l, step=1e-4, format="%.6f")
                        u_sl = st.number_input("U_sl (m/s)", min_value=0.0, value=default_u_sl, step=0.05)
                        p_in = st.number_input("P_in (Pa)", min_value=0.0, value=101325.0, step=100.0)
                        rho_l = st.number_input("rho_L (kg/m^3)", min_value=0.0, value=1000.0, step=10.0)
                    with col2:
                        q_g = st.number_input("q_G (m^3/s)", min_value=0.0, value=default_q_g, step=1e-4, format="%.6f")
                        u_sg = st.number_input("U_sg (m/s)", min_value=0.0, value=default_u_sg, step=0.05)
                        t = st.number_input("T (K)", min_value=0.0, value=298.0, step=1.0)
                        mu_l = st.number_input("mu_L (Pa*s)", min_value=0.0, value=0.001, step=0.0001, format="%.6f")
                    with col3:
                        z = st.number_input("Z (-)", min_value=0.0, value=1.0, step=0.01)
                        mu_g = st.number_input("mu_G (Pa*s)", min_value=0.0, value=1.8e-5, step=1e-6, format="%.7f")
                        d = st.number_input("D (m)", min_value=0.0, value=default_d, step=0.005, format="%.4f")
                        r_g = st.number_input(
                            "R_g (J*kg^-1*K^-1)",
                            min_value=0.0,
                            value=287.0,
                            step=1.0,
                            disabled=True
                        )

                    use_phase_velocities = st.checkbox("Provide u_L and u_G (optional)", value=False)
                    u_l = None
                    u_g = None
                    if use_phase_velocities:
                        col4, col5 = st.columns(2)
                        with col4:
                            u_l = st.number_input("u_L (m/s)", min_value=0.0, value=0.5, step=0.05)
                        with col5:
                            u_g = st.number_input("u_G (m/s)", min_value=0.0, value=0.5, step=0.05)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        direct_vsg = st.number_input(
                            "Superficial Gas Velocity (Vsg) [m/s]",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.5,
                            step=0.1
                        )
                    with col2:
                        direct_vsl = st.number_input(
                            "Superficial Liquid Velocity (Vsl) [m/s]",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.5,
                            step=0.1
                        )

                # --- Model Prediction ---
                st.markdown("""
                <div class="cta-card">
                    <div class="cta-title">Run Flow Regime Classification</div>
                    <div class="cta-subtitle">
                         Click below to see the classification results.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
                with cta_col2:
                    run_prediction = st.button("🔍 Predict Flow Regime", width="stretch")

                if run_prediction:
                    with st.spinner("Running prediction..."):
                        pressure_data = df[pressure_col].values

                        derived_outputs = []
                        vsg_input = None
                        vsl_input = None

                        if input_mode == "Mass flow rates":
                            mass_inputs = {
                                "mdot_l": mdot_l,
                                "mdot_g": mdot_g,
                                "p_in": p_in,
                                "t": t,
                                "z": z,
                                "rho_l": rho_l,
                                "mu_l": mu_l,
                                "mu_g": mu_g,
                                "d": d,
                                "r_g": r_g,
                            }
                            derived_outputs, vsg_input, vsl_input = compute_mass_flow_outputs(mass_inputs)
                        elif input_mode == "Volume flow rates":
                            volume_inputs = {
                                "q_l": q_l,
                                "q_g": q_g,
                                "u_sl": u_sl,
                                "u_sg": u_sg,
                                "p_in": p_in,
                                "t": t,
                                "z": z,
                                "rho_l": rho_l,
                                "mu_l": mu_l,
                                "mu_g": mu_g,
                                "d": d,
                                "r_g": r_g,
                                "u_l": u_l,
                                "u_g": u_g,
                            }
                            derived_outputs, u_sl_input, u_sg_input, u_sl_calc, u_sg_calc, vsg_input, vsl_input = compute_volume_flow_outputs(volume_inputs)

                            tolerance = 0.05
                            if u_sl_input > 0 and abs(u_sl_input - u_sl_calc) > tolerance * max(abs(u_sl_calc), 1e-9):
                                st.warning("U_sl input does not match q_L / A. Outputs use q_L and q_G.")
                            if u_sg_input > 0 and abs(u_sg_input - u_sg_calc) > tolerance * max(abs(u_sg_calc), 1e-9):
                                st.warning("U_sg input does not match q_G / A. Outputs use q_L and q_G.")
                        else:
                            derived_outputs = []
                            vsg_input = direct_vsg
                            vsl_input = direct_vsl

                        if vsg_input is None or vsl_input is None or np.isnan(vsg_input) or np.isnan(vsl_input):
                            st.error("❌ Unable to compute superficial velocities. Please check your inputs.")
                            result = None
                        else:
                            result = predict_flow_regime(model, scalers, pressure_data, vsg_input, vsl_input)
                        
                        if result:
                            st.markdown('<div id="prediction-results-anchor"></div>', unsafe_allow_html=True)
                            components.html("""
                            <script>
                              (function() {
                                const doc = window.parent.document || document;
                                let tries = 0;
                                const scrollToResults = () => {
                                  const el = doc.getElementById("prediction-results-anchor");
                                  if (el) {
                                    el.scrollIntoView({ behavior: "smooth", block: "start" });
                                    return;
                                  }
                                  tries += 1;
                                  if (tries < 30) {
                                    setTimeout(scrollToResults, 200);
                                  }
                                };
                                setTimeout(scrollToResults, 200);
                              })();
                            </script>
                            """, height=0)
                            st.divider()
                            st.subheader("3. Prediction Results")
                            
                            predicted_class_name = CLASS_NAMES[result['predicted_class']]
                            
                            st.markdown("**Prediction Summary:**")
                            st.markdown('<div class="regime-card">', unsafe_allow_html=True)
                            st.metric("Predicted Flow Regime", predicted_class_name)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Display flow regime GIF/image
                            gif_mapping = {
                                "Dispersed Flow": "video_library/Dispersed-Flow/Dispersed-Flow.gif",
                                "Slug Flow": "video_library/Slug-Flow/Slug-Flow.gif",
                                "Plug Flow": "video_library/Plug-Flow/Plug-Flow.gif"
                            }
                            
                            image_mapping = {
                                "Dispersed Flow": "video_library/Dispersed-Flow/Dispersed-Flow.png",
                                "Slug Flow": "video_library/Slug-Flow/Slug-Flow.png",
                                "Plug Flow": "video_library/Plug-Flow/Plug-Flow.png"
                            }
                            
                            # Try GIF first, then fallback to PNG
                            gif_path = gif_mapping.get(predicted_class_name)
                            image_path = image_mapping.get(predicted_class_name)
                            
                            media_path = None
                            if gif_path and os.path.exists(gif_path):
                                media_path = gif_path
                            elif image_path and os.path.exists(image_path):
                                media_path = image_path
                            
                            vis_col, derived_col = st.columns([1.1, 1])
                            with vis_col:
                                if media_path:
                                    render_media_card(
                                        media_path,
                                        f"{predicted_class_name} Visualization",
                                        f"{predicted_class_name}"
                                    )
                                else:
                                    st.warning(f"⚠️ Visualization not found for {predicted_class_name}")
                            with derived_col:
                                if derived_outputs:
                                    build_output_table(
                                        derived_outputs,
                                        vsg_pred=result['vsg_predicted'],
                                        vsl_pred=result['vsl_predicted'],
                                        vsg_input=vsg_input,
                                        vsl_input=vsl_input
                                    )

        else:
            st.info("👆 Please upload a file to begin visualization and prediction.")

# ------------------------------------------------------------------------
# 📋 USER GUIDELINE PAGE
# ------------------------------------------------------------------------
elif page == "User Guideline":
    st.title("User Guideline")
    st.markdown("""
    <div class="justified-text">
    <h3>How to Use the Flow Regime Classification System</h3>
    
    <h4>Step 1: Prepare Your Data</h4>
    Ensure your dataset meets the following requirements:
    <ul>
    <li>File format: CSV or Excel (.xlsx)</li>
    <li>Must contain a column with "Pressure" in its name (e.g., "Pressure (barA)", "Pressure_Data")</li>
    <li>Minimum of 40 data points for analysis</li>
    <li>Pressure values should be in barA (absolute pressure)</li>
    </ul>
    
    <h4>Step 2: Navigate to Evaluate Model</h4>
    Click on the <strong>"Evaluate Model"</strong> button in the sidebar to access the prediction interface.
    
    <h4>Step 3: Upload Your Dataset</h4>
    <ul>
    <li>Click the "Browse files" button</li>
    <li>Select your CSV or Excel file containing pressure measurements</li>
    <li>The system will automatically detect and validate the pressure column</li>
    <li>Preview your data to ensure it loaded correctly</li>
    </ul>
    
    <h4>Step 4: Enter Flow Parameters</h4>
    Input the operating conditions for your multiphase flow system:
    <ul>
    <li><strong>Superficial Gas Velocity (Vsg):</strong> Enter the gas phase velocity in m/s (range: 0-10 m/s)</li>
    <li><strong>Superficial Liquid Velocity (Vsl):</strong> Enter the liquid phase velocity in m/s (range: 0-10 m/s)</li>
    </ul>
    
    <h4>Step 5: Run Prediction</h4>
    <ul>
    <li>Click the <strong>"Predict Flow Regime"</strong> button</li>
    <li>The system will analyze your pressure data using sliding windows</li>
    <li>Processing typically takes a few seconds depending on data size</li>
    </ul>
    
    <h4>Step 6: Interpret Results</h4>
    The prediction results include:
    <ul>
    <li><strong>Predicted Flow Regime:</strong> The most likely flow pattern (Dispersed, Plug, or Slug Flow)</li>
    <li><strong>Confidence Score:</strong> Model confidence in the prediction (0-100%)</li>
    <li><strong>Class Probabilities:</strong> Probability distribution across all flow regime classes</li>
    <li><strong>Predicted Velocities:</strong> Model's estimate of Vsg and Vsl based on pressure patterns</li>
    <li><strong>Flow Visualization:</strong> Representative image of the predicted flow regime</li>
    </ul>
    
    <h4>Understanding Flow Regimes</h4>
    <ul>
    <li><strong>Dispersed Flow:</strong> Small gas bubbles uniformly distributed in continuous liquid phase</li>
    <li><strong>Plug Flow:</strong> Large elongated gas bubbles (plugs) separated by liquid slugs</li>
    <li><strong>Slug Flow:</strong> Intermittent flow with alternating gas pockets and liquid slugs</li>
    </ul>
    
    <h4>Tips for Best Results</h4>
    <ul>
    <li>Use data collected at steady-state conditions for more accurate predictions</li>
    <li>Provide accurate velocity values that match your experimental conditions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------
# 🔒 PRIVACY AND POLICY PAGE
# ------------------------------------------------------------------------
elif page == "Privacy and Policy":
    st.title("Privacy and Policy")
    st.markdown("""
    <div class="justified-text">
    
    <h3>Data Privacy and Security</h3>
    
    <h4>Data Collection and Usage</h4>
    The MTPINN Flow Regime Classification System is designed with privacy and security in mind. We are committed to protecting your data and ensuring transparency in how it is processed.
    
    <h4>What Data We Process</h4>
    <ul>
    <li><strong>Uploaded Files:</strong> CSV and Excel files containing pressure measurement data and flow parameters</li>
    <li><strong>Input Parameters:</strong> Superficial gas velocity (Vsg) and superficial liquid velocity (Vsl) values</li>
    <li><strong>Session Data:</strong> Temporary navigation state stored locally in your browser</li>
    </ul>
    
    <h4>How We Handle Your Data</h4>
    <ul>
    <li><strong>Local Processing:</strong> All uploaded data is processed locally in real-time and is not stored on any server</li>
    <li><strong>No Data Storage:</strong> We do not permanently store, save, or retain any uploaded files or user data</li>
    <li><strong>Session-Based:</strong> Data exists only during your active session and is automatically discarded when you close the application</li>
    <li><strong>No Third-Party Sharing:</strong> Your data is never shared with, sold to, or transferred to any third parties</li>
    <li><strong>No User Tracking:</strong> We do not track user behavior, collect personal information, or use cookies for analytics</li>
    </ul>
    
    <h4>Model and Algorithm</h4>
    <ul>
    <li>The classification model is a Multi-Task Physics-Informed Neural Network (MTPINN)</li>
    <li>The model processes pressure signals, extracts features, and predicts flow regimes based on learned patterns</li>
    <li>Predictions are made using pre-trained model weights and do not require external API calls</li>
    <li>All computations are performed locally on the server hosting this application</li>
    </ul>
    
    <h4>Research and Academic Use</h4>
    This application is developed for research and educational purposes at Qatar University under the leadership of:
    <ul>
    <li>Dr. Amith Khandakar</li>
    <li>Dr. Mohammad Azizur Rahman</li>
    </ul>
    
    The MTPINN model integrates:
    <ul>
    <li>Deep learning for pattern recognition in pressure signals</li>
    <li>Physics-informed constraints based on fluid dynamics principles</li>
    <li>Multi-task learning for simultaneous flow regime classification and velocity prediction</li>
    </ul>
    
    <h4>Limitations and Disclaimers</h4>
    <ul>
    <li><strong>Experimental Tool:</strong> This application is a research prototype and should not be used as the sole basis for critical operational decisions</li>
    <li><strong>Accuracy:</strong> While the model achieves high accuracy on test data, predictions may vary based on data quality and operating conditions</li>
    <li><strong>Scope:</strong> The model is trained on specific flow conditions and may not generalize to all multiphase flow scenarios</li>
    <li><strong>No Warranty:</strong> The application is provided "as is" without warranties of any kind, express or implied</li>
    </ul>
    
    <h4>User Responsibilities</h4>
    <ul>
    <li>Ensure uploaded data does not contain sensitive, proprietary, or confidential information beyond flow measurements</li>
    <li>Verify that your use of this application complies with your organization's data policies</li>
    <li>Validate predictions against experimental observations or other measurement methods</li>
    <li>Do not rely solely on model predictions for safety-critical applications</li>
    </ul>
    
    <h4>Future Development</h4>
    Future versions of this system may integrate:
    <ul>
    <li>CNN-based video feature extraction for visual flow pattern analysis</li>
    <li>Enhanced physics-informed constraints for improved accuracy</li>
    <li>Expanded multi-task learning capabilities for additional flow parameters</li>
    <li>Support for additional flow regimes and operating conditions</li>
    </ul>
    
    <h4>Contact and Support</h4>
    For questions, feedback, or research collaboration inquiries, please contact:
    <ul>
    <li>Qatar University Research Team</li>
    </ul>
    
    <h4>Updates to This Policy</h4>
    This privacy policy may be updated periodically to reflect changes in functionality or data handling practices. Users will be notified of significant changes through the application interface.
    
    <h4>Acceptance of Terms</h4>
    By using this application, you acknowledge that you have read and understood this privacy policy and agree to its terms regarding data processing and usage limitations.
    
    <hr>
    
    <p style="text-align: center; font-size: 0.9em; color: #999;">
    <strong>Last Updated:</strong> November 2024<br>
    <strong>Version:</strong> 1.0<br>
    © Qatar University Research Team
    </p>
    
    </div>
    """, unsafe_allow_html=True)
