"""
Model utilities for loading and inference
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import re
from scipy.fft import fft, fftfreq


class MultiTaskPINN(nn.Module):
    """Multi-Task Physics-Informed Neural Network"""
    
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
        
        conv_output_size = 64 * (input_size // 2)
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
        
        return class_output, velocity_output, physics_output, shared_repr


class FlowRegimePredictor:
    """Main predictor class for flow regime classification"""
    
    def __init__(self, model_path, scalers_path, window_size=40, stride=20):
        self.window_size = window_size
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        self.class_names = ["Dispersed Flow", "Plug Flow", "Slug Flow"]
        
        # Load model
        self.model = MultiTaskPINN(input_size=window_size).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store model info
        self.model_info = {
            'val_acc': checkpoint.get('val_acc', None),
            'val_velocity_mae': checkpoint.get('val_velocity_mae', None)
        }
        
        # Load scalers
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        self.scaler_features = scalers['features']
        self.scaler_pressure = scalers['pressure']
        self.scaler_vsg = scalers['vsg']
        self.scaler_vsl = scalers['vsl']
    
    def extract_velocities_from_filename(self, filename):
        """Extract Vsg and Vsl from filename"""
        vsg_match = re.search(r'Vsg=([\d.]+)', filename)
        vsl_match = re.search(r'Vsl=([\d.]+)', filename)
        vsg = float(vsg_match.group(1).rstrip('.')) if vsg_match else None
        vsl = float(vsl_match.group(1).rstrip('.')) if vsl_match else None
        return vsg, vsl
    
    def extract_features(self, pressure_window):
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
            freqs = fftfreq(len(pressure_window), d=0.5)
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
    
    def predict_from_file(self, file_path, manual_vsg=None, manual_vsl=None):
        """Predict flow regime from file"""
        # Load data
        df = pd.read_excel(file_path)
        
        # Find pressure column
        pressure_col = None
        for col in ['Pressure/bar', 'Pressure (barA)', 'Pressure']:
            if col in df.columns:
                pressure_col = col
                break
        
        if pressure_col is None:
            raise ValueError(f"No pressure column found. Available: {df.columns.tolist()}")
        
        pressure = df[pressure_col].values
        filename = os.path.basename(file_path)
        
        # Get velocities
        if manual_vsg is not None and manual_vsl is not None:
            vsg_input = manual_vsg
            vsl_input = manual_vsl
        else:
            vsg_input, vsl_input = self.extract_velocities_from_filename(filename)
            if vsg_input is None or vsl_input is None:
                raise ValueError("Could not extract velocities from filename and manual values not provided")
        
        # Create windows
        windows = []
        for i in range(0, len(pressure) - self.window_size + 1, self.stride):
            window = pressure[i:i+self.window_size]
            if len(window) == self.window_size:
                windows.append(window)
        
        if len(windows) == 0:
            raise ValueError(f"No valid windows created. Pressure length: {len(pressure)}")
        
        windows = np.array(windows)
        
        # Extract features
        features_list = [self.extract_features(window) for window in windows]
        features = np.array(features_list)
        
        # Scale inputs
        pressure_scaled = self.scaler_pressure.transform(windows)
        features_scaled = self.scaler_features.transform(features)
        
        vsg_scaled = self.scaler_vsg.transform([[vsg_input]])[0]
        vsl_scaled = self.scaler_vsl.transform([[vsl_input]])[0]
        velocities_scaled = np.hstack([vsg_scaled, vsl_scaled])
        velocities_input = np.tile(velocities_scaled, (len(windows), 1))
        
        # Convert to tensors
        pressure_tensor = torch.FloatTensor(pressure_scaled).to(self.device)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        velocities_tensor = torch.FloatTensor(velocities_input).to(self.device)
        
        # Inference
        with torch.no_grad():
            class_output, velocity_output, physics_output, embeddings = self.model(
                pressure_tensor, features_tensor, velocities_tensor
            )
            
            # Get predictions
            class_probs = torch.softmax(class_output, dim=1)
            avg_class_probs = class_probs.mean(dim=0).cpu().numpy()
            predicted_class = torch.argmax(torch.tensor(avg_class_probs)).item()
            
            # Average velocity predictions
            avg_velocity = velocity_output.mean(dim=0).cpu().numpy()
            vsg_pred = self.scaler_vsg.inverse_transform(avg_velocity[0].reshape(-1, 1))[0, 0]
            vsl_pred = self.scaler_vsl.inverse_transform(avg_velocity[1].reshape(-1, 1))[0, 0]
            
            # Average embedding
            avg_embedding = embeddings.mean(dim=0).cpu().numpy()
        
        results = {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'class_probabilities': avg_class_probs,
            'predicted_vsg': vsg_pred,
            'predicted_vsl': vsl_pred,
            'vsg_true': vsg_input,
            'vsl_true': vsl_input,
            'embedding': avg_embedding,
            'num_windows': len(windows)
        }
        
        return results
    
    def retrieve_videos(self, query_embedding, query_class, video_library_path, 
                       top_k=5, similarity_metric='cosine'):
        """Retrieve similar videos from library"""
        # Load video library
        video_df = pd.read_csv(video_library_path)
        
        # Filter by class
        same_regime = video_df[video_df['flow_regime_idx'] == query_class].copy()
        
        if len(same_regime) == 0:
            return pd.DataFrame()
        
        # Compute embeddings for library videos (simplified - use synthetic)
        similarities = []
        for idx, row in same_regime.iterrows():
            # In real application, you would load actual embeddings
            # For now, use velocity-based similarity
            vsg_diff = abs(row['vsg'] - query_embedding.mean())
            vsl_diff = abs(row['vsl'] - query_embedding.mean())
            similarity = 1.0 / (1.0 + vsg_diff + vsl_diff)
            similarities.append(similarity)
        
        same_regime['similarity_score'] = similarities
        
        # Get top-k
        top_videos = same_regime.nlargest(min(top_k, len(same_regime)), 'similarity_score')
        
        return top_videos