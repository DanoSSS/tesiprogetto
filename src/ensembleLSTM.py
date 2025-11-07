"""
LSTM Ensemble System for MEMS Beam Displacement Prediction
===========================================================

Sistema LSTM con meccanismo di attenzione per previsione spostamento verticale
di trave incastrata-incastrata da simulazioni COMSOL con attuazione elettrica
periodica (wave) e a gradino (step).

Architecture:
- Separate models trained on wave and step actuation types
- Each model uses attention mechanism for improved temporal awareness
- Features integration: geometric parameters (beam_length, beam_height, air_gap)
- Output: 2 displacement predictions (mid-span and 3/4-span)

Author: Claude Code
Date: 2025
"""

import glob
import os
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class BeamDataset(Dataset):
    """
    Custom dataset for COMSOL beam displacement data.

    Handles loading and preprocessing of vertical displacement time-series
    with geometric parameters and actuation type information.

    Data Loading Behavior:
    - 'wave': Loads ONLY vertdisptime_*.txt files (sinusoidal actuation)
    - 'step': Loads ONLY stepvertdisptime_*.txt files (step actuation)
    - 'both': Loads both file types separately
    """

    def __init__(self, data_dir: str, actuation_type: str = 'both',
                 sequence_length: int = 50, stride: int = 10):
        """
        Args:
            data_dir: Directory containing .txt files
            actuation_type: 'wave' (vertdisptime_*.txt only),
                           'step' (stepvertdisptime_*.txt only),
                           or 'both' (both file types)
            sequence_length: Length of temporal sequences
            stride: Step size for creating overlapping sequences

        File Naming Convention:
            Wave files:  vertdisptime_*.txt
            Step files:  stepvertdisptime_*.txt
        """
        self.data_dir = data_dir
        self.actuation_type = actuation_type
        self.sequence_length = sequence_length
        self.stride = stride

        # Scalers for normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.geom_scaler = StandardScaler()

        # Load and preprocess data
        self.load_data()

    def load_data(self):
        """Load and preprocess data from COMSOL files"""
        all_sequences = []
        all_targets = []
        all_geom_features = []
        all_voltage_info = []

        # Define file patterns based on actuation type
        if self.actuation_type == 'wave':
            # Wave actuation: load ONLY vertdisptime_*.txt files (exclude step files)
            pattern = 'vertdisptime_*.txt'
            print(f"\nLoading WAVE actuation data from: {pattern}")
        elif self.actuation_type == 'step':
            # Step actuation: load ONLY stepvertdisptime_*.txt files (exclude wave files)
            pattern = 'stepvertdisptime_*.txt'
            print(f"\nLoading STEP actuation data from: {pattern}")
        else:  # both
            # Both: load wave and step separately for clarity
            pattern = None
            print(f"\nLoading BOTH wave and step actuation data")

        # Load displacement files
        if pattern:
            # Single pattern for wave or step
            files = glob.glob(os.path.join(self.data_dir, pattern))
            print(f"Found {len(files)} files matching pattern: {pattern}")

            for file in files:
                try:
                    self._process_file(file, all_sequences, all_targets,
                                     all_geom_features, all_voltage_info)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
        else:
            # Load both wave and step separately for 'both' mode
            for pattern in ['vertdisptime_*.txt', 'stepvertdisptime_*.txt']:
                files = glob.glob(os.path.join(self.data_dir, pattern))
                print(f"Found {len(files)} files matching pattern: {pattern}")

                for file in files:
                    try:
                        self._process_file(file, all_sequences, all_targets,
                                         all_geom_features, all_voltage_info)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue

        # Convert to numpy arrays
        if all_sequences:
            self.sequences = np.array(all_sequences)
            self.targets = np.array(all_targets)
            self.geom_features = np.array(all_geom_features)
            self.voltage_info = np.array(all_voltage_info) if all_voltage_info else None

            print(f"Loaded {len(self.sequences)} sequences")

            # Normalize data
            self._normalize_data()
        else:
            print("Warning: No sequences loaded!")
            self.sequences = np.empty((0, self.sequence_length, 2))
            self.targets = np.empty((0, 2))
            self.geom_features = np.empty((0, 3))
            self.voltage_info = None

    def _process_file(self, file_path: str, all_sequences: List, all_targets: List,
                     all_geom_features: List, all_voltage_info: List):
        """
        Process a single displacement file.

        Expected format:
        Column 0: Vbase (voltage or param)
        Column 1: beam_length (b_len)
        Column 2: beam_height (b_height)
        Column 3: air_gap
        Column 4: time
        Column 5: displacement_mid (L/2)
        Column 6: displacement_3/4 (3L/4)
        """
        data = pd.read_csv(file_path, sep=r'\s+', header=None)

        if data.shape[1] < 7:
            print(f"Skipping {file_path}: insufficient columns")
            return

        # Extract geometric parameters and voltage info
        vbase = data.iloc[0, 0]
        b_len = data.iloc[0, 1]
        b_height = data.iloc[0, 2]
        air_gap = data.iloc[0, 3]

        # Extract time series
        time = data.iloc[:, 4].values
        disp_mid = data.iloc[:, 5].values
        disp_3q = data.iloc[:, 6].values

        # Create sliding window sequences
        for i in range(0, len(time) - self.sequence_length, self.stride):
            seq_time = time[i:i+self.sequence_length]
            seq_disp_mid = disp_mid[i:i+self.sequence_length]
            seq_disp_3q = disp_3q[i:i+self.sequence_length]

            # Combine displacements as input (seq_len, 2)
            sequence = np.column_stack([seq_disp_mid, seq_disp_3q])
            all_sequences.append(sequence)

            # Target: next displacement values
            if i + self.sequence_length < len(time):
                target = np.array([
                    disp_mid[i+self.sequence_length],
                    disp_3q[i+self.sequence_length]
                ])
                all_targets.append(target)
                all_geom_features.append([b_len, b_height, air_gap])
                all_voltage_info.append(vbase)

    def _normalize_data(self):
        """Normalize sequences, targets, and geometric features"""
        # Normalize input sequences
        seq_shape = self.sequences.shape
        self.sequences = self.sequences.reshape(-1, seq_shape[-1])
        self.sequences = self.input_scaler.fit_transform(self.sequences)
        self.sequences = self.sequences.reshape(seq_shape)

        # Normalize output targets
        self.targets = self.output_scaler.fit_transform(self.targets)

        # Normalize geometric features
        self.geom_features = self.geom_scaler.fit_transform(self.geom_features)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor(self.targets[idx]),
                torch.FloatTensor(self.geom_features[idx]))


class AttentionLayer(nn.Module):
    """Scaled dot-product attention mechanism"""

    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = np.sqrt(hidden_size)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        Q = self.query(lstm_output)  # (batch, seq_len, hidden_size)
        K = self.key(lstm_output)
        V = self.value(lstm_output)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, seq_len, hidden_size)
        context = context[:, -1, :]  # Take last time step (batch, hidden_size)

        return context, attention_weights


class AttentionBeamLSTM(nn.Module):
    """
    LSTM with attention mechanism for beam displacement prediction.

    Architecture:
    1. Bidirectional LSTM to capture temporal patterns
    2. Scaled dot-product attention to focus on important time steps
    3. Feature encoder for geometric parameters
    4. Output network combining LSTM context and features
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 2,
                 feature_size: int = 3, dropout: float = 0.2):
        """
        Args:
            input_size: Input dimension (disp_mid, disp_3q)
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            output_size: Output dimension (2 displacements)
            feature_size: Geometric features dimension
            dropout: Dropout rate
        """
        super(AttentionBeamLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)

        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x, features=None):
        """
        Forward pass.

        Args:
            x: Input sequences (batch, seq_len, input_size)
            features: Geometric features (batch, feature_size)

        Returns:
            output: Predictions (batch, output_size)
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        context, _ = self.attention(lstm_out)

        # Process features
        if features is not None:
            encoded_features = self.feature_encoder(features)
            combined = torch.cat([context, encoded_features], dim=1)
        else:
            combined = context

        # Output prediction
        output = self.output_net(combined)

        return output


class EnsembleBeamPredictor:
    """
    Ensemble system managing separate models for different actuation types.

    Maintains independent models for 'wave' and 'step' actuation, allowing
    specialized learning for each type of electrical driving.
    """

    def __init__(self, model_configs: Dict = None):
        """
        Args:
            model_configs: Configuration for each model type
        """
        self.models = {}
        self.scalers = {}
        self.model_configs = model_configs or {
            'wave': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'step': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}
        }

    def create_models(self):
        """Create LSTM models for each actuation type"""
        for name, config in self.model_configs.items():
            model = AttentionBeamLSTM(
                input_size=2,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=2,
                feature_size=3,
                dropout=config.get('dropout', 0.2)
            )
            self.models[name] = model
            print(f"Created model for '{name}' actuation type")

    def train_model(self, model_name: str, train_loader: DataLoader,
                   val_loader: DataLoader, epochs: int = 100,
                   learning_rate: float = 0.001, device: str = 'cpu'):
        """
        Train a single model.

        Args:
            model_name: Name of the model ('wave' or 'step')
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            device: 'cpu' or 'cuda'

        Returns:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
        """
        model = self.models[model_name].to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for sequences, targets, features in train_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                features = features.to(device)

                optimizer.zero_grad()
                outputs = model(sequences, features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sequences, targets, features in val_loader:
                    sequences = sequences.to(device)
                    targets = targets.to(device)
                    features = features.to(device)

                    outputs = model(sequences, features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"[{model_name}] Epoch [{epoch+1}/{epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f}")

        return train_losses, val_losses

    def predict(self, sequences: torch.Tensor, features: torch.Tensor,
               actuation_type: str = 'wave', device: str = 'cpu') -> np.ndarray:
        """
        Make predictions using the appropriate model.

        Args:
            sequences: Input sequences (batch, seq_len, 2)
            features: Geometric features (batch, 3)
            actuation_type: 'wave' or 'step'
            device: 'cpu' or 'cuda'

        Returns:
            predictions: numpy array of shape (batch, 2)
        """
        if actuation_type not in self.models:
            print(f"Model '{actuation_type}' not found. Using 'wave'.")
            actuation_type = 'wave'

        model = self.models[actuation_type].to(device)
        model.eval()

        with torch.no_grad():
            predictions = model(sequences.to(device), features.to(device))

        return predictions.cpu().numpy()

    def save_models(self, save_dir: str):
        """Save all models to directory"""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': self.model_configs[name]
            }
            torch.save(checkpoint, os.path.join(save_dir, f'{name}_model.pt'))
            print(f"Saved {name} model to {save_dir}")

    def load_models(self, save_dir: str):
        """Load models from directory"""
        for name in self.model_configs.keys():
            path = os.path.join(save_dir, f'{name}_model.pt')
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu')
                model = AttentionBeamLSTM(
                    hidden_size=checkpoint['model_config']['hidden_size'],
                    num_layers=checkpoint['model_config']['num_layers'],
                    dropout=checkpoint['model_config'].get('dropout', 0.2)
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                self.models[name] = model
                print(f"Loaded {name} model from {save_dir}")
            else:
                print(f"Warning: Model file {path} not found")


def visualize_training(train_losses: List, val_losses: List, title: str,
                      save_path: str = None):
    """Visualize training curves with publication quality"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].plot(train_losses, label='Training Loss', linewidth=2.5, marker='o', markersize=4)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{title} - Linear Scale', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(train_losses, label='Training Loss', linewidth=2.5, marker='o', markersize=4)
    axes[1].semilogy(val_losses, label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss (MSE) - Log Scale', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{title} - Log Scale', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")

    plt.show()


def visualize_predictions(true_values: np.ndarray, predictions: np.ndarray,
                         title: str = "Predictions", save_path: str = None):
    """Visualize prediction quality with publication-quality plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mid-span displacement: scatter
    axes[0, 0].scatter(true_values[:, 0], predictions[:, 0], alpha=0.6, s=20)
    min_val, max_val = true_values[:, 0].min(), true_values[:, 0].max()
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True Displacement (µm)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Displacement (µm)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Mid-span Displacement (L/2)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 3/4-span displacement: scatter
    axes[0, 1].scatter(true_values[:, 1], predictions[:, 1], alpha=0.6, s=20, color='orange')
    min_val, max_val = true_values[:, 1].min(), true_values[:, 1].max()
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('True Displacement (µm)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Predicted Displacement (µm)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('3/4-span Displacement (3L/4)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Errors over time
    errors_mid = np.abs(true_values[:, 0] - predictions[:, 0])
    axes[1, 0].plot(errors_mid, linewidth=1.5, color='blue', alpha=0.7)
    axes[1, 0].fill_between(range(len(errors_mid)), errors_mid, alpha=0.3)
    axes[1, 0].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Absolute Error (µm)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Mid-span Prediction Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 3/4-span errors
    errors_3q = np.abs(true_values[:, 1] - predictions[:, 1])
    axes[1, 1].plot(errors_3q, linewidth=1.5, color='orange', alpha=0.7)
    axes[1, 1].fill_between(range(len(errors_3q)), errors_3q, alpha=0.3, color='orange')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Absolute Error (µm)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('3/4-span Prediction Error', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction plot to {save_path}")

    plt.show()

    # Print metrics
    mse_mid = np.mean((true_values[:, 0] - predictions[:, 0])**2)
    mse_3q = np.mean((true_values[:, 1] - predictions[:, 1])**2)
    mae_mid = np.mean(errors_mid)
    mae_3q = np.mean(errors_3q)
    rmse_mid = np.sqrt(mse_mid)
    rmse_3q = np.sqrt(mse_3q)

    print("\n" + "="*60)
    print("PREDICTION METRICS")
    print("="*60)
    print(f"Mid-span (L/2):")
    print(f"  MSE:  {mse_mid:.6e}")
    print(f"  RMSE: {rmse_mid:.6e}")
    print(f"  MAE:  {mae_mid:.6e}")
    print(f"\n3/4-span (3L/4):")
    print(f"  MSE:  {mse_3q:.6e}")
    print(f"  RMSE: {rmse_3q:.6e}")
    print(f"  MAE:  {mae_3q:.6e}")
    print("="*60)


def main():
    """Main training pipeline"""

    print("\n" + "="*70)
    print("ENSEMBLE LSTM FOR MEMS BEAM DISPLACEMENT PREDICTION")
    print("="*70)

    # Configuration
    data_dir = "D:/exportfiles"
    batch_size = 16
    epochs = 100
    learning_rate = 0.001
    sequence_length = 50
    stride = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = "./results"

    print(f"\nUsing device: {device}")

    # 1. Load wave dataset
    print("\n[1] Loading wave actuation data...")
    wave_dataset = BeamDataset(data_dir, actuation_type='wave',
                              sequence_length=sequence_length, stride=stride)

    if len(wave_dataset) == 0:
        print("No wave data found. Skipping wave model training.")
        wave_dataset = None
    else:
        print(f"Loaded {len(wave_dataset)} wave sequences")

    # 2. Load step dataset
    print("\n[2] Loading step actuation data...")
    step_dataset = BeamDataset(data_dir, actuation_type='step',
                              sequence_length=sequence_length, stride=stride)

    if len(step_dataset) == 0:
        print("No step data found. Skipping step model training.")
        step_dataset = None
    else:
        print(f"Loaded {len(step_dataset)} step sequences")

    if wave_dataset is None and step_dataset is None:
        print("ERROR: No data loaded!")
        return

    # 3. Create ensemble
    print("\n[3] Creating ensemble predictor...")
    ensemble = EnsembleBeamPredictor()
    ensemble.create_models()

    # 4. Train models
    print("\n[4] Training models...")

    if wave_dataset is not None:
        print("\n--- Training Wave Model ---")
        train_size = int(0.7 * len(wave_dataset))
        val_size = int(0.15 * len(wave_dataset))
        test_size = len(wave_dataset) - train_size - val_size

        train_wave, val_wave, test_wave = torch.utils.data.random_split(
            wave_dataset, [train_size, val_size, test_size]
        )

        train_loader_wave = DataLoader(train_wave, batch_size=batch_size, shuffle=True)
        val_loader_wave = DataLoader(val_wave, batch_size=batch_size)
        test_loader_wave = DataLoader(test_wave, batch_size=batch_size)

        train_losses_wave, val_losses_wave = ensemble.train_model(
            'wave', train_loader_wave, val_loader_wave,
            epochs=epochs, learning_rate=learning_rate, device=device
        )

        # Test wave model
        print("\nTesting wave model...")
        model = ensemble.models['wave'].to(device)
        model.eval()

        predictions_wave = []
        targets_wave = []

        with torch.no_grad():
            for sequences, targets, features in test_loader_wave:
                sequences = sequences.to(device)
                features = features.to(device)

                outputs = model(sequences, features)
                predictions_wave.append(outputs.cpu().numpy())
                targets_wave.append(targets.numpy())

        predictions_wave = np.vstack(predictions_wave)
        targets_wave = np.vstack(targets_wave)

        # Denormalize
        predictions_wave = wave_dataset.output_scaler.inverse_transform(predictions_wave)
        targets_wave = wave_dataset.output_scaler.inverse_transform(targets_wave)

        # Visualize
        visualize_predictions(targets_wave, predictions_wave,
                            title="Wave Actuation - Model Predictions",
                            save_path=f"{results_dir}/wave_predictions.png")
        visualize_training(train_losses_wave, val_losses_wave,
                         title="Wave Model Training",
                         save_path=f"{results_dir}/wave_training_curves.png")

    if step_dataset is not None:
        print("\n--- Training Step Model ---")
        train_size = int(0.7 * len(step_dataset))
        val_size = int(0.15 * len(step_dataset))
        test_size = len(step_dataset) - train_size - val_size

        train_step, val_step, test_step = torch.utils.data.random_split(
            step_dataset, [train_size, val_size, test_size]
        )

        train_loader_step = DataLoader(train_step, batch_size=batch_size, shuffle=True)
        val_loader_step = DataLoader(val_step, batch_size=batch_size)
        test_loader_step = DataLoader(test_step, batch_size=batch_size)

        train_losses_step, val_losses_step = ensemble.train_model(
            'step', train_loader_step, val_loader_step,
            epochs=epochs, learning_rate=learning_rate, device=device
        )

        # Test step model
        print("\nTesting step model...")
        model = ensemble.models['step'].to(device)
        model.eval()

        predictions_step = []
        targets_step = []

        with torch.no_grad():
            for sequences, targets, features in test_loader_step:
                sequences = sequences.to(device)
                features = features.to(device)

                outputs = model(sequences, features)
                predictions_step.append(outputs.cpu().numpy())
                targets_step.append(targets.numpy())

        predictions_step = np.vstack(predictions_step)
        targets_step = np.vstack(targets_step)

        # Denormalize
        predictions_step = step_dataset.output_scaler.inverse_transform(predictions_step)
        targets_step = step_dataset.output_scaler.inverse_transform(targets_step)

        # Visualize
        visualize_predictions(targets_step, predictions_step,
                            title="Step Actuation - Model Predictions",
                            save_path=f"{results_dir}/step_predictions.png")
        visualize_training(train_losses_step, val_losses_step,
                         title="Step Model Training",
                         save_path=f"{results_dir}/step_training_curves.png")

    # 5. Save models
    print("\n[5] Saving models...")
    ensemble.save_models("./saved_models")

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
