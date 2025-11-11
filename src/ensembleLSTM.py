"""
LSTM Ensemble for MEMS Beam Displacement Prediction

Separate models for wave (sinusoidal) and step voltage actuation.
Each uses a 2-layer LSTM (96 hidden units) with scaled dot-product attention.

Handles:
- Data loading from COMSOL simulations
- Preprocessing and normalization
- Training with validation monitoring
- Model persistence and inference

Italian: Sistema di ensemble LSTM per la previsione dello spostamento
di travi MEMS soggette ad attuazione elettrostatica periodica (wave) e a gradino (step).
"""

import glob
import os
import pickle
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset


class BeamDataset(Dataset):
    """Load COMSOL displacement data and create time-series samples."""

    def __init__(self, data_dir: str, actuation_type: str = 'both',
                 sequence_length: int = 30, stride: int = 10):
        """
        Args:
            data_dir: Directory with simulation data files
            actuation_type: 'wave', 'step', or 'both'
            sequence_length: Sliding window size
            stride: Window step size
        """
        self.data_dir = data_dir
        self.actuation_type = actuation_type
        self.sequence_length = sequence_length
        self.stride = stride

        # Data normalization
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        self.input_scaler = MinMaxScaler(feature_range=(0, 1))
        self.output_scaler = MinMaxScaler(feature_range=(0, 1))
        self.geom_scaler = StandardScaler()

        self.load_data()

    def load_data(self):
        """Load and preprocess COMSOL data files."""
        all_sequences = []
        all_targets = []
        all_geom_features = []
        all_voltage_info = []

        # Determine which files to load
        if self.actuation_type == 'wave':
            patterns = ['vertdisptime_*.txt']
            print(f"\nLoading WAVE actuation files...")
        elif self.actuation_type == 'step':
            patterns = ['stepvertdisptime_*.txt']
            print(f"\nLoading STEP actuation files...")
        else:
            patterns = ['vertdisptime_*.txt', 'stepvertdisptime_*.txt']
            print(f"\nLoading BOTH wave and step actuation files...")

        # Process each file pattern
        for pattern in patterns:
            files = glob.glob(os.path.join(self.data_dir, pattern))
            print(f"Found {len(files)} files for pattern: {pattern}")

            for file_path in files:
                try:
                    self._process_file(file_path, all_sequences, all_targets,
                                     all_geom_features, all_voltage_info)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    continue

        # Convert to numpy arrays
        if all_sequences:
            self.sequences = np.array(all_sequences)
            self.targets = np.array(all_targets)
            self.geom_features = np.array(all_geom_features)
            self.voltage_info = np.array(all_voltage_info) if all_voltage_info else None
            print(f"Loaded {len(self.sequences)} samples")

            self._normalize_data()
        else:
            print("WARNING: No data loaded!")
            self.sequences = np.empty((0, self.sequence_length, 2))
            self.targets = np.empty((0, 2))
            self.geom_features = np.empty((0, 3))
            self.voltage_info = None

    def _process_file(self, file_path: str, all_sequences, all_targets,
                     all_geom_features, all_voltage_info):
        """Process a single COMSOL output file."""
        data = pd.read_csv(file_path, sep=r'\s+', header=None)

        if data.shape[1] < 7:
            raise ValueError(f"File has {data.shape[1]} columns, need at least 7")

        # Extract geometry and voltage
        vbase = data.iloc[0, 0]
        beam_len = data.iloc[0, 1]
        beam_height = data.iloc[0, 2]
        air_gap = data.iloc[0, 3]

        # Extract time series
        time = data.iloc[:, 4].values
        disp_mid = data.iloc[:, 5].values
        disp_3q = data.iloc[:, 6].values

        # Use only first period (T_0 = 10 µs = 500 samples @ 50 MHz)
        samples_per_period = 500
        if len(time) < samples_per_period:
            raise ValueError(f"Not enough samples: {len(time)} < {samples_per_period}")

        time = time[:samples_per_period]
        disp_mid = disp_mid[:samples_per_period]
        disp_3q = disp_3q[:samples_per_period]

        # Create sliding windows
        for i in range(0, len(time) - self.sequence_length, self.stride):
            seq = np.column_stack([
                disp_mid[i:i+self.sequence_length],
                disp_3q[i:i+self.sequence_length]
            ])
            all_sequences.append(seq)

            if i + self.sequence_length < len(time):
                target = np.array([
                    disp_mid[i+self.sequence_length],
                    disp_3q[i+self.sequence_length]
                ])
                all_targets.append(target)
                all_geom_features.append([beam_len, beam_height, air_gap])
                all_voltage_info.append(vbase)

    def _normalize_data(self):
        """Normalize sequences, targets, and features."""
        # Reshape and normalize inputs
        shape = self.sequences.shape
        self.sequences = self.sequences.reshape(-1, shape[-1])
        self.sequences = self.input_scaler.fit_transform(self.sequences)
        self.sequences = self.sequences.reshape(shape)

        # Normalize targets
        self.targets = self.output_scaler.fit_transform(self.targets)

        # Normalize geometry
        self.geom_features = self.geom_scaler.fit_transform(self.geom_features)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor(self.targets[idx]),
                torch.FloatTensor(self.geom_features[idx]))


class AttentionLayer(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = np.sqrt(hidden_size)

    def forward(self, lstm_out):
        """Apply attention to LSTM outputs."""
        Q = self.query(lstm_out)
        K = self.key(lstm_out)
        V = self.value(lstm_out)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)

        # Apply to values, take last timestep
        context = torch.matmul(weights, V)[:, -1, :]
        return context, weights


class AttentionBeamLSTM(nn.Module):
    """LSTM with attention for beam displacement prediction."""

    def __init__(self, input_size: int = 2, hidden_size: int = 96,
                 num_layers: int = 2, output_size: int = 2,
                 feature_size: int = 3, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)

        self.attention = AttentionLayer(hidden_size)

        # Encode geometric features
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
        # LSTM pass
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)

        # Combine with features
        if features is not None:
            feat_encoded = self.feature_encoder(features)
            combined = torch.cat([context, feat_encoded], dim=1)
        else:
            combined = context

        return self.output_net(combined)


class EnsembleBeamPredictor:
    """Manage separate models for wave and step actuation."""

    def __init__(self, model_configs: Dict = None):
        self.models = {}
        self.model_configs = model_configs or {
            'wave': {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.2},
            'step': {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.2}
        }

    def create_models(self):
        """Create a model for each actuation type."""
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
            print(f"Created model for '{name}' actuation")

    def train_model(self, model_name: str, train_loader: DataLoader,
                   val_loader: DataLoader, epochs: int = 80,
                   learning_rate: float = 0.001, device: str = 'cpu'):
        """Train a single model."""
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == 'cuda':
            print(f"\n  GPU: {torch.cuda.get_device_name(device.index)}")
            mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"  GPU Memory: {mem_gb:.2f} GB")

        model = self.models[model_name].to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        scaler = GradScaler('cuda') if device.type == 'cuda' else None

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            n_batches = 0

            for seq, target, feat in train_loader:
                seq = seq.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                feat = feat.to(device, non_blocking=True)

                optimizer.zero_grad()

                if device.type == 'cuda':
                    with autocast('cuda'):
                        out = model(seq, feat)
                        loss = criterion(out, target)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(seq, feat)
                    loss = criterion(out, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                train_loss += loss.item()
                n_batches += 1

                if device.type == 'cuda' and n_batches % 10 == 0:
                    torch.cuda.empty_cache()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, target, feat in val_loader:
                    seq = seq.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    feat = feat.to(device, non_blocking=True)

                    if device.type == 'cuda':
                        with autocast('cuda'):
                            out = model(seq, feat)
                            loss = criterion(out, target)
                    else:
                        out = model(seq, feat)
                        loss = criterion(out, target)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            train_losses.append(avg_train)
            val_losses.append(avg_val)

            scheduler.step(avg_val)

            if (epoch + 1) % 10 == 0:
                gpu_info = ""
                if device.type == 'cuda':
                    mem = torch.cuda.memory_allocated(device) / 1e9
                    gpu_info = f" | GPU: {mem:.2f} GB"
                print(f"[{model_name}] Epoch {epoch+1}/{epochs} | "
                      f"Train: {avg_train:.6f} | Val: {avg_val:.6f}{gpu_info}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return train_losses, val_losses

    def predict(self, sequences: torch.Tensor, features: torch.Tensor,
               actuation_type: str = 'wave', device: str = 'cpu') -> np.ndarray:
        """Make predictions with the appropriate model."""
        if actuation_type not in self.models:
            print(f"Model '{actuation_type}' not found, using 'wave'")
            actuation_type = 'wave'

        if isinstance(device, str):
            device = torch.device(device)

        model = self.models[actuation_type]
        if next(model.parameters()).device != device:
            model = model.to(device)
            self.models[actuation_type] = model
        model.eval()

        # Convert to tensors if needed
        if isinstance(sequences, np.ndarray):
            sequences = torch.FloatTensor(sequences)
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)

        sequences = sequences.to(device)
        features = features.to(device)

        with torch.no_grad():
            predictions = model(sequences, features)

        result = predictions.cpu().detach().numpy()

        if device.type == 'cuda':
            del sequences, features, predictions
            torch.cuda.empty_cache()

        return result

    def save_models(self, save_dir: str):
        """Save all models."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': self.model_configs[name]
            }
            path = os.path.join(save_dir, f'{name}_model.pt')
            torch.save(checkpoint, path)
            print(f"Saved {name} model to {path}")

    def save_scalers(self, save_dir: str):
        """Save normalization scalers."""
        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self, '_input_scaler') and self._input_scaler:
            with open(os.path.join(save_dir, 'input_scaler.pkl'), 'wb') as f:
                pickle.dump(self._input_scaler, f)
            print(f"Saved input_scaler")

        if hasattr(self, '_output_scaler') and self._output_scaler:
            with open(os.path.join(save_dir, 'output_scaler.pkl'), 'wb') as f:
                pickle.dump(self._output_scaler, f)
            print(f"Saved output_scaler")

        if hasattr(self, '_geom_scaler') and self._geom_scaler:
            with open(os.path.join(save_dir, 'geom_scaler.pkl'), 'wb') as f:
                pickle.dump(self._geom_scaler, f)
            print(f"Saved geom_scaler")

    def load_models(self, save_dir: str):
        """Load saved models."""
        for name in self.model_configs.keys():
            path = os.path.join(save_dir, f'{name}_model.pt')
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu')
                model = AttentionBeamLSTM(
                    hidden_size=checkpoint['config']['hidden_size'],
                    num_layers=checkpoint['config']['num_layers'],
                    dropout=checkpoint['config'].get('dropout', 0.2)
                )
                model.load_state_dict(checkpoint['state_dict'])
                self.models[name] = model
                print(f"Loaded {name} model")
            else:
                print(f"Model file not found: {path}")


def plot_training(train_losses, val_losses, title, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].plot(train_losses, label='Training', linewidth=2.5)
    axes[0].plot(val_losses, label='Validation', linewidth=2.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'{title} - Linear', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Log scale
    axes[1].semilogy(train_losses, label='Training', linewidth=2.5)
    axes[1].semilogy(val_losses, label='Validation', linewidth=2.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[1].set_title(f'{title} - Log', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_predictions(true_vals, pred_vals, title="Predictions", save_path=None):
    """Plot actual vs predicted values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mid-span scatter
    axes[0, 0].scatter(true_vals[:, 0], pred_vals[:, 0], alpha=0.6, s=20)
    min_v, max_v = true_vals[:, 0].min(), true_vals[:, 0].max()
    axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('True (µm)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted (µm)', fontsize=11)
    axes[0, 0].set_title('Mid-span', fontsize=12)
    axes[0, 0].grid(alpha=0.3)

    # 3/4-span scatter
    axes[0, 1].scatter(true_vals[:, 1], pred_vals[:, 1], alpha=0.6, s=20, color='orange')
    min_v, max_v = true_vals[:, 1].min(), true_vals[:, 1].max()
    axes[0, 1].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('True (µm)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted (µm)', fontsize=11)
    axes[0, 1].set_title('3/4-span', fontsize=12)
    axes[0, 1].grid(alpha=0.3)

    # Error curves
    err_mid = np.abs(true_vals[:, 0] - pred_vals[:, 0])
    axes[1, 0].plot(err_mid, color='blue', alpha=0.7)
    axes[1, 0].fill_between(range(len(err_mid)), err_mid, alpha=0.3)
    axes[1, 0].set_ylabel('Error (µm)', fontsize=11)
    axes[1, 0].set_title('Mid-span Error', fontsize=12)
    axes[1, 0].grid(alpha=0.3)

    err_3q = np.abs(true_vals[:, 1] - pred_vals[:, 1])
    axes[1, 1].plot(err_3q, color='orange', alpha=0.7)
    axes[1, 1].fill_between(range(len(err_3q)), err_3q, alpha=0.3, color='orange')
    axes[1, 1].set_ylabel('Error (µm)', fontsize=11)
    axes[1, 1].set_title('3/4-span Error', fontsize=12)
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()

    # Print metrics
    mse_mid = np.mean((true_vals[:, 0] - pred_vals[:, 0])**2)
    mse_3q = np.mean((true_vals[:, 1] - pred_vals[:, 1])**2)
    mae_mid = np.mean(err_mid)
    mae_3q = np.mean(err_3q)

    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    print(f"Mid-span: MSE={mse_mid:.6e}, MAE={mae_mid:.6e}")
    print(f"3/4-span: MSE={mse_3q:.6e}, MAE={mae_3q:.6e}")
    print("="*50)


def main():
    """Train LSTM ensemble models."""
    print("\n" + "="*70)
    print("LSTM ENSEMBLE TRAINING - MEMS BEAM DISPLACEMENT")
    print("="*70)

    # Configuration
    data_dir = "D:/exportfiles"
    batch_size = 32
    epochs = 80
    learning_rate = 0.001
    seq_len = 30
    stride = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = "./results"

    print(f"\nDevice: {device}")

    # Load datasets
    print("\n[1] Loading wave data...")
    wave_data = BeamDataset(data_dir, actuation_type='wave',
                           sequence_length=seq_len, stride=stride)
    print(f"    Loaded {len(wave_data)} samples")

    print("\n[2] Loading step data...")
    step_data = BeamDataset(data_dir, actuation_type='step',
                           sequence_length=seq_len, stride=stride)
    print(f"    Loaded {len(step_data)} samples")

    if len(wave_data) == 0 and len(step_data) == 0:
        print("ERROR: No data loaded!")
        return

    # Create ensemble
    print("\n[3] Creating ensemble...")
    ensemble = EnsembleBeamPredictor()
    ensemble.create_models()
    ensemble._input_scaler = None
    ensemble._output_scaler = None
    ensemble._geom_scaler = None

    # Training
    print("\n[4] Training...")
    pin_mem = device.type == 'cuda'

    # Train wave model
    if len(wave_data) > 0:
        print("\n--- Wave Model ---")
        n = len(wave_data)
        train_idx, val_idx, test_idx = int(0.7*n), int(0.15*n), n - int(0.7*n) - int(0.15*n)

        from torch.utils.data import random_split
        train_w, val_w, test_w = random_split(wave_data, [train_idx, val_idx, test_idx])

        train_loader_w = DataLoader(train_w, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
        val_loader_w = DataLoader(val_w, batch_size=batch_size, pin_memory=pin_mem)
        test_loader_w = DataLoader(test_w, batch_size=batch_size, pin_memory=pin_mem)

        train_loss_w, val_loss_w = ensemble.train_model(
            'wave', train_loader_w, val_loader_w,
            epochs=epochs, learning_rate=learning_rate, device=device
        )

        # Test
        print("\nTesting wave model...")
        model = ensemble.models['wave'].to(device)
        model.eval()

        pred_w, true_w = [], []
        with torch.no_grad():
            for seq, target, feat in test_loader_w:
                seq = seq.to(device)
                feat = feat.to(device)
                out = model(seq, feat)
                pred_w.append(out.cpu().numpy())
                true_w.append(target.numpy())

        pred_w = np.vstack(pred_w)
        true_w = np.vstack(true_w)

        # Denormalize
        pred_w = wave_data.output_scaler.inverse_transform(pred_w)
        true_w = wave_data.output_scaler.inverse_transform(true_w)

        ensemble._input_scaler = wave_data.input_scaler
        ensemble._output_scaler = wave_data.output_scaler
        ensemble._geom_scaler = wave_data.geom_scaler

        plot_predictions(true_w, pred_w, "Wave - Predictions",
                        f"{results_dir}/wave_pred.png")
        plot_training(train_loss_w, val_loss_w, "Wave Training",
                     f"{results_dir}/wave_train.png")

    # Train step model
    if len(step_data) > 0:
        print("\n--- Step Model ---")
        n = len(step_data)
        train_idx, val_idx, test_idx = int(0.7*n), int(0.15*n), n - int(0.7*n) - int(0.15*n)

        from torch.utils.data import random_split
        train_s, val_s, test_s = random_split(step_data, [train_idx, val_idx, test_idx])

        train_loader_s = DataLoader(train_s, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
        val_loader_s = DataLoader(val_s, batch_size=batch_size, pin_memory=pin_mem)
        test_loader_s = DataLoader(test_s, batch_size=batch_size, pin_memory=pin_mem)

        train_loss_s, val_loss_s = ensemble.train_model(
            'step', train_loader_s, val_loader_s,
            epochs=epochs, learning_rate=learning_rate, device=device
        )

        # Test
        print("\nTesting step model...")
        model = ensemble.models['step'].to(device)
        model.eval()

        pred_s, true_s = [], []
        with torch.no_grad():
            for seq, target, feat in test_loader_s:
                seq = seq.to(device)
                feat = feat.to(device)
                out = model(seq, feat)
                pred_s.append(out.cpu().numpy())
                true_s.append(target.numpy())

        pred_s = np.vstack(pred_s)
        true_s = np.vstack(true_s)

        # Denormalize
        pred_s = step_data.output_scaler.inverse_transform(pred_s)
        true_s = step_data.output_scaler.inverse_transform(true_s)

        if ensemble._input_scaler is None:
            ensemble._input_scaler = step_data.input_scaler
            ensemble._output_scaler = step_data.output_scaler
            ensemble._geom_scaler = step_data.geom_scaler

        plot_predictions(true_s, pred_s, "Step - Predictions",
                        f"{results_dir}/step_pred.png")
        plot_training(train_loss_s, val_loss_s, "Step Training",
                     f"{results_dir}/step_train.png")

    # Save
    print("\n[5] Saving models...")
    ensemble.save_models("./saved_models")
    ensemble.save_scalers("./saved_models")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
