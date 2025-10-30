"""
Simple LSTM Model for MEMS Beam Displacement Prediction
A beginner-friendly implementation to understand the fundamentals.

Data Structure:
- Each row: [actuation_param, b_length, b_height, air_gap, time, disp_mid, disp_3q]
- Goal: Predict next displacement (disp_mid, disp_3q) from historical sequence
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler


# ============================================================================
# 1. SIMPLE DATASET CLASS
# ============================================================================

class SimpleBeamDataset(Dataset):
    """
    Loads COMSOL beam displacement data and creates sequences for LSTM training.

    Key concept: We create sliding windows of time-series data.
    For example, if sequence_length=10, we use the last 10 displacement measurements
    to predict the next one.
    """

    def __init__(self, data_dir: str, sequence_length: int = 10, stride: int = 5):
        """
        Args:
            data_dir: Directory containing the .txt files
            sequence_length: Number of time steps to look back (window size)
            stride: Step size when sliding the window (smaller = more overlapping sequences)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride

        # Normalizers for input and output
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Storage for sequences, targets, and geometric features
        self.X_sequences = []      # Input sequences: (seq_len, 2) - [disp_mid, disp_3q]
        self.y_targets = []        # Target: next displacement (2,) - [disp_mid, disp_3q]
        self.geom_features = []    # Geometric params: (3,) - [b_length, b_height, air_gap]

        # Load data from files
        self._load_displacement_files()

        # Convert to numpy arrays
        self._prepare_arrays()

    def _load_displacement_files(self):
        """Load all displacement files from the directory"""

        # Find all vertical displacement time files
        pattern = os.path.join(self.data_dir, "*vertdisptime*.txt")
        files = glob.glob(pattern)

        print(f"Found {len(files)} displacement files")

        for file_path in files:
            try:
                print(f"  Loading: {os.path.basename(file_path)}")
                self._process_file(file_path)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue

    def _process_file(self, file_path: str):
        """
        Process a single displacement file.

        File format (whitespace-delimited):
        Column 0: actuation_param (e.g., voltage level)
        Column 1: beam_length
        Column 2: beam_height
        Column 3: air_gap
        Column 4: time
        Column 5: displacement_mid (at L/2)
        Column 6: displacement_3q (at 3L/4)
        """

        # Read the file
        data = pd.read_csv(file_path, sep='\s+', header=None)

        # Extract geometric parameters (same for all rows in this file)
        geom_params = data.iloc[0, 1:4].values  # [b_length, b_height, air_gap]

        # Extract time series data
        time = data.iloc[:, 4].values
        disp_mid = data.iloc[:, 5].values
        disp_3q = data.iloc[:, 6].values

        # Stack the two displacement measurements: (N, 2)
        displacement_data = np.column_stack([disp_mid, disp_3q])

        # Create sliding window sequences
        num_sequences = (len(time) - self.sequence_length) // self.stride

        for i in range(num_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length
            next_idx = end_idx + 1

            if next_idx < len(time):
                # Sequence of past displacements
                sequence = displacement_data[start_idx:end_idx]  # (seq_len, 2)

                # Target: next displacement values
                target = displacement_data[next_idx]  # (2,)

                self.X_sequences.append(sequence)
                self.y_targets.append(target)
                self.geom_features.append(geom_params)

    def _prepare_arrays(self):
        """Convert lists to numpy arrays and normalize"""

        if len(self.X_sequences) == 0:
            print("Warning: No sequences loaded!")
            return

        # Convert to numpy arrays
        self.X_sequences = np.array(self.X_sequences)      # (N_samples, seq_len, 2)
        self.y_targets = np.array(self.y_targets)          # (N_samples, 2)
        self.geom_features = np.array(self.geom_features)  # (N_samples, 3)

        print(f"Loaded {len(self.X_sequences)} sequences")

        # Normalize input sequences
        # Reshape for scaler: (N_samples * seq_len, 2)
        X_shape = self.X_sequences.shape
        X_flat = self.X_sequences.reshape(-1, X_shape[-1])
        X_flat_normalized = self.input_scaler.fit_transform(X_flat)
        self.X_sequences = X_flat_normalized.reshape(X_shape)

        # Normalize output targets
        self.y_targets = self.output_scaler.fit_transform(self.y_targets)

        # Normalize geometric features
        self.geom_features = (self.geom_features - self.geom_features.mean(axis=0)) / \
                            (self.geom_features.std(axis=0) + 1e-6)

    def __len__(self):
        """Return number of sequences"""
        return len(self.X_sequences)

    def __getitem__(self, idx):
        """Return a single sample as PyTorch tensors"""
        return (
            torch.FloatTensor(self.X_sequences[idx]),      # Input sequence
            torch.FloatTensor(self.y_targets[idx]),        # Target output
            torch.FloatTensor(self.geom_features[idx])     # Geometric features
        )


# ============================================================================
# 2. SIMPLE LSTM MODEL
# ============================================================================

class SimpleBeamLSTM(nn.Module):
    """
    Simple LSTM model for beam displacement prediction.

    Architecture:
    1. LSTM layer: Processes the sequence of displacement measurements
    2. Feature encoder: Processes geometric parameters
    3. Output layer: Combines LSTM output + features to predict next displacement

    Key idea:
    - LSTM learns temporal patterns in displacement
    - Geometric features help the model understand how parameters affect displacement
    - Final layers combine both to make the prediction
    """

    def __init__(
        self,
        input_size: int = 2,           # disp_mid, disp_3q
        hidden_size: int = 32,         # LSTM hidden dimension (try 32, 64, 128)
        num_layers: int = 1,           # Number of stacked LSTM layers
        output_size: int = 2,          # Predict disp_mid, disp_3q
        feature_size: int = 3,         # b_length, b_height, air_gap
        dropout: float = 0.1
    ):
        super(SimpleBeamLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer: Takes sequences of displacement measurements
        # input_size=2 (disp_mid, disp_3q)
        # hidden_size=32 (internal state dimension)
        # num_layers=1 (single layer for simplicity)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0
        )

        # Feature encoder: Process geometric parameters
        # Maps geometric features to a fixed-size representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Output network: Combine LSTM hidden state + encoded features
        # hidden_size (from LSTM) + 8 (from feature encoder) -> output_size
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size + 8, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input sequences, shape (batch_size, seq_len, input_size)
            features: Geometric features, shape (batch_size, feature_size)

        Returns:
            predictions: Shape (batch_size, output_size)
        """

        # LSTM processing
        # lstm_out: (batch, seq_len, hidden_size) - output at each time step
        # (h_n, c_n): final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take only the last hidden state
        # h_n shape: (num_layers, batch, hidden_size)
        # We want: (batch, hidden_size)
        last_hidden = h_n[-1]

        # Encode the geometric features
        encoded_features = self.feature_encoder(features)

        # Concatenate LSTM hidden state and encoded features
        combined = torch.cat([last_hidden, encoded_features], dim=1)

        # Generate final prediction
        predictions = self.output_head(combined)

        return predictions


# ============================================================================
# 3. TRAINING FUNCTION
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str
) -> float:
    """Train for one epoch and return average loss"""

    model.train()
    total_loss = 0.0

    for X_seq, y_target, features in train_loader:
        # Move data to device
        X_seq = X_seq.to(device)
        y_target = y_target.to(device)
        features = features.to(device)

        # Forward pass
        predictions = model(X_seq, features)
        loss = criterion(predictions, y_target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: str
) -> float:
    """Validate the model and return average loss"""

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_seq, y_target, features in val_loader:
            X_seq = X_seq.to(device)
            y_target = y_target.to(device)
            features = features.to(device)

            predictions = model(X_seq, features)
            loss = criterion(predictions, y_target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """Complete training loop"""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler: reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


# ============================================================================
# 4. TESTING & VISUALIZATION
# ============================================================================

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    dataset: SimpleBeamDataset,
    device: str = 'cpu'
):
    """Test the model and return predictions + targets (denormalized)"""

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_seq, y_target, features in test_loader:
            X_seq = X_seq.to(device)
            features = features.to(device)

            predictions = model(X_seq, features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_target.numpy())

    # Stack all batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Denormalize (convert back to original scale)
    predictions = dataset.output_scaler.inverse_transform(predictions)
    targets = dataset.output_scaler.inverse_transform(targets)

    return predictions, targets


def plot_results(predictions: np.ndarray, targets: np.ndarray, title: str = "Predictions"):
    """Visualize predictions vs targets"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Mid-span displacement: scatter plot
    axes[0, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.5)
    axes[0, 0].plot(
        [targets[:, 0].min(), targets[:, 0].max()],
        [targets[:, 0].min(), targets[:, 0].max()],
        'r--',
        label='Perfect prediction'
    )
    axes[0, 0].set_xlabel('True Displacement (mid)')
    axes[0, 0].set_ylabel('Predicted Displacement (mid)')
    axes[0, 0].set_title('Mid-span Prediction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 3/4-span displacement: scatter plot
    axes[0, 1].scatter(targets[:, 1], predictions[:, 1], alpha=0.5)
    axes[0, 1].plot(
        [targets[:, 1].min(), targets[:, 1].max()],
        [targets[:, 1].min(), targets[:, 1].max()],
        'r--',
        label='Perfect prediction'
    )
    axes[0, 1].set_xlabel('True Displacement (3/4)')
    axes[0, 1].set_ylabel('Predicted Displacement (3/4)')
    axes[0, 1].set_title('3/4-span Prediction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Errors over time
    errors_mid = np.abs(targets[:, 0] - predictions[:, 0])
    axes[1, 0].plot(errors_mid, label='Absolute Error')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Mid-span Prediction Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 3/4-span errors
    errors_3q = np.abs(targets[:, 1] - predictions[:, 1])
    axes[1, 1].plot(errors_3q, label='Absolute Error')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('3/4-span Prediction Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print metrics
    mse_mid = np.mean((targets[:, 0] - predictions[:, 0])**2)
    mse_3q = np.mean((targets[:, 1] - predictions[:, 1])**2)
    mae_mid = np.mean(errors_mid)
    mae_3q = np.mean(errors_3q)

    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"Mid-span Displacement:")
    print(f"  MSE: {mse_mid:.6e}")
    print(f"  MAE: {mae_mid:.6e}")
    print(f"\n3/4-span Displacement:")
    print(f"  MSE: {mse_3q:.6e}")
    print(f"  MAE: {mae_3q:.6e}")
    print("="*50)


# ============================================================================
# 5. MAIN: PUT IT ALL TOGETHER
# ============================================================================

def main():
    """Main function: load data, create model, train, test"""

    print("\n" + "="*70)
    print("SIMPLE LSTM FOR MEMS BEAM DISPLACEMENT PREDICTION")
    print("="*70)

    # Configuration
    DATA_DIR = r"D:\exportfiles"  # Point to your actual COMSOL export directory
    BATCH_SIZE = 16
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001

    # Device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # -----------------------------------------------------------------------
    # Step 1: Load and prepare data
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading dataset...")
    dataset = SimpleBeamDataset(DATA_DIR, sequence_length=SEQUENCE_LENGTH, stride=5)

    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return

    print(f"Total sequences loaded: {len(dataset)}")

    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------------------------
    # Step 2: Create model
    # -----------------------------------------------------------------------
    print("\n[Step 2] Creating model...")
    model = SimpleBeamLSTM(
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        output_size=2,
        feature_size=3,
        dropout=0.1
    ).to(device)

    # Print model architecture
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Step 3: Train
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] Training for {NUM_EPOCHS} epochs...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # -----------------------------------------------------------------------
    # Step 4: Test
    # -----------------------------------------------------------------------
    print("\n[Step 4] Testing on test set...")
    predictions, targets = test_model(model, test_loader, dataset, device)

    # -----------------------------------------------------------------------
    # Step 5: Visualize results
    # -----------------------------------------------------------------------
    print("\n[Step 5] Visualizing results...")
    plot_results(predictions, targets, title="Simple LSTM Model Results")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
