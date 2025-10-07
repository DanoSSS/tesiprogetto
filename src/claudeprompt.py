from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class MEMSDataset(Dataset):
    """Custom Dataset for MEMS displacement data"""

    def __init__(
        self,
        geometric_features: np.ndarray,
        voltage_sequences: np.ndarray,
        displacement_sequences: np.ndarray,
        actuation_types: np.ndarray,
        sequence_length: int = 1500,
    ):
        """
        Args:
            geometric_features: (N, 3) array of [length, height, air_gap]
            voltage_sequences: (N, sequence_length) array of voltage inputs
            displacement_sequences: (N, sequence_length, 2) array of displacements
                                  [:, :, 0] = mid-cross section
                                  [:, :, 1] = three-fourth cross section
            actuation_types: (N,) array of actuation types (0=step, 1=sine)
            sequence_length: Number of time steps (default 1500)
        """
        self.geometric_features = torch.FloatTensor(geometric_features)
        self.voltage_sequences = torch.FloatTensor(voltage_sequences)
        self.displacement_sequences = torch.FloatTensor(displacement_sequences)
        self.actuation_types = torch.FloatTensor(actuation_types)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.geometric_features)

    def __getitem__(self, idx):
        return {
            "geometric": self.geometric_features[idx],
            "voltage": self.voltage_sequences[idx],
            "displacement": self.displacement_sequences[idx],
            "actuation_type": self.actuation_types[idx],
        }


class MEMSLSTMModel(nn.Module):
    """LSTM model for MEMS displacement prediction"""

    def __init__(
        self,
        geometric_dim: int = 3,
        actuation_info_dim: int = 2,  # voltage + actuation_type
        hidden_dim: int = 256,
        lstm_layers: int = 3,
        dropout_rate: float = 0.2,
        output_dim: int = 2,
    ):  # 2 displacement outputs
        super(MEMSLSTMModel, self).__init__()

        # Geometric feature encoder
        self.geometric_encoder = nn.Sequential(
            nn.Linear(geometric_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # LSTM for processing temporal data
        # Input: voltage + actuation_type + encoded geometric features
        lstm_input_dim = actuation_info_dim + 64

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True,  # Bidirectional for better context
        )

        # Attention mechanism for focusing on important time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # Output layers for displacement prediction
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),  # 2 outputs: mid and 3/4 cross-section
        )

    def forward(self, geometric_features, voltage_sequence, actuation_type):
        batch_size, seq_len = voltage_sequence.shape

        # Encode geometric features
        geometric_encoded = self.geometric_encoder(geometric_features)

        # Expand geometric features to match sequence length
        geometric_expanded = geometric_encoded.unsqueeze(1).expand(-1, seq_len, -1)

        # Expand actuation type to match sequence length
        actuation_expanded = (
            actuation_type.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, 1)
        )

        # Expand voltage to add feature dimension
        voltage_expanded = voltage_sequence.unsqueeze(2)

        # Concatenate all inputs
        lstm_input = torch.cat(
            [voltage_expanded, actuation_expanded, geometric_expanded], dim=2
        )

        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # Apply attention mechanism (optional, can be commented out for simpler model)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        lstm_out_weighted = lstm_out * attention_weights

        # Predict displacement for each time step
        displacement_pred = self.output_network(lstm_out_weighted)

        return displacement_pred


class MEMSDataPreprocessor:
    """Preprocessor for MEMS data normalization and encoding"""

    def __init__(self):
        self.geometric_scaler = StandardScaler()
        self.voltage_scaler = StandardScaler()
        self.displacement_scaler = StandardScaler()
        self.actuation_encoder = LabelEncoder()

    def fit(self, data_dict: Dict):
        """Fit the preprocessors on training data"""
        # Fit geometric scaler
        self.geometric_scaler.fit(data_dict["geometric"])

        # Fit voltage scaler (flatten all sequences)
        voltage_flat = data_dict["voltage"].flatten().reshape(-1, 1)
        self.voltage_scaler.fit(voltage_flat)

        # Fit displacement scaler
        displacement_flat = data_dict["displacement"].reshape(-1, 2)
        self.displacement_scaler.fit(displacement_flat)

        # Fit actuation type encoder
        self.actuation_encoder.fit(data_dict["actuation_type"])

    def transform(self, data_dict: Dict) -> Dict:
        """Transform the data"""
        transformed = {}

        # Transform geometric features
        transformed["geometric"] = self.geometric_scaler.transform(
            data_dict["geometric"]
        )

        # Transform voltage sequences
        original_shape = data_dict["voltage"].shape
        voltage_flat = data_dict["voltage"].flatten().reshape(-1, 1)
        voltage_scaled = self.voltage_scaler.transform(voltage_flat)
        transformed["voltage"] = voltage_scaled.reshape(original_shape)

        # Transform displacement
        original_shape = data_dict["displacement"].shape
        displacement_flat = data_dict["displacement"].reshape(-1, 2)
        displacement_scaled = self.displacement_scaler.transform(displacement_flat)
        transformed["displacement"] = displacement_scaled.reshape(original_shape)

        # Encode actuation type
        transformed["actuation_type"] = self.actuation_encoder.transform(
            data_dict["actuation_type"]
        )

        return transformed

    def inverse_transform_displacement(
        self, displacement_scaled: np.ndarray
    ) -> np.ndarray:
        """Inverse transform displacement predictions"""
        original_shape = displacement_scaled.shape
        displacement_flat = displacement_scaled.reshape(-1, 2)
        displacement_original = self.displacement_scaler.inverse_transform(
            displacement_flat
        )
        return displacement_original.reshape(original_shape)


import numpy as np
import pandas as pd
import os  # Aggiungi questa importazione
from typing import List, Dict  # Aggiungi questa importazione


def load_comsol_data(folder_path: str) -> Dict:
    """
    Carica i dati di simulazione COMSOL da una cartella.

    I nomi dei file devono avere un formato specifico, ad esempio:
    'L<length>_H<height>_G<gap>_V<voltage>_<type>.txt'
    Esempio: 'L40_H1_G1_V50_step.txt'

    Args:
        folder_path: Il percorso della cartella contenente i file di dati.

    Returns:
        Un dizionario contenente gli array NumPy per ogni tipo di dato.
    """
    geometric_features = []
    voltage_amplitudes = []
    displacement_sequences = []
    actuation_types = []

    print(f"Caricamento dati dalla cartella: {folder_path}")

    # Itera su tutti i file nella cartella specificata
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".csv"):
            try:
                # --- 1. Estrai i parametri dal nome del file ---
                parts = filename.replace(".txt", "").replace(".csv", "").split("_")

                length = float(parts[0].replace("L", ""))
                height = float(parts[1].replace("H", ""))
                air_gap = float(parts[2].replace("G", ""))
                voltage = float(parts[3].replace("V", ""))
                act_type = parts[4]

                # --- 2. Aggiungi i dati alle liste ---
                geometric_features.append([length, height, air_gap])
                voltage_amplitudes.append(
                    [voltage]
                )  # Deve essere una lista per la forma (N, 1)
                actuation_types.append(act_type)

                # --- 3. Carica i dati della sequenza di spostamento ---
                file_path = os.path.join(folder_path, filename)
                # Supponiamo che il file abbia 2 colonne: 'mid_disp', 'three_fourth_disp'
                # e 1500 righe. Usa pd.read_csv che è molto robusto.
                # Se il tuo file non ha un'intestazione, usa header=None.
                df = pd.read_csv(
                    file_path, sep="\s+", header=None
                )  # '\s+' gestisce spazi e tab

                # Assicurati che abbia la forma corretta (1500, 2)
                if df.shape == (1500, 2):
                    displacement_sequences.append(df.values)
                else:
                    print(
                        f"Attenzione: il file {filename} ha una forma inattesa {df.shape}. Saltato."
                    )
                    # Rimuovi i dati aggiunti in precedenza per questo file non valido
                    geometric_features.pop()
                    voltage_amplitudes.pop()
                    actuation_types.pop()

            except (ValueError, IndexError) as e:
                print(
                    f"Errore nel parsing del nome del file: {filename}. Errore: {e}. Saltato."
                )

    print(f"Caricati {len(displacement_sequences)} campioni validi.")

    # --- 4. Converti le liste in array NumPy ---
    return {
        "geometric": np.array(geometric_features),
        "voltage": np.array(voltage_amplitudes),
        "displacement": np.array(displacement_sequences),
        "actuation_type": np.array(actuation_types),
    }


# Esempio di utilizzo nel blocco main


def create_data_loaders(
    data_dict: Dict,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""

    # Initialize preprocessor
    preprocessor = MEMSDataPreprocessor()

    # Split data
    n_samples = len(data_dict["geometric"])
    indices = np.arange(n_samples)

    # Stratified split based on actuation type
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        stratify=data_dict["actuation_type"],
        random_state=42,
    )

    val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        stratify=data_dict["actuation_type"][temp_idx],
        random_state=42,
    )

    # Create subset dictionaries
    def create_subset(data_dict, indices):
        return {
            "geometric": data_dict["geometric"][indices],
            "voltage": data_dict["voltage"][indices],
            "displacement": data_dict["displacement"][indices],
            "actuation_type": data_dict["actuation_type"][indices],
        }

    train_data = create_subset(data_dict, train_idx)
    val_data = create_subset(data_dict, val_idx)
    test_data = create_subset(data_dict, test_idx)

    # Fit preprocessor on training data only
    preprocessor.fit(train_data)

    # Transform all datasets
    train_data = preprocessor.transform(train_data)
    val_data = preprocessor.transform(val_data)
    test_data = preprocessor.transform(test_data)

    # Create datasets
    train_dataset = MEMSDataset(**train_data)
    val_dataset = MEMSDataset(**val_data)
    test_dataset = MEMSDataset(**test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, preprocessor


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Train the MEMS LSTM model"""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            geometric = batch["geometric"].to(device)
            voltage = batch["voltage"].to(device)
            displacement = batch["displacement"].to(device)
            actuation_type = batch["actuation_type"].to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(geometric, voltage, actuation_type)
            loss = criterion(predictions, displacement)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                geometric = batch["geometric"].to(device)
                voltage = batch["voltage"].to(device)
                displacement = batch["displacement"].to(device)
                actuation_type = batch["actuation_type"].to(device)

                predictions = model(geometric, voltage, actuation_type)
                loss = criterion(predictions, displacement)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mems_lstm_model.pth")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    preprocessor: MEMSDataPreprocessor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Evaluate the model on test data"""

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            geometric = batch["geometric"].to(device)
            voltage = batch["voltage"].to(device)
            displacement = batch["displacement"].to(device)
            actuation_type = batch["actuation_type"].to(device)

            predictions = model(geometric, voltage, actuation_type)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(displacement.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Inverse transform to original scale
    predictions_original = preprocessor.inverse_transform_displacement(all_predictions)
    targets_original = preprocessor.inverse_transform_displacement(all_targets)

    # Calculate metrics
    mse_mid = np.mean((predictions_original[:, :, 0] - targets_original[:, :, 0]) ** 2)
    mse_three_fourth = np.mean(
        (predictions_original[:, :, 1] - targets_original[:, :, 1]) ** 2
    )

    rmse_mid = np.sqrt(mse_mid)
    rmse_three_fourth = np.sqrt(mse_three_fourth)

    # R² score
    from sklearn.metrics import r2_score

    r2_mid = r2_score(
        targets_original[:, :, 0].flatten(), predictions_original[:, :, 0].flatten()
    )
    r2_three_fourth = r2_score(
        targets_original[:, :, 1].flatten(), predictions_original[:, :, 1].flatten()
    )

    return {
        "predictions": predictions_original,
        "targets": targets_original,
        "rmse_mid": rmse_mid,
        "rmse_three_fourth": rmse_three_fourth,
        "r2_mid": r2_mid,
        "r2_three_fourth": r2_three_fourth,
    }


def plot_results(results: Dict, sample_idx: int = 0):
    """Plot prediction results for a sample"""

    predictions = results["predictions"]
    targets = results["targets"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot mid cross-section
    axes[0].plot(targets[sample_idx, :, 0], label="True", alpha=0.7)
    axes[0].plot(predictions[sample_idx, :, 0], label="Predicted", alpha=0.7)
    axes[0].set_title("Mid Cross-Section Displacement")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Displacement")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot three-fourth cross-section
    axes[1].plot(targets[sample_idx, :, 1], label="True", alpha=0.7)
    axes[1].plot(predictions[sample_idx, :, 1], label="Predicted", alpha=0.7)
    axes[1].set_title("Three-Fourth Cross-Section Displacement")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Displacement")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nModel Performance:")
    print(
        f"Mid Cross-Section - RMSE: {results['rmse_mid']:.6f}, R²: {results['r2_mid']:.4f}"
    )
    print(
        f"3/4 Cross-Section - RMSE: {results['rmse_three_fourth']:.6f}, R²: {results['r2_three_fourth']:.4f}"
    )


if __name__ == "__main__":
    # --- SOSTITUISCI QUESTO BLOCCO CON IL CARICAMENTO DEI DATI REALI ---
    # n_samples = 1000
    # seq_length = 1500
    # data = {
    #     'geometric': np.random.rand(n_samples, 3) * np.array([30, 5, 3]) + np.array([40, 1, 1]),
    #     'voltage': np.random.rand(n_samples, 1) * 100,
    #     'displacement': np.random.randn(n_samples, seq_length, 2),
    #     'actuation_type': np.random.choice(['step', 'sine'], n_samples),
    # }
    # --------------------------------------------------------------------

    # --- USA QUESTO BLOCCO INVECE ---
    # Specifica il percorso della tua cartella di dati
    DATA_FOLDER = "percorso/alla/tua/cartella/dati"
    data = load_comsol_data(DATA_FOLDER)
    seq_length = data["displacement"].shape[
        1
    ]  # Ottieni la lunghezza della sequenza dai dati
    # --------------------------------

    # Il resto dello script rimane invariato...
    if len(data["geometric"]) > 0:  # Esegui solo se hai caricato dei dati
        train_loader, val_loader, test_loader, preprocessor = create_data_loaders(
            data, batch_size=32
        )

        model = MEMSLSTMModel(
            geometric_dim=3,
            hidden_dim=256,
            lstm_layers=3,
            dropout_rate=0.2,
            output_dim=2,
            sequence_length=seq_length,
        )

        print(
            f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=100,  # Puoi aumentare le epoche per i dati reali
        )

        results = evaluate_model(model, test_loader, preprocessor)
        plot_results(results, sample_idx=5)

# Example usage
if __name__ == "__main__":
    # Example data structure - replace with your actual data loading
    # Creating synthetic data for demonstration
    n_samples = 3000
    seq_length = 1500

    # Generate synthetic data (replace this with your actual COMSOL data)
    data = {
        "geometric": np.random.randn(n_samples, 3),  # [length, height, air_gap]
        "voltage": np.random.randn(n_samples, seq_length),  # Voltage sequences
        "displacement": np.random.randn(
            n_samples, seq_length, 2
        ),  # Two displacement outputs
        "actuation_type": np.random.choice(
            ["step", "sine"], n_samples
        ),  # Actuation types
    }

    # Create data loaders
    train_loader, val_loader, test_loader, preprocessor = create_data_loaders(
        data, batch_size=32, train_ratio=0.7, val_ratio=0.15
    )

    # Initialize model
    model = MEMSLSTMModel(
        geometric_dim=3,
        actuation_info_dim=2,
        hidden_dim=256,
        lstm_layers=3,
        dropout_rate=0.2,
        output_dim=2,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("Starting training...")
    training_history = train_model(
        model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3
    )

    # Load best model
    model.load_state_dict(torch.load("best_mems_lstm_model.pth"))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, preprocessor)

    # Plot results
    plot_results(results, sample_idx=0)
