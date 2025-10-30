"""
Sistema LSTM per previsione spostamento verticale di trave incastrata-incastrata
da simulazioni COMSOL con attuazione elettrica periodica e a gradino
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
warnings.filterwarnings('ignore')


class BeamDataset(Dataset):
    """Dataset personalizzato per i dati della trave COMSOL"""

    def __init__(self, data_dir: str, actuation_type: str = 'both', 
                 sequence_length: int = 50, stride: int = 10):
        """
        Args:
            data_dir: Directory contenente i file .txt
            actuation_type: 'wave', 'step', o 'both'
            sequence_length: Lunghezza delle sequenze temporali
            stride: Passo per creare sequenze sovrapposte
        """
        self.data_dir = data_dir
        self.actuation_type = actuation_type
        self.sequence_length = sequence_length
        self.stride = stride

        # Scaler per normalizzazione
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Carica i dati
        self.load_data()

    def load_data(self):
        """Carica e preprocessa i dati da file COMSOL"""
        all_sequences = []
        all_targets = []
        all_features = []

        # Pattern per i file in base al tipo di attuazione
        if self.actuation_type == 'wave':
            patterns = ['vertdisptime_*.txt', 'fftabsdisp_*.txt']
        elif self.actuation_type == 'step':
            patterns = ['stepvertdisptime_*.txt', 'stepfftabsdisp_*.txt']
        else:  # both
            patterns = ['*vertdisptime_*.txt', '*fftabsdisp_*.txt']

        # Carica file di spostamento nel tempo
        for pattern in patterns:
            if 'vertdisptime' in pattern:
                files = glob.glob(os.path.join(self.data_dir, pattern))

                for file in files:
                    try:
                        # Leggi il file (assumendo formato: tempo, disp_mezzo, disp_3quarti)
                        data = pd.read_csv(file, delim_whitespace=True, header=None)

                        # Estrai colonne rilevanti
                        if data.shape[1] >= 6:
                            # Assumendo struttura: param1, param2, param3, param4, tempo, disp1, disp2
                            time = data.iloc[:, 4].values
                            disp_mid = data.iloc[:, 5].values  # Spostamento a metà
                            disp_3q = data.iloc[:, 6].values if data.shape[1] > 6 else disp_mid

                            # Parametri geometrici (prime 4 colonne)
                            params = data.iloc[0, :4].values

                            # Crea sequenze temporali
                            for i in range(0, len(time) - self.sequence_length, self.stride):
                                seq_time = time[i:i+self.sequence_length]
                                seq_disp_mid = disp_mid[i:i+self.sequence_length]
                                seq_disp_3q = disp_3q[i:i+self.sequence_length]

                                # Combina gli spostamenti come input
                                sequence = np.column_stack([seq_time, seq_disp_mid, seq_disp_3q])
                                all_sequences.append(sequence)

                                # Target: prossimi valori di spostamento
                                if i + self.sequence_length < len(time):
                                    target = np.array([
                                        disp_mid[i+self.sequence_length],
                                        disp_3q[i+self.sequence_length]
                                    ])
                                    all_targets.append(target)
                                    all_features.append(params)

                    except Exception as e:
                        print(f"Errore nel caricamento di {file}: {e}")
                        continue

        # Carica dati ausiliari (pull-in voltage e frequenza eigenvalue)
        self.load_auxiliary_data()

        # Converti in array numpy
        self.sequences = np.array(all_sequences) if all_sequences else np.empty((0, self.sequence_length, 3))
        self.targets = np.array(all_targets) if all_targets else np.empty((0, 2))
        self.features = np.array(all_features) if all_features else np.empty((0, 4))

        # Normalizza i dati
        if len(self.sequences) > 0:
            seq_shape = self.sequences.shape
            self.sequences = self.sequences.reshape(-1, seq_shape[-1])
            self.sequences = self.input_scaler.fit_transform(self.sequences)
            self.sequences = self.sequences.reshape(seq_shape)

            self.targets = self.output_scaler.fit_transform(self.targets)

    def load_auxiliary_data(self):
        """Carica dati ausiliari come pull-in voltage e frequenze"""
        self.auxiliary_data = {}

        # Carica pull-in voltage
        pullin_files = glob.glob(os.path.join(self.data_dir, 'pullinvoltage_*.txt'))
        for file in pullin_files:
            try:
                data = pd.read_csv(file, delim_whitespace=True, header=None)
                # Estrai il valore di pull-in (ultima colonna)
                self.auxiliary_data['pullin_voltage'] = data.iloc[:, -1].values
            except:
                pass

        # Carica frequenze eigenvalue
        eigen_files = glob.glob(os.path.join(self.data_dir, 'eigenvaluefreq_*.txt'))
        for file in eigen_files:
            try:
                data = pd.read_csv(file, delim_whitespace=True, header=None)
                # Estrai frequenze (ultime colonne)
                self.auxiliary_data['eigenfreq'] = data.iloc[:, -1].values
            except:
                pass

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor(self.targets[idx]),
                torch.FloatTensor(self.features[idx]))


class BeamLSTM(nn.Module):
    """Modello LSTM per previsione spostamento trave"""

    def __init__(self, input_size: int = 3, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 2,
                 feature_size: int = 4, dropout: float = 0.2):
        """
        Args:
            input_size: Dimensione input (tempo, disp_mid, disp_3q)
            hidden_size: Dimensione stato nascosto LSTM
            num_layers: Numero di layer LSTM
            output_size: Dimensione output (2 spostamenti)
            feature_size: Dimensione features ausiliarie
            dropout: Dropout rate
        """
        super(BeamLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Layer per features ausiliarie
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Layer di output combinato
        self.output_fc = nn.Sequential(
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
        Forward pass
        Args:
            x: Input sequences (batch, seq_len, input_size)
            features: Features ausiliarie (batch, feature_size)
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Usa l'ultimo hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Processa features ausiliarie se disponibili
        if features is not None:
            feat_encoded = self.feature_fc(features)
            combined = torch.cat([last_hidden, feat_encoded], dim=1)
        else:
            combined = last_hidden

        # Output finale
        output = self.output_fc(combined)

        return output


class AttentionLSTM(nn.Module):
    """LSTM con meccanismo di attenzione per migliore performance"""

    def __init__(self, input_size: int = 3, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 2,
                 feature_size: int = 4, dropout: float = 0.2):
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x, features=None):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # Process features
        if features is not None:
            feat_encoded = self.feature_fc(features)
            combined = torch.cat([context_vector, feat_encoded], dim=1)
        else:
            combined = context_vector

        # Output
        output = self.output_fc(combined)

        return output


class EnsembleBeamPredictor:
    """Sistema ensemble che gestisce modelli per diverse condizioni"""

    def __init__(self, model_configs: Dict = None):
        """
        Args:
            model_configs: Configurazioni per i diversi modelli
        """
        self.models = {}
        self.scalers = {}
        self.model_configs = model_configs or {
            'wave': {'hidden_size': 128, 'num_layers': 2, 'use_attention': False},
            'step': {'hidden_size': 128, 'num_layers': 2, 'use_attention': False},
            'combined': {'hidden_size': 256, 'num_layers': 3, 'use_attention': True}
        }

    def create_models(self):
        """Crea i modelli per diverse condizioni"""
        for name, config in self.model_configs.items():
            if config.get('use_attention', False):
                model = AttentionLSTM(
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers']
                )
            else:
                model = BeamLSTM(
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers']
                )
            self.models[name] = model

    def train_model(self, model_name: str, train_loader: DataLoader, 
                    val_loader: DataLoader, epochs: int = 100,
                    learning_rate: float = 0.001, device: str = 'cpu'):
        """
        Addestra un singolo modello
        """
        model = self.models[model_name].to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
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

            # Validation
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

            if epoch % 10 == 0:
                print(f'Model: {model_name}, Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        return train_losses, val_losses

    def predict(self, sequences: torch.Tensor, features: torch.Tensor,
                actuation_type: str = 'combined', device: str = 'cpu'):
        """
        Effettua previsioni usando il modello appropriato
        """
        if actuation_type in self.models:
            model = self.models[actuation_type].to(device)
            model.eval()
            with torch.no_grad():
                predictions = model(sequences.to(device), features.to(device))
            return predictions.cpu().numpy()
        else:
            # Usa modello combinato come default
            return self.predict(sequences, features, 'combined', device)

    def save_models(self, save_dir: str):
        """Salva tutti i modelli"""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.model_configs[name]
            }, os.path.join(save_dir, f'{name}_model.pt'))

    def load_models(self, save_dir: str):
        """Carica modelli salvati"""
        for name in self.model_configs.keys():
            checkpoint = torch.load(os.path.join(save_dir, f'{name}_model.pt'))

            if self.model_configs[name].get('use_attention', False):
                model = AttentionLSTM(
                    hidden_size=checkpoint['model_config']['hidden_size'],
                    num_layers=checkpoint['model_config']['num_layers']
                )
            else:
                model = BeamLSTM(
                    hidden_size=checkpoint['model_config']['hidden_size'],
                    num_layers=checkpoint['model_config']['num_layers']
                )

            model.load_state_dict(checkpoint['model_state_dict'])
            self.models[name] = model


def visualize_predictions(true_values: np.ndarray, predictions: np.ndarray,
                          title: str = "Beam Displacement Predictions"):
    """Visualizza confronto tra valori veri e predetti"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Spostamento a metà lunghezza
    axes[0, 0].scatter(true_values[:, 0], predictions[:, 0], alpha=0.5)
    axes[0, 0].plot([true_values[:, 0].min(), true_values[:, 0].max()],
                    [true_values[:, 0].min(), true_values[:, 0].max()], 'r--')
    axes[0, 0].set_xlabel('True Displacement (Mid)')
    axes[0, 0].set_ylabel('Predicted Displacement (Mid)')
    axes[0, 0].set_title('Mid-span Displacement')

    # Spostamento a 3/4 lunghezza
    axes[0, 1].scatter(true_values[:, 1], predictions[:, 1], alpha=0.5)
    axes[0, 1].plot([true_values[:, 1].min(), true_values[:, 1].max()],
                    [true_values[:, 1].min(), true_values[:, 1].max()], 'r--')
    axes[0, 1].set_xlabel('True Displacement (3/4)')
    axes[0, 1].set_ylabel('Predicted Displacement (3/4)')
    axes[0, 1].set_title('3/4-span Displacement')

    # Errori nel tempo per mid-span
    errors_mid = np.abs(true_values[:, 0] - predictions[:, 0])
    axes[1, 0].plot(errors_mid)
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Mid-span Prediction Error')

    # Errori nel tempo per 3/4-span
    errors_3q = np.abs(true_values[:, 1] - predictions[:, 1])
    axes[1, 1].plot(errors_3q)
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('3/4-span Prediction Error')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # Calcola metriche
    mse_mid = np.mean((true_values[:, 0] - predictions[:, 0])**2)
    mse_3q = np.mean((true_values[:, 1] - predictions[:, 1])**2)
    mae_mid = np.mean(np.abs(true_values[:, 0] - predictions[:, 0]))
    mae_3q = np.mean(np.abs(true_values[:, 1] - predictions[:, 1]))

    print(f"\nMetrics:")
    print(f"Mid-span - MSE: {mse_mid:.6e}, MAE: {mae_mid:.6e}")
    print(f"3/4-span - MSE: {mse_3q:.6e}, MAE: {mae_3q:.6e}")


def main():
    """Funzione principale per training e test"""

    # Configurazione
    data_dir = "./comsol_data"  # Directory con i file .txt
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # 1. Carica dataset per onda periodica
    print("\n1. Loading wave actuation data...")
    wave_dataset = BeamDataset(data_dir, actuation_type='wave', 
                               sequence_length=50, stride=10)

    if len(wave_dataset) > 0:
        # Split train/val/test
        train_size = int(0.7 * len(wave_dataset))
        val_size = int(0.15 * len(wave_dataset))
        test_size = len(wave_dataset) - train_size - val_size

        train_wave, val_wave, test_wave = torch.utils.data.random_split(
            wave_dataset, [train_size, val_size, test_size]
        )

        train_loader_wave = DataLoader(train_wave, batch_size=batch_size, shuffle=True)
        val_loader_wave = DataLoader(val_wave, batch_size=batch_size)
        test_loader_wave = DataLoader(test_wave, batch_size=batch_size)

    # 2. Carica dataset per attuazione a gradino
    print("\n2. Loading step actuation data...")
    step_dataset = BeamDataset(data_dir, actuation_type='step',
                               sequence_length=50, stride=10)

    if len(step_dataset) > 0:
        train_size = int(0.7 * len(step_dataset))
        val_size = int(0.15 * len(step_dataset))
        test_size = len(step_dataset) - train_size - val_size

        train_step, val_step, test_step = torch.utils.data.random_split(
            step_dataset, [train_size, val_size, test_size]
        )

        train_loader_step = DataLoader(train_step, batch_size=batch_size, shuffle=True)
        val_loader_step = DataLoader(val_step, batch_size=batch_size)
        test_loader_step = DataLoader(test_step, batch_size=batch_size)

    # 3. Carica dataset combinato
    print("\n3. Loading combined data...")
    combined_dataset = BeamDataset(data_dir, actuation_type='both',
                                   sequence_length=50, stride=10)

    if len(combined_dataset) > 0:
        train_size = int(0.7 * len(combined_dataset))
        val_size = int(0.15 * len(combined_dataset))
        test_size = len(combined_dataset) - train_size - val_size

        train_combined, val_combined, test_combined = torch.utils.data.random_split(
            combined_dataset, [train_size, val_size, test_size]
        )

        train_loader_combined = DataLoader(train_combined, batch_size=batch_size, shuffle=True)
        val_loader_combined = DataLoader(val_combined, batch_size=batch_size)
        test_loader_combined = DataLoader(test_combined, batch_size=batch_size)

    # 4. Crea sistema ensemble
    print("\n4. Creating ensemble system...")
    ensemble = EnsembleBeamPredictor()
    ensemble.create_models()

    # 5. Training modelli
    print("\n5. Training models...")

    # Train wave model
    if len(wave_dataset) > 0:
        print("\n   Training wave actuation model...")
        train_losses_wave, val_losses_wave = ensemble.train_model(
            'wave', train_loader_wave, val_loader_wave, 
            epochs=epochs, learning_rate=learning_rate, device=device
        )

    # Train step model
    if len(step_dataset) > 0:
        print("\n   Training step actuation model...")
        train_losses_step, val_losses_step = ensemble.train_model(
            'step', train_loader_step, val_loader_step,
            epochs=epochs, learning_rate=learning_rate, device=device
        )

    # Train combined model
    if len(combined_dataset) > 0:
        print("\n   Training combined model...")
        train_losses_combined, val_losses_combined = ensemble.train_model(
            'combined', train_loader_combined, val_loader_combined,
            epochs=epochs, learning_rate=learning_rate, device=device
        )

    # 6. Test e visualizzazione
    print("\n6. Testing models...")

    # Test combined model
    if len(combined_dataset) > 0:
        model = ensemble.models['combined'].to(device)
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets, features in test_loader_combined:
                sequences = sequences.to(device)
                features = features.to(device)

                predictions = model(sequences, features)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        # Denormalizza i risultati
        predictions = combined_dataset.output_scaler.inverse_transform(predictions)
        targets = combined_dataset.output_scaler.inverse_transform(targets)

        # Visualizza risultati
        visualize_predictions(targets, predictions, "Combined Model Predictions")

    # 7. Salva modelli
    print("\n7. Saving models...")
    ensemble.save_models("./saved_models")
    print("Models saved successfully!")

    # Plot training curves
    if len(combined_dataset) > 0:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses_combined, label='Train Loss')
        plt.plot(val_losses_combined, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Combined Model Training')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Esempio di utilizzo
    print("=" * 60)
    print("LSTM System for COMSOL Beam Vertical Displacement Prediction")
    print("=" * 60)

    # Per eseguire il training completo, decommentare:
    # main()

    # Esempio di utilizzo con dati simulati
    print("\nCreating example with simulated data...")

    # Crea modello singolo per dimostrazione
    model = AttentionLSTM(input_size=3, hidden_size=128, num_layers=2, 
                          output_size=2, feature_size=4)

    # Dati simulati
    batch_size = 16
    seq_length = 50

    # Input: (batch, seq_len, features) - tempo, disp_mid, disp_3q
    x = torch.randn(batch_size, seq_length, 3)
    # Features ausiliarie: parametri geometrici
    features = torch.randn(batch_size, 4)

    # Forward pass
    output = model(x, features)
    print(f"\nInput shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first 3 samples):\n{output[:3].detach().numpy()}")

    print("\n" + "=" * 60)
    print("Sistema completato con successo!")
    print("Per usare con dati reali:")
    print("1. Metti i file .txt nella cartella './comsol_data'")
    print("2. Esegui main() per il training completo")
    print("3. I modelli saranno salvati in './saved_models'")
    print("=" * 60)
