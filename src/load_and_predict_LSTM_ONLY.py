"""
MEMS Beam Displacement Prediction - Autoregressive LSTM Generation

Pure LSTM-based signal generation without synthetic waveforms.
Generates 10 microsecond (T_0) displacement signals iteratively from trained models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ensembleLSTM import EnsembleBeamPredictor


class BeamPredictionSystem:
    """Predict MEMS beam displacement using autoregressive LSTM generation."""

    def __init__(self, models_dir: str = "./saved_models"):
        self.models_dir = models_dir
        self.ensemble = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Scalers
        self.input_scaler = None
        self.output_scaler = None
        self.geom_scaler = None

        print(f"Initialized prediction system on {self.device}")

    def load_models(self) -> bool:
        """Load trained models and scalers from disk."""
        if not os.path.exists(self.models_dir):
            print(f"ERROR: Models directory not found: {self.models_dir}")
            return False

        print(f"\nLoading models from {self.models_dir}...")

        try:
            self.ensemble = EnsembleBeamPredictor()
            self.ensemble.load_models(self.models_dir)

            if not self.ensemble.models:
                print("ERROR: No models loaded!")
                return False

            print(f"Loaded {len(self.ensemble.models)} model(s)")

            self._load_scalers()
            return True

        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def _load_scalers(self):
        """Load normalization scalers."""
        try:
            # Input scaler
            input_path = os.path.join(self.models_dir, 'input_scaler.pkl')
            if os.path.exists(input_path):
                with open(input_path, 'rb') as f:
                    self.input_scaler = pickle.load(f)
                print("✓ Loaded input_scaler")
            else:
                print(f"✗ Missing input_scaler")

            # Output scaler (critical)
            output_path = os.path.join(self.models_dir, 'output_scaler.pkl')
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    self.output_scaler = pickle.load(f)
                print("✓ Loaded output_scaler")
            else:
                print(f"✗ Missing output_scaler - predictions will fail!")

            # Geometry scaler
            geom_path = os.path.join(self.models_dir, 'geom_scaler.pkl')
            if os.path.exists(geom_path):
                with open(geom_path, 'rb') as f:
                    self.geom_scaler = pickle.load(f)
                print("✓ Loaded geom_scaler")
            else:
                print(f"✗ Missing geom_scaler")

        except Exception as e:
            print(f"ERROR loading scalers: {e}")

    def get_user_input(self) -> dict:
        """Prompt for beam parameters."""
        print("\n" + "="*70)
        print("BEAM DISPLACEMENT PREDICTION - 10 MICROSECOND WINDOW")
        print("="*70)

        params = {}

        # Actuation type
        print("\n1. Actuation type:")
        print("   [1] Wave (Sinusoidal)")
        print("   [2] Step (Step voltage)")
        while True:
            choice = input("   Enter choice (1 or 2): ").strip()
            if choice == '1':
                params['voltage_type'] = 'wave'
                break
            elif choice == '2':
                params['voltage_type'] = 'step'
                break
            else:
                print("   Invalid. Enter 1 or 2.")

        # Voltage value
        print("\n2. Voltage (V):")
        print("   Range: 0.1 to 50 V")
        while True:
            try:
                voltage = float(input("   Value: "))
                if voltage < 0:
                    print("   Must be non-negative.")
                    continue
                params['voltage_value'] = voltage
                break
            except ValueError:
                print("   Enter a valid number.")

        # Beam geometry
        print("\n3. Beam parameters (micrometers):")
        print("   Training range: 20-100 µm length, 0.5-3 µm height/gap")

        while True:
            try:
                length = float(input("   Length (µm): "))
                if length <= 0:
                    print("   Must be positive.")
                    continue
                params['beam_length'] = length
                break
            except ValueError:
                print("   Invalid input.")

        while True:
            try:
                height = float(input("   Height (µm): "))
                if height <= 0:
                    print("   Must be positive.")
                    continue
                params['beam_height'] = height
                break
            except ValueError:
                print("   Invalid input.")

        while True:
            try:
                gap = float(input("   Air gap (µm): "))
                if gap <= 0:
                    print("   Must be positive.")
                    continue
                params['air_gap'] = gap
                break
            except ValueError:
                print("   Invalid input.")

        return params

    def predict_signal(self, params: dict) -> dict:
        """Generate displacement signal using autoregressive LSTM."""
        # Time domain (10 µs period at 50 MHz = 500 samples)
        T_0 = 1e-5
        sampling_rate = 50e6
        num_samples = int(T_0 * sampling_rate) + 1
        time = np.linspace(0, T_0, num_samples)

        print(f"\n{'='*70}")
        print("AUTOREGRESSIVE PREDICTION")
        print(f"{'='*70}")
        print(f"Time: {T_0*1e6:.1f} µs (one period)")
        print(f"Rate: {sampling_rate/1e6:.1f} MHz")
        print(f"Samples: {num_samples-1}")

        # Initialize outputs
        disp_mid = np.zeros(num_samples)
        disp_3q = np.zeros(num_samples)
        voltage_type = params['voltage_type']

        # Normalize geometry
        if self.geom_scaler is None:
            raise ValueError("Geometry scaler not loaded!")

        geom_raw = np.array([
            params['beam_length'],
            params['beam_height'],
            params['air_gap']
        ]).reshape(1, -1)
        geom_norm = self.geom_scaler.transform(geom_raw)[0]

        print(f"\nParameters:")
        print(f"  Type: {voltage_type.upper()}")
        print(f"  Voltage: {params['voltage_value']:.2f} V")
        print(f"  Beam: {params['beam_length']:.1f} × {params['beam_height']:.2f} µm")
        print(f"  Gap: {params['air_gap']:.2f} µm")

        # Get model
        seq_len = 30  # Must match training
        model = self.ensemble.models[voltage_type].to(self.device)
        model.eval()

        print(f"\nModel:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Device: {self.device}")
        print(f"\nGenerating...")

        # Autoregressive generation
        with torch.no_grad():
            for i in range(seq_len, num_samples):
                # Build sequence from last 30 samples
                seq = np.column_stack([
                    disp_mid[i-seq_len:i],
                    disp_3q[i-seq_len:i]
                ]).astype(np.float32)

                # Normalize input
                if self.input_scaler is None:
                    raise ValueError("Input scaler not loaded!")

                seq_shape = seq.shape
                seq_flat = seq.reshape(-1, seq_shape[-1])
                seq_norm = self.input_scaler.transform(seq_flat)
                seq_norm = seq_norm.reshape(seq_shape)

                # Predict
                seq_tensor = torch.FloatTensor(seq_norm).unsqueeze(0).to(self.device)
                feat_tensor = torch.FloatTensor(geom_norm).unsqueeze(0).to(self.device)
                pred_norm = model(seq_tensor, feat_tensor).cpu().numpy()

                # Denormalize
                if self.output_scaler is None:
                    raise ValueError("Output scaler not loaded!")

                pred = self.output_scaler.inverse_transform(pred_norm)

                # Update arrays
                disp_mid[i] = pred[0, 0]
                disp_3q[i] = pred[0, 1]

                # Progress
                if (i + 1) % 50 == 0:
                    pct = 100 * (i + 1) / num_samples
                    print(f"  {pct:5.1f}% | mid: {pred[0, 0]:+.4e} m | "
                          f"3/4: {pred[0, 1]:+.4e} m")

        print(f"\n✓ Done")
        print(f"\nStats:")
        print(f"  Mid-span: {disp_mid.min():.4e} to {disp_mid.max():.4e} m")
        print(f"            amplitude: {disp_mid.max() - disp_mid.min():.4e} m")
        print(f"  3/4-span: {disp_3q.min():.4e} to {disp_3q.max():.4e} m")
        print(f"            amplitude: {disp_3q.max() - disp_3q.min():.4e} m")
        print(f"{'='*70}\n")

        return {
            'time': time,
            'disp_mid': disp_mid,
            'disp_3q': disp_3q,
            'params': params
        }

    def visualize(self, results: dict, save_dir: str = "./results"):
        """Plot predicted displacements."""
        os.makedirs(save_dir, exist_ok=True)

        time = results['time']
        disp_mid = results['disp_mid']
        disp_3q = results['disp_3q']
        params = results['params']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        time_us = time * 1e6

        # Mid-span
        axes[0].plot(time_us, disp_mid, linewidth=2, color='blue', label='Mid-span (L/2)')
        axes[0].set_xlabel('Time (µs)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Displacement (m)', fontsize=12, fontweight='bold')
        axes[0].set_title('Mid-span - LSTM Generated (T_0)', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        axes[0].legend(fontsize=11)

        # 3/4-span
        axes[1].plot(time_us, disp_3q, linewidth=2, color='orange', label='3/4-span (3L/4)')
        axes[1].set_xlabel('Time (µs)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Displacement (m)', fontsize=12, fontweight='bold')
        axes[1].set_title('3/4-span - LSTM Generated (T_0)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        axes[1].legend(fontsize=11)

        # Info box
        info = (
            f"Parameters:\n"
            f"Type: {params['voltage_type'].upper()}\n"
            f"Voltage: {params['voltage_value']:.2f} V\n"
            f"Length: {params['beam_length']:.2f} µm\n"
            f"Height: {params['beam_height']:.2f} µm\n"
            f"Gap: {params['air_gap']:.2f} µm"
        )
        fig.text(0.99, 0.01, info, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('MEMS Beam - LSTM Autoregressive Prediction (T_0 = 10 µs)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.08, 1, 0.99])

        # Save
        path = os.path.join(save_dir, 'predicted_displacement.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {path}")
        plt.show()

    def export_data(self, results: dict, output_file: str = "./results/prediction.txt"):
        """Save signal to text file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        time = results['time']
        disp_mid = results['disp_mid']
        disp_3q = results['disp_3q']
        params = results['params']

        with open(output_file, 'w') as f:
            f.write("# MEMS BEAM DISPLACEMENT - LSTM AUTOREGRESSIVE\n")
            f.write(f"# Domain: 10 microseconds (T_0, one period)\n")
            f.write(f"# Type: {params['voltage_type']}\n")
            f.write(f"# Voltage: {params['voltage_value']:.4f} V\n")
            f.write(f"# Length: {params['beam_length']:.4f} µm\n")
            f.write(f"# Height: {params['beam_height']:.4f} µm\n")
            f.write(f"# Gap: {params['air_gap']:.4f} µm\n")
            f.write(f"# Sampling: 50 MHz, 500 samples\n")
            f.write(f"#\n")
            f.write("# Columns: Time(s) | Mid-span(m) | 3/4-span(m)\n")
            f.write("# " + "="*68 + "\n")

            for i in range(len(time)):
                f.write(f"{time[i]:.10e}\t{disp_mid[i]:.10e}\t{disp_3q[i]:.10e}\n")

        print(f"✓ Saved to {output_file}")

    def run(self):
        """Complete workflow."""
        if not self.load_models():
            print("\nERROR: Could not load models.")
            return

        params = self.get_user_input()
        results = self.predict_signal(params)

        self.visualize(results)
        self.export_data(results)

        print("\n" + "="*70)
        print("✓ PREDICTION COMPLETE")
        print("="*70)
        print("Results saved to ./results/")
        print("\nFiles:")
        print("  - predicted_displacement.png")
        print("  - prediction.txt")
        print("\nSignal: Pure LSTM autoregressive generation")
        print("="*70)


def main():
    """Run prediction system."""
    print("\n" + "="*70)
    print("BEAM DISPLACEMENT PREDICTION - AUTOREGRESSIVE LSTM")
    print("="*70)
    print("\nMethod:")
    print("  • Pure LSTM-based signal generation")
    print("  • Autoregressive iteration (one sample per step)")
    print("  • 10 microsecond period (500 samples at 50 MHz)")
    print("  • No synthetic signals")
    print("="*70)

    system = BeamPredictionSystem(models_dir="./saved_models")
    system.run()


if __name__ == "__main__":
    main()
