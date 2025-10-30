"""
Configuration file for MEMS LSTM models
Easily switch between sample data and full dataset
"""

# ============================================================================
# DATA PATHS
# ============================================================================

# Use sample data for quick testing (smaller dataset)
SAMPLE_DATA_DIR = "./sample"

# Use full COMSOL export data (your actual dataset)
FULL_DATA_DIR = r"D:\exportfiles"

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Simple LSTM Configuration
SIMPLE_LSTM_CONFIG = {
    'batch_size': 16,
    'sequence_length': 10,      # How many past time steps to use
    'hidden_size': 32,          # LSTM hidden dimension
    'num_layers': 1,            # Number of LSTM layers
    'output_size': 2,           # Predict [disp_mid, disp_3q]
    'feature_size': 3,          # [b_length, b_height, air_gap]
    'dropout': 0.1,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'device': 'cuda',           # 'cuda' for GPU, 'cpu' for CPU
}

# Train/Val/Test split
DATA_SPLIT = {
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# ============================================================================
# EXPERIMENTATION SETTINGS
# ============================================================================

# Try these configurations to understand their effect:

EXPERIMENTS = {
    'baseline': {
        'sequence_length': 10,
        'hidden_size': 32,
        'num_layers': 1,
        'learning_rate': 0.001,
        'num_epochs': 30,
    },
    'larger_model': {
        'sequence_length': 10,
        'hidden_size': 64,      # Double the hidden size
        'num_layers': 2,        # Add another layer
        'learning_rate': 0.001,
        'num_epochs': 50,
    },
    'longer_sequences': {
        'sequence_length': 30,  # Use more history
        'hidden_size': 32,
        'num_layers': 1,
        'learning_rate': 0.001,
        'num_epochs': 30,
    },
    'slower_learning': {
        'sequence_length': 10,
        'hidden_size': 32,
        'num_layers': 1,
        'learning_rate': 0.0001,  # 10x smaller
        'num_epochs': 100,        # Train longer
    },
}

# ============================================================================
# QUICK START GUIDE
# ============================================================================

"""
To use this config in simple_beam_lstm.py:

    from config import SIMPLE_LSTM_CONFIG, FULL_DATA_DIR, SAMPLE_DATA_DIR

    # Use full data
    config = SIMPLE_LSTM_CONFIG.copy()
    config['data_dir'] = FULL_DATA_DIR

    # Or use sample data for testing
    config['data_dir'] = SAMPLE_DATA_DIR

    # Then pass config values to main():
    main(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        # ... etc
    )

Experiment idea:
    1. Run baseline configuration
    2. Try larger_model configuration
    3. Compare validation loss to see which is better
    4. Fine-tune the best one
"""
