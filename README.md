# DL_Assignment3
# Hindi Transliteration Model

This project implements a sequence-to-sequence model with attention for transliterating Hindi text from Latin script to Devanagari script using the Dakshina dataset.

## Features

- Sequence-to-sequence model with attention mechanism
- Support for different RNN cell types (LSTM, GRU, RNN)
- Bidirectional encoder
- Teacher forcing during training
- Weights & Biases integration for experiment tracking
- Early stopping to prevent overfitting

## Setup

1. Install the required packages:
```bash
pip install torch pandas numpy wandb tqdm
```

2. Download the Dakshina dataset and extract it to the project directory:
```bash
wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar.gz
tar -xzf dakshina_dataset_v1.0.tar.gz
```

3. Make sure you have your Weights & Biases credentials set up. The project uses:
- Entity: rajunaik-iit-madras
- Project: My_Assignment-3

## Project Structure

- `transliteration_model.py`: Main model implementation with encoder-decoder architecture
- `data_loader.py`: Data loading and preprocessing utilities
- `README.md`: This file

## Training

To train the model, simply run:
```bash
python transliteration_model.py
```

The script will:
1. Load and preprocess the Dakshina dataset
2. Initialize the model with the specified configuration
3. Train the model while logging metrics to Weights & Biases
4. Save the best model based on validation loss
5. Evaluate on the test set
6. Save the final model with vocabularies

## Model Architecture

- Encoder:
  - Character embeddings
  - Bidirectional LSTM/GRU/RNN
  - Dropout for regularization

- Attention:
  - Additive attention mechanism
  - Attention masking for padded sequences

- Decoder:
  - Character embeddings
  - LSTM/GRU/RNN with attention
  - Dropout for regularization

## Hyperparameters

The default configuration includes:
- Embedding dimension: 256
- Hidden dimension: 512
- Number of layers: 2
- Dropout: 0.3
- Learning rate: 0.001
- Batch size: 64
- Maximum epochs: 30
- Early stopping patience: 5

You can modify these in the `config` dictionary in `transliteration_model.py`.

## Experiment Tracking

The training progress is tracked using Weights & Biases. You can view:
- Training and validation loss curves
- Model architecture summary
- Hyperparameter configurations
- Final test set performance
- Saved model artifacts

Visit your W&B project page to view the results: https://wandb.ai/
