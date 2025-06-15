# PyTorch Transformer Implementation: Attention Is All You Need

This repository provides a comprehensive PyTorch implementation of the groundbreaking Transformer architecture introduced in the paper [**"Attention Is All You Need"**](https://arxiv.org/abs/1706.03762). The implementation includes the full encoder-decoder structure with multi-head attention mechanisms, positional encoding, and residual connections.

## 🚀 Features
- Complete Transformer model implementation
- Training pipeline with configurable hyperparameters
- Customizable dataset handling
- PyTorch-based implementation

## ⚙️ Requirements
| Package       | Minimum Version |
|---------------|-----------------|
| Python        | 3.96            |
| PyTorch       | 1.12.0          |
| TorchText     | 0.13.0          |
| NumPy         | 1.26.4          |
| [UV](https://github.com/astral-sh/uv) | Latest          |

## 📂 Project Structure
```bash
.
├── config/           # Configuration files (hyperparameters, paths)
│   └── config.yaml   # The yaml file for configuration
├── datasets/         # Dataset handling
│   ├── dataset       # The txt file for dataset
│   ├── dataset.py    # Dataloader class for training or interence
│   └── util.py       # Read data from txt file
├── models/           # Transformer implementation
│   ├── embedding/    # Tolen embedding and positional embedding
│   ├── layers/       # Encoder and Decoder
│   ├── model/        # Model architecture
│   └── utils/        # Utils for model, such as masking
├── train.py          # Main training script
├── infer.py          # Main interence script
├── requirements.txt  # This dependencies
└── README.md         # This documentation
```

## 🚀 Quick Start

### Clone the repository
```bash
git clone git@github.com:fanfan-yu/transformer.git
cd transformer
```

### Install UV package manager
```bash
pip install uv
```

### Download the dependencies
```bash
# Initialize environment and install dependencies
uv init
uv sync
```

### Run the Project
```bash
# Start training the Transformer model
uv run train.py
```

## 🔧 Configuration
The `config/config.yaml` file contains all configurable parameters:
```yaml
path:
  sentences_path: ./datasets/dataset/sentences.txt
  src_vocab_path: ./datasets/dataset/src_vocab.txt
  tgt_vocab_path: ./datasets/dataset/tgt_vocab.txt
# model parameters
model:
  d_model: 512
  n_head: 8
  d_key: 64
  d_value: 64
  d_feedforward: 2048
  max_len: 5000
  num_encoder_layers: 6
  num_decoder_layers: 6
# train parameters
train:
  batch_size: 2
  epoch: 20
  learning_rate: 0.001
  dropout: 0.1
```
## 📊 Todo List
- [ ] Implement inference
- [ ] Add support for external datasets (WMT, WikiText)
- [ ] Create Jupyter Notebook tutorials for beginners
- [ ] Optimize training for GPU environments

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (git checkout -b feat/your-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a pull request

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
