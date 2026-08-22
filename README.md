# LTP-ML
Lightweight TEE-Assisted Secure Multi-Party Learning Framework with Privileged Parties

This repository contains the code implementation for LTP-ML, a secure multi-party learning framework that integrates vector space secret sharing with lightweight trusted hardware (LTH) to achieve privileged control, malicious security, and efficient communication.

## Features

- **Hierarchical Privileged Architecture**: Supports multiple privileged parties with customized access structures; only privileged parties can recover the final model
- **LTH-Assisted Acceleration**: Offloads non-linear computations (ReLU, Sigmoid) to lightweight trusted hardware, reducing online communication rounds
- **Malicious Security**: Detects tampering via LTH-assisted share consistency verification with abort
- **Dropout Tolerance**: Supports up to `nd` assistant parties dropping out during training
- **Configurable**: Supports multiple parties, multiple privileged parties, and configurable security models

## Configure

**constant.json**

Configure the number of parties, training parameters (e.g., batch size), LTH settings, and machine learning models.

| Parameter | Description |
|-----------|-------------|
| `M` | Total number of parties |
| `np` | Number of privileged parties |
| `na` | Number of assistant parties ($na = M - np$) |
| `nd` | Number of assistant parties allowed to drop out ($nd < na$) |
| `lth_mode` | LTH type: `0` for LTH-chip (simulated), `1` for LTH-SoC |
| `prf_key` | Shared PRF key for LTH mask generation |
| `batch_size` | Training batch size |
| `epochs` | Number of training epochs |
| `model` | Model type: `linear`, `logistic`, `neural_network` |
| `dataset` | Dataset path |

**Example `constant.json`**:
```json
{
    "M": 3,
    "np": 1,
    "na": 2,
    "nd": 1,
    "lth_mode": 0,
    "prf_key": "0x0123456789abcdef",
    "batch_size": 128,
    "epochs": 20,
    "model": "neural_network",
    "dataset": "./data/mnist/"
}
