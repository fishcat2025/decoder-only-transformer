# Decoder-Only Transformer Language Model

## Overview
This repository contains an implementation of a **decoder-only Transformer language model** for autoregressive text modeling.  
The project focuses on implementing the model architecture, data handling, and training pipeline, with support for **distributed training using PyTorch Lightning**.

The code was developed during **2024** as part of academic work and is published after cleanup and documentation.

---

## Model Description
The implemented model follows a **decoder-only Transformer architecture**, where the model learns to predict the next token in a sequence given all previous tokens.

Core components include:
- Token embeddings
- Positional embeddings
- Multi-head self-attention with **causal masking**
- Feed-forward layers
- Layer normalization
- Cross-entropy loss for next-token prediction

The model is trained in an **autoregressive** manner.

---

## Training Pipeline
Training is implemented in two main scripts:

- `decoder_only_transformer_distributed.py`  
  Contains the core model definition and training logic.

- `distributed_with_lightning_trainer.py`  
  Wraps the model using **PyTorch Lightning** to enable cleaner training loops and distributed execution.

Training features include:
- Mini-batch training
- Learning-rate scheduling
- Support for multi-device / distributed training
- Model checkpointing (checkpoints not included in this repository)

---

## Dataset and Tokenization
The training dataset is a plain text file located at:

