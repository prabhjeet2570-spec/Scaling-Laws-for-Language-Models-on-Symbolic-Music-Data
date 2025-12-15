# Scaling Laws for Music Language Models

A comprehensive study investigating whether neural scaling laws, originally discovered for text language models, apply to music modeling. This project trains decoder-only transformers and LSTMs of varying sizes on ABC notation and analyzes how validation loss scales with model size.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Setup and Reproduction](#setup-and-reproduction)
- [Generated Samples](#generated-samples)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project explores a fundamental question: **Do scaling laws hold for music the same way they hold for language?**

We investigate this by:
1. Converting 116,000+ MIDI files to ABC notation
2. Training transformer models ranging from 3.7M to 157M parameters
3. Training LSTM models with comparable parameter counts
4. Fitting power laws to measure scaling behavior
5. Generating and evaluating music samples from the best model

### Key Findings

| Architecture | Scaling Exponent (α) | Best Val Loss |
|--------------|---------------------|---------------|
| Transformer  | 0.8717              | 0.3751        |
| LSTM         | 0.8813              | 0.5656        |

**Surprising result:** Our scaling exponents (~0.87) are roughly 10x higher than those reported for text language models (α ≈ 0.076 by Kaplan et al.). This suggests music benefits more steeply from increased model capacity than text does.

---

## Project Structure
```
├── notebooks/
│   ├── Part_1.ipynb          # Data collection and preprocessing
│   ├── Part_2.ipynb          # Transformer scaling study
│   ├── Part_3.ipynb          # LSTM scaling study and comparison
│   └── Part_4.ipynb          # Best model training and generation
├── data/
│   ├── tokenizer.json        # Token to index mapping
│   ├── statistics.json       # Dataset statistics
├── results/
│   ├── results_tiny.json     # Tiny transformer results
│   ├── results_small.json    # Small transformer results
│   ├── results_medium.json   # Medium transformer results
│   ├── results_large.json    # Large transformer results
│   ├── results_xl.json       # XL transformer results
│   ├── lstm_results_*.json   # LSTM results
│   ├── best_model_results.json  # Generation results
│   ├── scaling_plot.png      # Transformer scaling plot
│   ├── lstm_scaling_plot.png # LSTM scaling plot
│   ├── comparison_scaling_plot.png  # Combined comparison
│   ├── training_curves.png   # Transformer training curves
│   ├── lstm_training_curves.png     # LSTM training curves
│   └── sample_*.abc / *.mid  # Generated samples
├── report/
│   └── report.pdf            # Final project report
└── README.md
```

The data and results directories in the repository contain all the plots and JSON files generated during the project. These directories do not include the original dataset files or the trained machine learning model files. Those files were stored in Google Drive during development and because of their very large size, it was not feasible to upload them to GitHub.

To reproduce the full project pipeline, you only need the file lmd-dataset.zip placed inside the MLProject/data/ directory in your Google Drive. This file can be downloaded from the following Kaggle link:
https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean

Note that this is not the original source of the dataset. The original Lakh MIDI Dataset is hosted at:
https://colinraffel.com/projects/lmd/

However, I was unable to download the dataset directly from the original source, so I used the Kaggle-hosted version instead.
---

## Dataset

### Source
- **Lakh MIDI Dataset Clean (LMD)**: 116,189 MIDI files (https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean)  
- **Original source of the dataset**: https://colinraffel.com/projects/lmd/
- Converted to ABC notation using `midi2abc`
- 99.23% conversion success rate (115,296 files)

### Statistics

| Split | Sequences | Tokens |
|-------|-----------|--------|
| Train | 257,964   | 1,167,894,118 |
| Val   | 2,632     | 11,907,471 |
| Test  | 2,633     | 11,808,149 |
| **Total** | **263,229** | **1,191,609,738** |

## Model Architectures

### Transformer Models

| Model  | Layers | d_model | Heads | d_ff  | Parameters |
|--------|--------|---------|-------|-------|------------|
| Tiny   | 4      | 64      | 4     | 256   | 3.7M       |
| Small  | 6      | 128     | 4     | 512   | 8.2M       |
| Medium | 8      | 256     | 8     | 1024  | 20.3M      |
| Large  | 8      | 512     | 8     | 2048  | 53.2M      |
| XL     | 8      | 1024    | 16    | 4096  | 156.8M     |

### LSTM Models

| Model  | Hidden Size | Layers | Parameters |
|--------|-------------|--------|------------|
| Tiny   | 64          | 1      | 3.5M       |
| Small  | 128         | 2      | 7.3M       |
| Medium | 320         | 3      | 19.9M      |
| Large  | 680         | 4      | 51.9M      |

## Results

### Scaling Study

All models trained on **100M tokens** with identical hyperparameters:
- Context length: 256 tokens
- Batch size: 4,096 tokens
- Optimizer: AdamW (weight decay 0.1)
- Learning rate: 3e-4 with cosine decay
- Fixed random indices for fair comparison

### Transformer Results

| Model  | Parameters | Train Loss | Val Loss | Time (min) | GPU (GB) |
|--------|------------|------------|----------|------------|----------|
| Tiny   | 3.7M       | 0.9264     | 0.8684   | 5.7        | 2.08     |
| Small  | 8.2M       | 0.6485     | 0.6150   | 8.7        | 2.36     |
| Medium | 20.3M      | 0.4902     | 0.4662   | 16.0       | 3.25     |
| Large  | 53.2M      | 0.4205     | 0.4021   | 32.5       | 4.24     |
| XL     | 156.8M     | 0.4073     | 0.3751   | 88.5       | 7.50     |

### LSTM Results

| Model  | Parameters | Train Loss | Val Loss | Time (min) | GPU (GB) |
|--------|------------|------------|----------|------------|----------|
| Tiny   | 3.5M       | 1.0250     | 1.0413   | 10.3       | 1.86     |
| Small  | 7.3M       | 0.8085     | 0.7972   | 12.8       | 1.92     |
| Medium | 19.9M      | 0.6850     | 0.6251   | 26.5       | 2.18     |
| Large  | 51.9M      | 0.5776     | 0.5656   | 57.0       | 2.80     |

### Best Model (XL Transformer, 200M tokens)

| Metric | Value |
|--------|-------|
| Parameters | 156.8M |
| Test Loss | 0.3495 |
| Test Perplexity | 1.42 |
---

## Setup and Reproduction

### Why Google Colab?

This project was developed entirely in **Google Colab** because:

1. **GPU Access**: My laptop is a Mac, so no NVIDIA GPUs available for training.

2. **Cost**: Other cloud GPU options were expensive. I purchased Google Colab Pro twice for a total of about $22, which provided sufficient A100 GPU access to complete all experiments for this project.

### Running the Notebooks

To reproduce the results:

1. **Create folder in Google Drive:**
```
   /content/drive/MyDrive/MLProject/data/
```

2. **Upload the MIDI dataset:**
   - Place `lmd-dataset.zip` in the `data/` folder

3. **Open notebooks in Colab:**
   - Upload notebooks to Colab or open directly from Drive
   - Enable GPU runtime: `Runtime > Change runtime type > A100 GPU`

4. **Run in order, cell by cell:**
```
   Part_1.ipynb → Part_2.ipynb → Part_3.ipynb → Part_4.ipynb
```

Part 1 will extract the MIDI files, convert them to ABC notation, tokenize, and create train/val/test splits. Parts 2-4 will use the processed data.

### Requirements

All dependencies are installed within the notebooks. Main libraries:
- PyTorch
- NumPy
- midi2abc
- music21 (for MIDI conversion)
- matplotlib
- scipy

---

## Generated Samples

The best model generates ABC notation that can be played using online players.

### Example Output
```abc
X:1
M:3/4
L:1/8
K:G
DGBdBG^c|GBdBGB^c|G^cgcGc^f|dAd^gdG^c|
GBd4z|D3^F^cdf|d2edd^c|z8|d6-d/2z3/2|^CDADADB|^cAdADDB|
^cA-[e-A]2e/2Bz^dGB|[d^F-]3[^c-F]c2-c/2z3/2|E^FAFAFB|^c8|z3ABD2z|
dddd^fdfe|z8|d3^F^cdf|d3A^dAd|^cAdADDB|
```

### How to Play

1. Go to [https://abcjs.net/abcjs-editor.html](https://abcjs.net/abcjs-editor.html)
2. Paste the ABC notation
3. Click play to hear the generated music

---

## Acknowledgments

- **Dataset**: [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
- **Kaggle Hosted Dataset**: https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean
- **ABC Tools**: `midi2abc` from abcmidi package
- **Scaling Laws Reference**: Kaplan et al. "Scaling Laws for Neural Language Models" - [https://arxiv.org/pdf/2001.08361](https://arxiv.org/pdf/2001.08361)
- **Compute**: Google Colab

I would like to thank **Professor Pavel Izmailov** for teaching the course and guiding this project. I learned a lot from working on this project.

---

<p align="center">
  <i>Music is what feelings sound like.</i>
</p>
