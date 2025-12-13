# Inference-Induced Feedback: When ML Shapes Outcomes

## Overview

This project demonstrates a fundamental but often overlooked phenomenon in machine learning: **inference can influence outcomes without new data being collected**. Just like observing particles in the double-slit experiment changes their behavior, a model's predictions can subtly reshape the system it observes through feedback loops.

We simulate this phenomenon using synthetic data, a simple neural network, and a feedback loop, and quantify the effect using **mutual information (MI)** between the model's latent features and hidden attributes.

---

## Key Concepts

- **Hidden Attribute (`H`)**: A binary latent factor influencing observable data.
- **Latent Features (`z`)**: Model-learned representations capturing hidden attributes.
- **Feedback Loop**: The model's predictions slightly modify future inputs, simulating inference affecting reality.
- **Inference-Induced Collapse (IIC)**: Increase in MI between latent features and hidden attributes due to feedback.
- **Statistical Significance**: A t-test confirms whether feedback leads to meaningful changes in MI.

---

## Features

- Synthetic data generation with controllable hidden attributes
- Neural encoder to infer latent representations
- Mutual information estimation to quantify model knowledge
- Feedback loop to simulate inference influencing future data
- Statistical test for significance of the feedback effect
- Visualizations:
  - MI evolution over feedback steps
  - Boxplot comparing hidden attributes with latent features
  - Scatter plot of latent space colored by hidden attribute

---

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using:

```bash
pip install torch numpy scikit-learn matplotlib seaborn
