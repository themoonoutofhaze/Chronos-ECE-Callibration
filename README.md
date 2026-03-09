# Chronos-ECE-Calibration: Financial Time Series Forecasting & Calibration

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![Domain](https://img.shields.io/badge/Domain-Financial_Forecasting-success)

## 📌 Overview
**Chronos** is a powerful transformer model based on the T5 architecture, demonstrating strong predictive capabilities for time series data. This repository extends the base Chronos model with two major enhancements tailored for **domain-specific financial forecasting** and **reliable probability estimation**.

By combining domain-specific fine-tuning with a novel application of perturbation-based consistency calibration, this project bridges the gap between raw predictive power and trustworthy confidence estimation in Large Language Models (LLMs) used for time series.

## ✨ Key Contributions

### 1. Domain-Specific Fine-Tuning (Financial Modeling)
To adapt Chronos for financial forecasting, we fine-tuned the model on a custom hybrid dataset containing:
* **Real stock market data.**
* **Synthetic time series data** generated using Geometric Brownian Motion (GBM).
* **Result:** Achieved up to a **15% improvement** in forecasting accuracy, measured by the Mean Absolute Scaled Error (MASE).

### 2. Perturbation-Based Consistency Calibration (C3)
Accurate point predictions are not enough for financial models; reliable probability estimates are equally critical. To address this, we applied a **perturbation-based consistency calibration method (C3)**—a novel application in this specific context.
* **Mechanism:** We apply controlled perturbations directly to the logits and aggregate multiple perturbed predictions.
* **Result:** Significantly improved the Expected Calibration Error (ECE), resulting in highly stable and reliable confidence estimates.

## 📊 Results & Metrics
* **Accuracy:** Up to 15% improvement in **MASE** compared to the baseline Chronos model.
* **Reliability:** Lower **ECE** scores, demonstrating that the model's confidence levels strictly align with the actual probability of the forecasted outcomes.

## ⚙️ Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* HuggingFace `transformers`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/themoonoutofhaze/Chronos-ECE-Callibration.git
   cd Chronos-ECE-Callibration
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
*(Add your specific script instructions here)*

**To run the fine-tuning script with the GBM synthetic data:**
```bash
python scripts/fine_tune.py --config configs/financial_config.yaml
```

**To apply C3 Calibration and evaluate ECE:**
```bash
python scripts/calibrate.py --model_path checkpoints/best_model --apply_c3
```

## 📖 Citation & Index Terms
**Keywords:** Time series forecasting, Chronos, fine-tuning, consistency calibration, perturbation, financial modeling, large language models (LLMs).

