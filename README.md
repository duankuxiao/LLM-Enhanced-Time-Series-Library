<p align="center">
<h1 align="center">LLMformer: Probabilistic Forecasting with Large Language Models</h1>

<p align="center">
  This repository contains the official implementation of the paper "LLMformer: Hybrid LLM-Transformer Model for Probabilistic Time Series Forecasting".
</p>

## Overview

LLMformer is a hybrid architecture that combines pre-trained Large Language Models (LLMs) with Transformer encoders for probabilistic time series forecasting and imputation. The model outputs distribution parameters (mean and standard deviation) for interval predictions, supporting both Gaussian and Negative Binomial likelihood functions.

## Key Features

- **Probabilistic Output**: Predicts distribution parameters (mu, sigma) for uncertainty quantification
- **Dual Encoder**: LLM encoder for target variables + Transformer encoder for covariates
- **Multiple LLM Backends**: GPT-2, BERT, LLAMA (1b/3b/8b), Qwen
- **Likelihood Functions**: Gaussian and Negative Binomial distributions
- **Imputation Support**: Handles missing values in time series data
![Background](figures/LLMformer/workflow.png)
- 
## Architecture

```
Input → Normalization → LLM Encoder → Data Embedding
      → Transformer Encoder → Likelihood Layer → (mu, sigma) → Probabilistic Output
```
![Framework of LLMformer](figures/LLMformer/LLMformer.png)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers (Hugging Face)
- NumPy, Pandas, Scikit-learn

## Installation

```bash
pip install torch transformers numpy pandas scikit-learn matplotlib
```

## Usage

### Training

```bash
python run_probabilistic_forecast.py
```

### Configuration

Key parameters in `configs/HVAC_configs.py`:
- `seq_len`: Input sequence length (default: 48)
- `label_len`: Decoder prefix length (default: 32)
- `mask_rate`: Missing value ratio (0.1 - 0.5)
- `mask_method`: MCAR, MAR, or RDO
- `llm_model`: LLM backbone (GPT2/BERT/LLAMA/QWEN)
- `likelihood`: Distribution type ('g' for Gaussian, 'nb' for Negative Binomial)

### Hyperparameters

Model-specific hyperparameters are in `setup_LLMformer.py`.

## Dataset

The model is evaluated on the HVAC dataset:
- 25 feature columns
- 4 target variables: Total_Power, Total_Chiller_Power, System_Energy_Efficiency, Total_Cooling_Capacity
- 15-minute intervals (2023-2024)

The HVAC dataset is a trade secret, and the author has no right to open-source it.

## Probabilistic Forecasting

The model outputs:
- `mu`: Predicted mean values
- `sigma`: Predicted standard deviation
- Use these for interval predictions: `[mu - 1.96*sigma, mu + 1.96*sigma]` for 95% confidence intervals

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{llmformer2024,
  title={LLMformer: Hybrid LLM-Transformer Model for Probabilistic Time Series Forecasting},
  author={},
  journal={},
  year={2024}
}
```


## Related Papers

- [MultiAttLLM (Energy Forecasting)](../paper/MultiAttLLM)
- [ImputeLLM (Time Series Imputation)](../paper/ImputeLLM)
