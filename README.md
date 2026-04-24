<p align="center">
  <img src="figures/logo.png" alt="Logo" width="200"/>
</p>

<h1 align="center">LLM-Enhanced Time Series Models</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"></a>
</p>

<p align="center">
  This repository contains implementations for three research papers on Large Language Model (LLM) enhanced time series analysis.
</p>


## Papers and Branches

| Branch | Paper                                                            | Task | Model |
|--------|------------------------------------------------------------------|------|-------|
| [`paper/MultiAttLLM`](../../tree/paper/MultiAttLLM) | Multi-Attention LLM for Energy Demand and Generation Forecasting | Long-term Forecasting | MultiAttLLM |
| [`paper/LLMformer`](../../tree/paper/LLMformer) | Hybrid LLM-Transformer for Probabilistic Forecasting             | Probabilistic Forecasting | LLMformer |
| [`paper/ImputeLLM`](../../tree/paper/ImputeLLM) | LLM Enhanced Time Series Imputation                              | Missing Value Imputation | ImputeLLM |

## Quick Start

Clone a specific branch for the paper you're interested in:

```bash
# For energy demand forecasting
git clone -b paper/MultiAttLLM https://github.com/duankuxiao/LLM-Enhanced-Time-Series-Library.git

# For probabilistic forecasting
git clone -b paper/LLMformer https://github.com/duankuxiao/LLM-Enhanced-Time-Series-Library.git

# For time series imputation
git clone -b paper/ImputeLLM https://github.com/duankuxiao/LLM-Enhanced-Time-Series-Library.git
```

## Overview

### MultiAttLLM (Long-Term Forecasting)
- **Task**: Long-term electricity demand and renewable energy forecasting
- **Horizon**: 7-day (168-hour) predictions
- **Key Feature**: Multi-head cross-attention reprogramming between time series and LLM embeddings

### LLMformer (Probabilistic Forecasting)
- **Task**: Probabilistic time series forecasting
- **Output**: Distribution parameters (mean, standard deviation) for uncertainty quantification
- **Key Feature**: Hybrid dual-encoder architecture with likelihood-based output

### ImputeLLM (Imputation)
- **Task**: Missing value imputation in time series data
- **Patterns**: MCAR, MAR missing data patterns
- **Key Feature**: Dynamic prompts with statistical context for LLM guidance

## Supported LLM Backends

All models support multiple pre-trained LLMs:
- GPT-2 (768-dim)
- BERT (768-dim)
- LLAMA 1b/3b/8b (2048/3072/4096-dim)
- Qwen (varies)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers (Hugging Face)
- NumPy, Pandas, Scikit-learn

## Citation

If you use this code, please cite the relevant paper(s):

```bibtex
@article{hu2025novel,
  title={A novel attention-enhanced LLM approach for accurate power demand and generation forecasting},
  author={Hu, Zehuan and Gao, Yuan and Sun, Luning and Mae, Masayuki},
  journal={Renewable Energy},
  volume={252},
  pages={123465},
  year={2025},
  publisher={Elsevier}
}

@article{hu2026imputellm,
  title={ImputeLLM: A prompt-free large language model framework for robust time-series imputation in HVAC systems},
  author={Hu, Zehuan and Gao, Yuan and Cai, Gangwei and Liu, Mingzhe and Ke, Yan and Ruan, Yingjun},
  journal={Applied Energy},
  volume={414},
  pages={127822},
  year={2026},
  publisher={Elsevier}
}


```

