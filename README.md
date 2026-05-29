# STEP: Learning STructured Embeddings for Progressive Time Series

This repository contains code and resources for learning latent representations in progressive time series data.

## Contents

- **`/figures`**: Contains figures from HI evaluations.
- **`/CMAPSS`**: Includes code for training models on various datasets, a Jupyter notebook for interactive analysis, and scripts for running additional experiments.
- **`/vinograd`**: Contains Vinograd’s mouse neural activity data and related scripts.

## Setup

### Data Preparation

To unzip the CMAPSS dataset, run the following command in your terminal:

```bash
unzip data.zip
```

This will extract the dataset into the current directory.

## Figures

![Approach Overview](/figures/0.fig1.png)   

Figure 1. Approach overview. a) Model Learning using triplet sampling. b) Latent Space Views c) Latent Space Indicators: We derive quantifiable indicators as an interface to the latent; namely the angular progression (θt) between prototype vectors or radius (r) from the z-space origin to disentangle the progression modes. These can be used as features/indicators for downstream tasks including prognostics, forecasting, and regime identification.

### latent, theta and r capture state, not t/T

![FD002 Example of latent HIs](/figures/FD002_latent_hi_distribution_comparison_tT.png) 

### FD004 CMAPSS t-SNE of latent trajectory from engine 20

![FD004 Example of latent HIs](/figures/FD004.png) 

It reflects the learned progressive representation of the underlying estimated health states. Proxy indicators such as the angular position and the radius permit to interface with the manifold, which can be useful for industrials in quest of health indicators. A downstream model takes this latent and derived indicators, and models temporal relationships while also mitigating noise present in the dataset towards downstream tasks, such as RUL prediction in CMAPSS. 


# CMAPSS Results

### Downstream performance on RUL prediction, achieves SOTA while providing HIs + low dimensional latent.  

Tr.STEP is a downstream transformer, LR.STEP is a linear regressor. Models indicated with * provide health indicators.

| Model | FD001 | FD002 | FD003 | FD004 |
| --- | --- | --- | --- | --- |
| MLP | 37.56 | 80.03 | 37.39 | 77.30 |
| SVR | 20.96 | 42.00 | 21.05 | 45.35 |
| RVR | 23.80 | 31.30 | 22.37 | 34.34 |
| CNN | 18.45 | 30.29 | 19.82 | 29.10 |
| SSL | 12.56 | 22.73 | 12.10 | 22.66 |
| CNN-LSTM | 11.17 | – | 9.99 | – |
| MS-DCNN | 11.44 | 19.35 | 11.67 | 22.20 |
| VAE + RNN | 11.44 | 24.12 | 14.88 | 26.50 |
| MLE4X+CCF | 11.57 | 18.84 | 11.83 | 20.70 |
| RVE | 13.42 | 14.92 | 12.51 | 16.30 |
| Prob. CNN | 12.42 | 13.72 | 12.16 | 15.90 |
| Wang et al.* | – | – | – | 15.42 |
| I-GLIDE* | **9.47** | 16.18 | **8.29** | 12.30 |
| Tr.BaseAE | 11.48 | 35.21 | 12.23 | 36.53 |
| Tr.SoftCLT z=2 | 13.26 | 19.15 | 12.21 | 21.14 |
| Tr.SoftCLT z=4 | 11.56 | 17.72 | 13.36 | 19.75 |
| Tr.STEP z=2* | 10.65 | **12.52** | 11.48 | 16.93 |
| LR.STEP z=2* | 15.30 | 14.38 | 17.96 | 19.48 |
| Tr.STEP z=4* | 11.26 | **12.34** | 13.35 | **12.10** |
| LR.STEP z=4* | 12.87 | **12.94** | 14.49 | 20.40 |
| Tr.STEP z=8* | 11.34 | **12.87** | 11.34 | 17.37 |
| Tr.STEP z=16* | 11.62 | **12.74** | 11.33 | 17.59 |
| Tr.STEP z=32* | 12.40 | **12.94** | 11.46 | 17.54 |
