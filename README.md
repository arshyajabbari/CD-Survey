# Change Detection in Remote Sensing Imagery: A Survey

> A comprehensive review of statistical, machine learning, and deep learning methods for remote sensing change detection, covering 140 papers from 1996–2025.

**Mohammad Jabbarizadegan**  
Department of Electronics, Information, and Bioengineering  
Politecnico di Milano, Milan, Italy  
📧 mohammad.jabbarizadegan@polimi.it

---

## Overview

This repository accompanies the survey paper *"Change Detection in Remote Sensing Imagery: A Survey of Statistical, Machine Learning, and Deep Learning Methods"*. The survey systematically reviews change detection across three methodological paradigms — traditional statistical approaches, classical machine learning, and deep learning — with a structured taxonomy, benchmark comparisons, and analysis of open challenges.

**Key scope:**
- **140 non-redundant papers** compiled from two complementary corpora (1996–2025)
- **Three paradigms covered:** statistical (28 papers), machine learning (21), deep learning (88)
- **Full architectural arc:** from algebraic operators to Mamba/SSM and diffusion-based models
- **Weakly supervised paradigm** treated in dedicated depth (11-paper sub-corpus)
- **Quantitative comparison** on LEVIR-CD+, WHU-CD, and CDD benchmarks

---

## Architectural Taxonomy

```
Change Detection Methods
│
├── Statistical
│   ├── Algebraic      →  Differencing, CVA, Spectral Indices
│   ├── Transforms     →  PCA, MAD / IR-MAD, Time-Series Decomposition
│   └── Probabilistic  →  EM / Bayes, MRF / CRF
│
├── Machine Learning
│   ├── Supervised     →  SVM, Random Forest, OBIA
│   └── Hybrid         →  CA-Markov, Temporal ML, Semi-supervised
│
└── Deep Learning
    ├── Supervised     →  Siamese CNN, Transformer, Mamba/SSM,
    │                     Diffusion, CNN-Transformer Hybrid
    ├── Weakly Sup.    →  CAM-based, KD / Pseudo-label, Patch-level
    └── Unsupervised   →  Autoencoder, GAN-based
```

**Bi-temporal fusion strategies across DL papers:**

| Strategy | Papers | Share |
|---|---|---|
| Siamese middle-fusion | 69 | ~59% |
| Early fusion | 18 | ~15% |
| Late / other | 13 | ~11% |

---

## Benchmark Results

### LEVIR-CD+ (256×256 patches, 7120/1024/2048 train/val/test)

| Method | Paradigm | Prec. | Rec. | F1 | OA | IoU |
|---|---|---|---|---|---|---|
| FC-EF | CNN (early fusion) | 69.12 | 71.77 | 70.42 | 97.54 | 54.34 |
| SNUNet | CNN Siamese | 71.07 | 78.73 | 74.70 | 97.83 | 59.62 |
| BIT | CNN + Transformer | 83.91 | 81.20 | 82.53 | 98.60 | 70.26 |
| ChangeFormer | Transformer Siamese | 81.34 | 79.97 | 80.65 | 98.44 | 67.58 |
| SwinSUNet | Pure Transformer | 85.34 | 85.85 | 85.60 | 98.92 | 74.82 |
| HANet | CNN Siamese | 91.21 | 89.36 | 90.28 | 99.02 | 82.27 |
| MTCNet | CNN + Transformer | 90.85 | 89.62 | 90.24 | 97.02 | 82.22 |
| DESSN | CNN Siamese | 90.99 | 91.73 | 91.36 | — | — |
| USSFC-Net | Lightweight CNN | 89.70 | 92.42 | 91.04 | — | — |
| AGCDetNet | CNN (early fusion) | 92.12 | 89.45 | 90.76 | — | 83.09 |
| **CGNet** | **CNN Siamese** | **93.15** | **90.90** | **92.01** | **—** | **85.21** |
| MambaBCD-Tiny | Mamba/SSM | 88.82 | 87.26 | 88.04 | 99.03 | 78.63 |
| MambaBCD-Base | Mamba/SSM | 89.24 | 87.57 | 88.39 | 99.06 | 79.20 |

### WHU-CD

| Method | Paradigm | Prec. | Rec. | F1 | OA | IoU |
|---|---|---|---|---|---|---|
| FC-EF | CNN (early fusion) | 83.50 | 86.33 | 84.89 | 98.87 | 73.74 |
| SNUNet | CNN Siamese | 88.04 | 87.36 | 87.70 | 99.10 | 78.09 |
| BIT | CNN + Transformer | 89.83 | 90.24 | 90.04 | 99.27 | 81.88 |
| HANet | CNN Siamese | 88.30 | 88.01 | 88.16 | 99.16 | 78.82 |
| SwinSUNet | Pure Transformer | 94.08 | 92.03 | 93.04 | 99.50 | 87.00 |
| MambaBCD-Small | Mamba/SSM | 95.90 | 92.29 | 94.06 | 99.57 | 88.79 |
| **MambaBCD-Base** | **Mamba/SSM** | **96.18** | **92.23** | **94.19** | **99.58** | **89.02** |

### CDD

| Method | Paradigm | Prec. | Rec. | F1 |
|---|---|---|---|---|
| FC-EF | CNN (early fusion) | 83.45 | 98.47 | 90.34 |
| UNet++ | CNN | 89.54 | 87.11 | 87.56 |
| BIT | CNN + Transformer | 96.07 | 93.49 | 94.76 |
| DESSN | CNN Siamese | 95.56 | 93.50 | 94.53 |
| USSFC-Net | Lightweight CNN | 93.45 | 96.08 | 94.74 |
| **SNUNet** | **CNN Siamese** | **98.09** | **97.42** | **97.75** |

---

## Principal Datasets

| Dataset | Year | Modality | GSD | Pairs | Classes |
|---|---|---|---|---|---|
| LEVIR-CD | 2020 | RGB (Google Earth) | 0.5 m | 637 | 1 (building) |
| CDD | 2018 | RGB (Google Earth) | 0.03–1 m | 11 | 1 (binary) |
| WHU-CD | 2019 | Aerial RGB | 0.2–0.3 m | 1 large pair | 1 (building) |
| SYSU-CD | 2021 | Aerial RGB | 0.5 m | 20,000 | 1 (5 scenarios) |
| DSIFN-CD | 2020 | RGB | 2 m | 3,940 | 1 (binary) |
| SECOND | 2021 | Aerial RGB | 0.5–3 m | 4,662 | 6 (semantic) |
| HRSCD | 2019 | Aerial RGB | 0.5 m | 2 large pairs | 5 (semantic) |
| GZ-CD | 2020 | RGB (Google Earth) | 0.55 m | 19 | 1 (binary) |

---

## Repository Contents

```
.
├── README.md
├── change_detection_survey.pdf     # Full paper
└── figures/
    ├── figure1_temporal_distribution.png
    ├── figure2_method_distribution.png
    ├── figure3_application_domains.png
    ├── figure4_taxonomy.png
    ├── figure5_fusion_strategies.png
    └── figure6_levir_progression.png
```

---

## Open Challenges

The survey identifies four persistent challenges constraining operational impact:

1. **No universal benchmark** — the three dominant datasets over-represent building change in East Asian and North American urban contexts; no benchmark spans diverse geographies, multiple modalities, and multiple change types simultaneously.
2. **Benchmarking inconsistency** — train/test splits, patch extraction, and augmentation strategies differ across papers even for the same nominal dataset, making cross-paper numerical comparison approximate.
3. **Multi-temporal generalisation** — the vast majority of methods address only bi-temporal pairs; dense time-series architectures remain underdeveloped.
4. **Label efficiency** — changed pixels comprise only 4–5% of standard benchmarks; class imbalance handling remains unsystematic.

Promising directions include geospatial foundation model adaptation (Prithvi-100M, SatMAE), diffusion-based feature pre-training, weakly supervised learning under image-level annotation, and uncertainty quantification for operational deployment.

