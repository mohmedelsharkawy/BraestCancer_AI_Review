# Explainable AI for Breast Density & Breast Cancer: Review (2015–2025)

This repository accompanies our systematic review on explainable AI (XAI) for **breast density assessment** and **breast cancer (BC) diagnosis** across imaging modalities (mammography, DBT, MRI, ultrasound, histopathology, clinical reports). We synthesize **47 studies**, methods, datasets, and outcomes with a special focus on BI-RADS–aligned density assessment and model explainability.

> **Why it matters:** In 2022 there were **~2.3M new BC cases** and **~670k deaths** worldwide; incidence and mortality continue rising—particularly in Asia and Africa—underscoring the need for accurate, explainable, and deployable AI tools.

---

## Contents

- [Key Stats](#key-stats)
- [Imaging Modalities & What’s Different](#imaging-modalities--whats-different)
- [Public Datasets (with Links)](#public-datasets-with-links)
- [Benchmarked Tasks & Representative Results](#benchmarked-tasks--representative-results)
  - [Binary Tasks](#binary-tasks)
  - [Multi-class Tasks](#multi-class-tasks)
- [How to Cite](#how-to-cite)

---

## Key Stats

- **47** peer‑reviewed studies synthesized (2020–2025).
- **Publication peaks:** 2022 (**6** studies) and 2025 (**5** in-press/online first) per our PRISMA-tracked corpus.
- **Modalities covered:** mammography, DBT, MRI, ultrasound, PET/CT, histopathology, clinical reports.
- **Focus:** performance **and** explainability (saliency/attribution), dataset mapping, and clinical alignment.


## 🌍 Global Impact

Breast cancer remains a critical global health issue with rising incidence worldwide.

<div align="center">

### Projected Growth by Region (2022 → 2050)

| Region | 2022 Cases | 2050 Projection | Growth Rate |
|--------|------------|-----------------|-------------|
| 🌏 **Asia** | ~1,000,000 | ~1,400,000 | ⬆️ **+40%** |
| 🌍 **Africa** | ~200,000 | ~450,000 | ⬆️ **+125%** |
| 🌎 **Latin America** | ~230,000 | ~350,000 | ⬆️ **+52%** |
| 🌍 **Europe** | ~550,000 | ~600,000 | ⬆️ **+9%** |
| 🌎 **North America** | ~300,000 | ~400,000 | ⬆️ **+33%** |

</div>

> 💡 **2022 Statistics**: 2.3 million new cases and ~670,000 deaths globally, accounting for nearly 7% of all cancer-related fatalities.

---
---


## Imaging Modalities & What’s Different

| Modality | What it is (1–2 words) | Strengths | Limitations / Considerations | Typical Use |
|---|---|---|---|---|
| **Mammography (FFDM)** | X-ray | Screening workhorse; detects microcalcifications; wide availability; low cost | Lower sensitivity in dense breasts; ionizing radiation; compression discomfort | Population screening; baseline assessment |
| **Digital Breast Tomosynthesis (DBT)** | 3D X-ray | Reduces tissue overlap; improves lesion conspicuity; better recall rates vs 2D | Higher dose than 2D; larger data volume; specialized reading | Screening/diagnostic follow-up in dense breasts |
| **Contrast-Enhanced Mammography (CEM)** | X-ray + contrast | Highlights tumor vascularity; often higher specificity vs mammo/US; faster than MRI | Contrast risks; slightly higher dose; availability varies | Problem solving; extent of disease |
| **Ultrasound (B-mode)** | Sound waves | No radiation; good for cysts/solid masses; adjunct in dense breasts; accessible | Operator-dependent; limited for microcalcifications; variable specificity | Supplemental screening/diagnostic work-up |
| **Contrast-Enhanced Ultrasound (CEUS)** | US + microbubbles | Assesses perfusion/microvasculature; can improve lesion characterization | Protocols not standardized; contrast safety/availability | Characterization and therapy monitoring |
| **Breast MRI (DCE-MRI)** | Magnetic resonance | Very high sensitivity; whole-breast coverage; useful for occult/extent | Costly; variable specificity; contrast required (gadolinium) | High-risk screening; pre-op staging; problem solving |
| **BSGI / MBI** | Nuclear imaging | Functional uptake; helpful in dense tissue | Higher radiation; limited availability; cost | Problem solving when MRI/CEM unavailable |
| **PET/CT** | Metabolic + anatomic | Staging; metastasis detection; treatment response | Higher dose; lower spatial resolution for small lesions; cost | Systemic staging and response assessment |
| **Thermography** | Infrared heat map | Non-contact; low cost | Low specificity; not recommended as standalone | Research/adjunct only |
| **Photoacoustic Imaging (PAI)** | Optical–US hybrid | Non-ionizing; vascular/oxygenation contrast | Early-stage/limited availability; standardization needed | Research; lesion characterization pilots |


## Public Datasets (with Links)

> **Note:** Many datasets require data use agreements or registration; ensure proper approvals before downloading.

| Dataset | Modality | Size (approx.) | Link |
|---|---|---:|---|
| **MIAS** | Mammography | 322 images | https://www.kaggle.com/datasets/kmader/mias-mammography |
| **DDSM** | Mammography | ~10,000 images (film scans) | http://www.eng.usf.edu/cvprg/mammography/database.html |
| **CBIS-DDSM** | Mammography (curated for ML) | ~2,500 cases | https://www.cancerimagingarchive.net/collection/cbis-ddsm/ |
| **INbreast** | Mammography (FFDM) | 410 images | https://paperswithcode.com/dataset/inbreast |
| **BUSI** | Ultrasound | 780 images | https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset |
| **BreakHis** | Histopathology | 7,909 images | https://www.kaggle.com/datasets/kmader/breast-cancer-histology-images |
| **BACH** | Histopathology | 400 images | https://paperswithcode.com/dataset/bach |
| **TCGA-BRCA** | WSIs + clinical | >1,000 cases | https://portal.gdc.cancer.gov/projects/TCGA-BRCA |
| **TCIA Breast MRI** | MRI | varies | https://www.cancerimagingarchive.net/collections/ (search “Breast MRI”) |

---

## Benchmarked Tasks & Representative Results

### Binary Tasks

- **Breast density: dense vs. non‑dense (A/B vs. C/D).** Mohamed et al. (2018) — CNN from scratch and pretrained; **AUC 0.988/0.986** on 22k mammograms. [Ref. 51]

- **Breast density (clinical deployment).** Lehman et al. (2019) — ResNet‑18; **94.0%** of binary assessments endorsed by radiologists (95% CI 94.0–95.0). [Ref. 41]

- **Breast density (dense vs. fatty).** Ciritsis et al. (2019) — 11‑layer CNN; **99%/96%** agreement (MLO/CC) with radiologists on external 200‑image test set. [Ref. 13]

- **Breast density via MRI.** Jing et al. (2024) — transfer learning on 514 MRIs; **Acc 76.3%**, **κ=0.71**. [Ref. 31]

- **INbreast texture radiomics (C/D vs. A/B).** Xu et al. (2024) — residual learning; **Acc 96.8%** on 410 images from 115 patients. [Ref. 83]

- **Multi‑site robustness.** Gandomkar et al. (2020) — Inception‑V3 fine‑tuning across 9 units/3 vendors; **Acc 92.0%** on 3,813 mammos (test: 150 images from 14 units). [Ref. 22]

> See **Table 7–9** in the manuscript for a complete binary‑task matrix (goals, datasets, methods, metrics).


### Multi-class Tasks

- **4‑class BI‑RADS density (A–D).** Lehman et al. (2019) — ResNet‑18; **Acc 77%**; main confusion between B and C. [Ref. 41]

- **4‑class BI‑RADS density (A–D) with external agreement.** Ciritsis et al. (2019) — 11‑layer dCNN; **Acc 90.9% (MLO), 90.1% (CC)**; **agreement 92.2%/87.4%**. [Ref. 13]

- **4‑class BI‑RADS density at scale.** Chen et al. (2025) — InceptionV3; **Acc 94.6%** (Cat A), **AP 0.895–0.953**, **~0.027 s/image** on 57,282 mammograms. [Ref. 11]

- **Imbalance‑aware ensembles.** Lopez et al. (2022) — RegL ensemble (VGG‑19, ResNeXt4D, DenseNet121); **Acc 84.6%** on 1,081 mammograms. [Ref. 45]

- **Dual‑view fusion.** Busaleh et al. (2022) — TwoViewDensityNet; **Acc 95.83% (DDSM), 96% (INbreast)**; **AUC 0.9951**. [Ref. 9]

- **Preprocessing + attention.** Kate et al. (2022) — InceptionV3 + GSA + Kapur entropy; **Acc 97.98%** on DDSM. [Ref. 35]

- **Clinical multi‑site BI‑RADS.** Lewin et al. (2023) — CNN; **Acc 84.6–89.7%**, ≤1‑category deviation from radiologists. [Ref. 42]

- **GAN‑augmented density.** Saffari et al. (2020) — cGAN+CNN; **Precision/Sensitivity 97.85%**, **Specificity 99.28%**. [Ref. 68]

- **5‑class molecular subtype (CMMD).** Ben et al. (2025) — multimodal CNN+metadata; **AUC 88.87%**, outperformed imaging‑only (**61.3%**). [Ref. 6]

- **Microcalcification triage (BI‑RADS 1/2–3/4–5).** Schonenberger et al. (2021) — three CNNs; **Validation 99.5–99.6%**; clinical accuracy 39% (BI‑RADS 4), 80.9% (BI‑RADS 5). [Ref. 70]

> See **Table 10** in the manuscript for a complete multi‑class matrix.

### 📈 Binary Classification Performance

#### Breast Density Assessment (Dense vs. Non-Dense)

| Model | Accuracy | AUC | Sensitivity | Specificity | Year | Reference |
|-------|----------|-----|-------------|-------------|------|-----------|
| **Mohamed et al. CNN** | - | **0.9882** | - | - | 2018 | [[1]](#ref1) |
| **Kate et al. InceptionV3** | **97.98%** | - | - | - | 2022 | [[2]](#ref2) |
| **Lewin et al. CNN** | 97.4% | - | - | - | 2023 | [[3]](#ref3) |
| **Xu et al. Residual** | 96.8% | - | - | - | 2018 | [[4]](#ref4) |
| **Kriti et al. SVM** | 94.4% | - | - | - | 2015 | [[5]](#ref5) |
| **Ciritsis et al. dCNN** | 99% MLO<br>96% CC | - | - | - | 2019 | [[6]](#ref6) |

#### Cancer Detection (Benign vs. Malignant)

| Model | Accuracy | AUC | Sensitivity | Specificity | Year | Reference |
|-------|----------|-----|-------------|-------------|------|-----------|
| **Mahmood et al. CNN+LSTM** | - | **0.99** | **99%** | - | 2024 | [[7]](#ref7) |
| **Abdel et al. DBN+BPNN** | **99.68%** | - | - | - | 2016 | [[8]](#ref8) |
| **Puttegowda et al. YOLOv3** | 98.8% | 0.99 | 97.2% | - | 2025 | [[9]](#ref9) |
| **Suh et al. DenseNet-169** | - | 0.952 | 87% | 88% | 2020 | [[10]](#ref10) |
| **Shimokawa et al. BilAD** | 84% | 0.90 | 73% | 93% | 2023 | [[11]](#ref11) |

### 📊 Multi-Class BI-RADS Classification (A-D)

| Model | Accuracy | AUC | Classes | Year | Reference |
|-------|----------|-----|---------|------|-----------|
| **Kate et al. InceptionV3+GSA** | **97.98%** | - | 4-class | 2022 | [[2]](#ref2) |
| **Busaleh et al. TwoViewDensityNet** | 95.83% | **0.9951** | 4-class | 2022 | [[12]](#ref12) |
| **Chen et al. InceptionV3** | 94.6% | 0.895-0.953 | 4-class | 2025 | [[13]](#ref13) |
| **Deng et al. SE-Attention CNN** | 92.17% | - | 4-class | 2020 | [[14]](#ref14) |
| **Xu et al. Residual Learning** | 92.6% | - | 4-class | 2018 | [[4]](#ref4) |
| **Ciritsis et al. dCNN** | 90.9% MLO<br>90.1% CC | - | 4-class | 2019 | [[6]](#ref6) |
| **Pawar et al. DenseNet** | 90.06% | 0.9625 | 4-class | 2022 | [[15]](#ref15) |

### 🎯 Segmentation Performance

| Model | Accuracy | Dice | Jaccard | Pearson's r | Year | Reference |
|-------|----------|------|---------|-------------|------|-----------|
| **Saffari et al. cGAN-UNet** | **98%** | **0.88** | 0.78 | - | 2020 | [[16]](#ref16) |
| **Ahn et al. CNN** | - | - | - | **0.96** | 2017 | [[17]](#ref17) |
| **Li et al. DCNN** | - | 0.79±0.13 | - | 0.97 (CV) | 2018 | [[18]](#ref18) |
| **Lee et al. FCN** | - | - | - | 0.85 | 2018 | [[19]](#ref19) |

---

## 🤖 AI Framework

Our review analyzes a comprehensive AI and XAI workflow for breast cancer analysis:

### 🔄 Complete Pipeline
```
┌─────────────────────────────────────────┐
│         INPUT LAYER                     │
├─────────────────────────────────────────┤
│ • Medical Imaging (Mammography, MRI)   │
│ • Clinical Data (History, Demographics)│
│ • Histopathology (Tissue samples)      │
│ • Genomic Data (Gene expression)       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      PREPROCESSING LAYER                │
├─────────────────────────────────────────┤
│ • Image: Normalization, Augmentation    │
│ • Data: Missing value imputation        │
│ • Feature: Texture analysis, Radiomics  │
│ • Genomic: Quality control, filtering   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   MULTIMODAL FEATURE INTEGRATION        │
├─────────────────────────────────────────┤
│ • Correlation Analysis                  │
│ • Feature Selection                     │
│ • Dimensionality Reduction (PCA, t-SNE) │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         AI MODELS LAYER                 │
├─────────────────────────────────────────┤
│ CNN:                                    │
│  • ResNet, DenseNet, EfficientNet       │
│ Ensemble:                               │
│  • Random Forest, XGBoost               │
│ Transformer:                            │
│  • Vision Transformers, BERT            │
│ Hybrid:                                 │
│  • CNN-RNN, CNN-LSTM, Multimodal Fusion │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│        XAI METHODS LAYER                │
├─────────────────────────────────────────┤
│ • LIME (Local explanations)             │
│ • SHAP (Shapley values)                 │
│ • Grad-CAM (Visual heatmaps)            │
│ • Attention Mechanisms                  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         OUTPUT LAYER                    │
├─────────────────────────────────────────┤
│ Predictions:                            │
│  • Malignant/Benign classification      │
│  • BI-RADS density categories           │
│  • Risk scores, Survival estimates      │
│ Explanations:                           │
│  • Feature importance rankings          │
│  • Visual heatmaps                      │
│  • Decision rules                       │
│ Clinical Support:                       │
│  • Treatment recommendations            │
│  • Prognostic assessments               │
│  • Risk stratification                  │
└─────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   CLINICAL DECISION SUPPORT SYSTEM      │
├─────────────────────────────────────────┤
│ • Interactive Dashboard                 │
│ • Physician Interface                   │
│ • Treatment Planning                    │
│ • Patient Communication                 │
│ • EHR Integration                       │
└─────────────────────────────────────────┘
```

### 🧠 Popular Architectures Analyzed

**Convolutional Neural Networks (CNNs):**
- ResNet (18, 50, 152)
- DenseNet (121, 169)
- VGG (16, 19)
- InceptionV3
- EfficientNet (B0-B7)

**Transformer-Based Models:**
- Vision Transformers (ViT)
- BERT-based architectures
- MobileNet-v2 with ViT

**Hybrid Approaches:**
- CNN-LSTM combinations
- CNN-SVM integration
- Multi-stream networks
- cGAN-UNet

**Ensemble Methods:**
- Random Forest
- XGBoost
- Voting classifiers
- Multi-model ensembles

---

## 📈 Results

### 📊 Publication Trends

Our PRISMA-guided systematic review identified:

<div align="center">

| Metric | Count |
|--------|-------|
| **Initial Records** | 502 |
| **Records Screened** | 480 |
| **Full-Text Assessed** | 100 |
| **Studies Included** | **47** |
| **Time Period** | 2020-2025 |

</div>

### 📚 Top Publishers

| Publisher | Studies | Percentage |
|-----------|---------|------------|
| **Elsevier** | 9 | 19.1% |
| **MDPI** | 5 | 10.6% |
| **Springer** | 4 | 8.5% |
| **RSNA** | 3 | 6.4% |
| **SPIE** | 3 | 6.4% |
| **IEEE** | 3 | 6.4% |
| **Others** | 20 | 42.6% |

### 📅 Publication Year Distribution

| Year | Studies | Key Highlights |
|------|---------|----------------|
| **2015** | 5 | Early ML approaches (SVM, PCA) |
| **2016** | 3 | Deep Belief Networks emerge |
| **2017** | 4 | Transfer learning adoption |
| **2018** | 5 | CNN dominance begins |
| **2019** | 4 | XAI methods integration |
| **2020** | 2 | COVID-19 impact on publications |
| **2021** | 1 | Virtual conferences era |
| **2022** | 6 | **Peak year** - Multi-modal fusion |
| **2023** | 4 | Transformer models |
| **2024** | 5 | LLMs for radiology reports |
| **2025** | 5 | Multimodal learning, Clinical validation |

---

## Links to Papers (selection)

> Full bibliographic details and DOIs are provided in our manuscript reference list (see Tables 7–10). Examples from our corpus:
- DOI: [10.1002/jmri.29058](https://doi.org/10.1002/jmri.29058)
- DOI: [10.1002/mp.12683](https://doi.org/10.1002/mp.12683)
- DOI: [10.1002/mp.12763](https://doi.org/10.1002/mp.12763)
- DOI: [10.1002/mp.14573](https://doi.org/10.1002/mp.14573)
- DOI: [10.1007/978-3-319-21212-8_7](https://doi.org/10.1007/978-3-319-21212-8_7)
- DOI: [10.1007/978-3-319-52277-7_13](https://doi.org/10.1007/978-3-319-52277-7_13)
- DOI: [10.1007/s00330-017-5143-2](https://doi.org/10.1007/s00330-017-5143-2)
- DOI: [10.1007/s00330-019-06357-1](https://doi.org/10.1007/s00330-019-06357-1)
- DOI: [10.1007/s00330-020-06564-2](https://doi.org/10.1007/s00330-020-06564-2)
- DOI: [10.1007/s00330-023-09474-7](https://doi.org/10.1007/s00330-023-09474-7)
- DOI: [10.1007/s11548-018-1794-0](https://doi.org/10.1007/s11548-018-1794-0)
- DOI: [10.1007/s12194-022-00686-y](https://doi.org/10.1007/s12194-022-00686-y)
- DOI: [10.1007/s41870-022-00930-z](https://doi.org/10.1007/s41870-022-00930-z)
- DOI: [10.1016/B978-0-323-95462-4](https://doi.org/10.1016/B978-0-323-95462-4)
- DOI: [10.1016/j.bspc.2025.107607](https://doi.org/10.1016/j.bspc.2025.107607)
- DOI: [10.1016/j.clinimag.2023.06.023](https://doi.org/10.1016/j.clinimag.2023.06.023)
- DOI: [10.1016/j.cmpb.2020.105489](https://doi.org/10.1016/j.cmpb.2020.105489)
- DOI: [10.1016/j.cmpb.2022.106885](https://doi.org/10.1016/j.cmpb.2022.106885)
- DOI: [10.1016/j.compbiomed.2014.10.006](https://doi.org/10.1016/j.compbiomed.2014.10.006)
- DOI: [10.1016/j.compmedimag.2016.07](https://doi.org/10.1016/j.compmedimag.2016.07)
- DOI: [10.1016/j.eswa.2015.10.015](https://doi.org/10.1016/j.eswa.2015.10.015)
- DOI: [10.1016/j.eswa.2024.123747](https://doi.org/10.1016/j.eswa.2024.123747)
- DOI: [10.1016/j.glt.2025.04.007](https://doi.org/10.1016/j.glt.2025.04.007)
- DOI: [10.1016/j.jmir.2021.10.004](https://doi.org/10.1016/j.jmir.2021.10.004)
- DOI: [10.1016/j.lana.2025.101096](https://doi.org/10.1016/j.lana.2025.101096)
- DOI: [10.1038/s41467-020-17410-5](https://doi.org/10.1038/s41467-020-17410-5)
- DOI: [10.1038/s41598-018-19369-6](https://doi.org/10.1038/s41598-018-19369-6)
- DOI: [10.1038/s41598-025-95275-5](https://doi.org/10.1038/s41598-025-95275-5)
- DOI: [10.1038/s41746-019-0214-6](https://doi.org/10.1038/s41746-019-0214-6)
- DOI: [10.1088/1361-6560/aa9f87](https://doi.org/10.1088/1361-6560/aa9f87)

## Research Gaps Identified

- **External validity & domain shift:** Many studies report single‑site performance; few include multi‑vendor, multi‑site **external validation** or head‑to‑head clinical reader studies (density and diagnosis).  
- **Patient‑level splits & leakage controls:** Several reports lack strict **patient‑level** partitioning across views/time, risking optimistic estimates—especially in small datasets.  
- **Imbalance handling & calibration:** Class imbalance (e.g., BI‑RADS A/B vs. C/D; benign vs. malignant) is often mitigated with re‑sampling, but **probability calibration** and **decision‑threshold protocols** are under‑reported.  
- **Explainability verification:** Saliency/prototype maps are shown qualitatively; **prospective human‑factors or counterfactual tests** validating that explanations improve safety/utility are rare.  
- **Fairness & subgroup performance:** Limited analyses across **age, breast density, scanner/vendor, and ethnicity**; few report **subgroup CIs** or fairness metrics.  
- **Ordinal structure underused:** For BI‑RADS density (A<B<C<D), many models ignore **ordinal constraints**, which can waste supervision and harm clinical consistency.  
- **Multimodal fusion and reports:** Sparse integration of **mammography/DBT + US/MRI + clinical notes**; limited use of report‑mined labels and weak supervision.  
- **Robustness & security:** Few stress tests for **adversarial noise**, compression, view mismatches, or **out‑of‑distribution** detection in screening workflows.  
- **Benchmarking standards:** Heterogeneous **metrics, splits, and preprocessing** prevent apples‑to‑apples comparisons; public leaderboards for density are lacking.  
- **Regulatory & deployment evidence:** Minimal **prospective trials**, workflow impact assessments, or post‑deployment monitoring (drift, alert fatigue, equity).  


## Future Work & Recommendations

- **Strong external validation:** Pre‑register **multi‑site** (≥3 vendors) studies with patient‑level splits and **external test sets**; report **CIs** via bootstrap and **calibration** (ECE, reliability curves).  
- **Ordinal‑aware objectives:** For density, prefer **ordinal regression/losses** (e.g., CORAL/CMOL, isotonic calibration) or **hierarchical heads** that respect A<B<C<D.  
- **Uncertainty & thresholding:** Report **well‑calibrated probabilities**, operating points tied to clinical **PPV/NPV** targets, and **abstention** (defer‑to‑human) policies.  
- **Explainability that works:** Pair saliency/prototype methods with **sanity checks**, **region perturbation**, and **reader‑in‑the‑loop** studies measuring trust, time, and error reduction.  
- **Fairness audits:** Stratify performance by **density, age, race/ethnicity, site/vendor**; publish subgroup CIs and mitigation plans when gaps appear.  
- **Multimodal fusion:** Explore **DBT+US/MRI**, **imaging+EHR** fusion, and **report‑supervision**; evaluate cross‑modal consistency and failure modes.  
- **Foundation & self‑supervised models:** Leverage large **SSL/foundation** encoders adapted to breast imaging; study **few‑shot** transfer and label‑efficient fine‑tuning.  
- **Federated & privacy‑preserving learning:** Use **federated**, DP‑aware methods to unlock diverse sites while protecting PHI; benchmark against pooled‑data baselines.  
- **Robustness testing:** Stress test against **compression, resolution, view order**, and synthetic artifacts; add **OOD detectors** in screening triage.  
- **Open benchmarks:** Release **patient‑level** splits, preprocessing scripts, and **evaluation toolkits**; pursue public challenges for **binary** (dense vs non‑dense; cancer vs non‑cancer) and **multi‑class** (BI‑RADS A–D; lesion types).  
- **Prospective trials & workflow impact:** Design **prospective** reader studies measuring sensitivity at matched specificity, time‑to‑decision, recall/biopsy rates, and downstream outcomes.  

## Repo Structure (suggested)

```
.
├── README.md
├── data/                 # (optional) metadata, NOT raw PHI
├── figures/              # schematic diagrams and saliency examples
├── notebooks/            # exploratory notebooks
└── src/                  # helper scripts for metrics and visualization
```

---

## Disclaimer

This repository is for research and educational purposes. Clinical deployment requires regulatory clearance, clinical trials, and appropriate QA processes.
