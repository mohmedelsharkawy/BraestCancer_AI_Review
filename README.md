# Explainable AI for Breast Density & Breast Cancer: Review (2020–2025)

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



---

## Links to Papers (selection)

> Full bibliographic details and DOIs are provided in our manuscript reference list (see Tables 7–10). A few example DOIs captured in our corpus:  
- Diagnostics **12(7)**: [10.3390/diagnostics12071564](https://doi.org/10.3390/diagnostics12071564)  
- Med. Phys.: [10.1002/mp.12763](https://doi.org/10.1002/mp.12763)  
- 1361-6560 (Phys. Med. Biol.): [10.1088/1361-6560/aa9f87](https://doi.org/10.1088/1361-6560/aa9f87)


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
