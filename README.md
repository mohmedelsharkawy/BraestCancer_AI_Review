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

- **47** peer-reviewed studies synthesized (2020–2025).  
- **Modalities covered:** mammography, DBT, MRI, ultrasound, PET/CT, histopathology, clinical reports.  
- **Focus:** performance **and** explainability (saliency/attribution), dataset mapping, and clinical alignment.

---

## Imaging Modalities & What’s Different

**Mammography** – population workhorse; lower sensitivity in **dense** breasts; ionizing radiation.  
**CEM** – adds contrast for vascularized lesions; higher specificity than mammo/US in several settings; slightly higher dose; contrast risks.  
**Ultrasound** – accessible, radiation-free; operator-dependent; limited for microcalcifications.  
**CEUS** – microvasculature/perfusion; high sensitivity/specificity in studies; protocols not yet standardized.  
**MRI** – very high sensitivity for occult disease; costly; specificity varies.  
**DBT** – reduces tissue overlap; ~2× dose vs standard mammo.  
**BSGI/PET-CT** – functional/metabolic staging; higher dose/cost; limited availability.  
**Thermography** – non-contact, low-cost; low pathological specificity.  
**PAI** – hybrid optical–US; non-ionizing; promising for angiogenesis; needs standardization/validation.

---

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

- **Malignant vs. Benign (or Cancer vs. Non-Cancer):**  
  - Typical pipelines: CNN/ViT backbones + transfer learning; weighted loss for class imbalance; grad-CAM/IG for explainability.  
  - Representative metrics: **AUC 0.90–0.99**, **Accuracy 85–95%**, depending on modality and dataset size.  
  - Common pitfalls: data leakage across patient splits, cross-institution generalization, density-related sensitivity drop on mammograms.

- **Dense (BI-RADS C/D) vs. Non-dense (A/B):**  
  - Methods: ResNet/EfficientNet/ViT; calibration-aware thresholds; agreement measured against radiologist labels.  
  - Typical outcomes: **Quadratic κ 0.70–0.85**, macro-F1 **0.80–0.90** on curated digital cohorts.

### Multi-class Tasks

- **BI-RADS 4-class density (A/B/C/D)** or **multi-class lesion types (mass, calcification, architectural distortion, asymmetry)**:  
  - Approaches: multi-head classifiers, ordinal losses for density (A<B<C<D); prototype or hierarchical heads for interpretability.  
  - Typical outcomes: **macro-F1 0.75–0.90**, per-class AUCs often **>0.90** for C/D, lower for A/B due to label ambiguity.

> ⚠️ **Reproducibility tips:** Always report **patient-level splits**, **confidence intervals** (bootstrapping), and **external validation** if possible; include **saliency sanity checks** and **counterfactuals** for XAI.

---

## How to Cite

If you use this review or the accompanying assets, please cite:

```
@article{YourReview2025,
  title   = {Explainable AI for Breast Density and Breast Cancer: A Systematic Review (2020–2025)},
  author  = {Your Name and Coauthors},
  journal = {Journal/Archive Name},
  year    = {2025},
  note    = {GitHub: https://github.com/your-org/your-repo}
}
```

---

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
