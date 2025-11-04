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

### 🏆 Top Performing Models by Task

#### 1️⃣ Binary Density Classification (Dense vs. Non-Dense)
🥇 **Mohamed et al. (2018)** [[1]](#ref1)
- **AUC**: 0.9882 (trained from scratch)
- **Approach**: CNN with custom architecture
- **Dataset**: 22,000 mammograms

#### 2️⃣ Binary Cancer Detection (Benign vs. Malignant)
🥇 **Mahmood et al. (2024)** [[7]](#ref7)
- **Sensitivity**: 99%
- **AUC**: 0.99
- **Approach**: CNN+LSTM/SVM hybrid with Grad-CAM

#### 3️⃣ Multi-Class BI-RADS Classification (A-D)
🥇 **Kate et al. (2022)** [[2]](#ref2)
- **Accuracy**: 97.98%
- **Approach**: InceptionV3 + GSA optimization + Kapur's entropy
- **Dataset**: DDSM

#### 4️⃣ Dense Tissue Segmentation
🥇 **Saffari et al. (2020)** [[16]](#ref16)
- **Accuracy**: 98%
- **Dice Coefficient**: 0.88
- **Approach**: cGAN-UNet hybrid
- **Speed**: 7-8 images/second

---

## 🚧 Challenges & Future Directions

### ⚠️ Current Challenges

| Challenge | Impact | Severity | Solutions Proposed |
|-----------|--------|----------|-------------------|
| **Dataset Bias** | Limited generalization across demographics | 🔴 **Critical** | Multi-institutional collaboration |
| **Model Interpretability** | Low clinical trust and adoption | 🔴 **Critical** | Enhanced XAI techniques |
| **External Validation** | Unclear real-world performance | 🔴 **Critical** | Prospective multi-site trials |
| **Class Imbalance** | Poor minority class performance | 🟡 **High** | Advanced augmentation, loss functions |
| **Computational Cost** | Limited accessibility in low-resource settings | 🟡 **High** | Model compression, edge deployment |
| **Annotation Variability** | Inconsistent BI-RADS labeling | 🟡 **High** | Consensus-based labeling |
| **Regulatory Approval** | Slow clinical adoption pathway | 🟢 **Medium** | Standardized evaluation protocols |
| **Multimodal Integration** | Lack of fusion frameworks | 🟢 **Medium** | Cross-modal learning architectures |

### 🔮 Future Research Directions

#### 1. 📊 Data & Benchmarking
- ✨ **Curated cross-site benchmarks** with standardized protocols
- ✨ **Diverse, representative datasets** spanning ethnicities, ages, scanner types
- ✨ **Bias audits** for fairness and equity assessment
- ✨ **Federated learning** for privacy-preserving multi-institutional training

#### 2. 🤖 Model Development
- ✨ **Multimodal fusion architectures** (imaging + clinical + genomic)
- ✨ **Efficient models** for edge deployment and low-resource settings
- ✨ **Self-supervised learning** to reduce annotation burden
- ✨ **Foundation models** pre-trained on large-scale medical imaging

#### 3. 🔍 Explainability & Trust
- ✨ **Enhanced XAI techniques** beyond heatmaps (counterfactuals, concept-based)
- ✨ **Clinician-friendly interfaces** with intuitive visualizations
- ✨ **Human-in-the-loop systems** for collaborative decision-making
- ✨ **Uncertainty quantification** for reliable confidence estimates

#### 4. 🏥 Clinical Translation
- ✨ **Prospective clinical trials** with real-world validation
- ✨ **Workflow integration** with PACS, EHR systems
- ✨ **Regulatory pathways** (FDA, CE mark) compliance frameworks
- ✨ **Health economics studies** demonstrating cost-effectiveness

#### 5. 🌍 Global Impact
- ✨ **Low-resource adaptations** for developing countries
- ✨ **Multi-language support** for global accessibility
- ✨ **Teleradiology integration** for remote diagnostics
- ✨ **Open-source tools** for democratizing AI in breast cancer

---

## 📚 Citations

### Main Paper

**This Systematic Review:**
```bibtex

@article{doorgeshwarea2025explainable,
title={Explainable AI for Breast Density Across Imaging Modalities: A Systematic State-of-the-Art Review},
author={Doorgeshwarea, Bootun and Elsharkawy, Mohamed and Ghaza, Mohammed and Khalil, Ashraf and Contractor, Sohail and Giridharan, Guruprasad A. and Maleika, Mamodekhan and El-Baz, Ayman},
journal={Artificial Intelligence},
year={2025},
month={November},
publisher={Elsevier},
note={Preprint submitted to Artificial Intelligence},
keywords={Breast Cancer, Breast Density, Explainable AI, Deep Learning, Computer-Aided Diagnosis, Medical Imaging Modalities, BI-RADS}
}

---

### Key Papers on Binary Classification

<a name="ref1"></a>
**[1] Mohamed et al. (2018) - CNN for Mammographic Density Classification**
```bibtex@article{mohamed2018deep,
title={A deep learning method for classifying mammographic breast density categories},
author={Mohamed, Aly A and Berg, Wendie A and Peng, Hong and Luo, Yahong and Jankowitz, Rachel C and Wu, Shandong},
journal={Medical Physics},
volume={45},
number={1},
pages={314--321},
year={2018},
publisher={Wiley},
doi={10.1002/mp.12683},
note={AUC = 0.9882 for binary density classification}
}

<a name="ref3"></a>
**[3] Lewin et al. (2023) - PACS-Integrated ML Breast Density Classifier**
```bibtex@article{lewin2023pacs,
title={PACS-integrated machine learning breast density classifier: clinical validation},
author={Lewin, John and Schoenherr, Sven and Seebass, Martin and Lin, MingDe and Philpotts, Liane and Etesami, Maryam and Butler, Reni and Durand, Melissa and Heller, Samantha and Heacock, Laura and others},
journal={Clinical Imaging},
volume={101},
pages={200--205},
year={2023},
publisher={Elsevier},
doi={10.1016/j.clinimag.2023.06.023},
note={97.4% accuracy at Site B for binary classification}
}

<a name="ref6"></a>
**[6] Ciritsis et al. (2019) - Deep CNN for Density Determination**
```bibtex@article{ciritsis2019determination,
title={Determination of mammographic breast density using a deep convolutional neural network},
author={Ciritsis, Alexander and Rossi, Cristina and De Martini, Ilaria Vittoria and Eberhard, Matthias and Marcon, Magda and Becker, Anton S and Berger, Nicole and Boss, Andreas},
journal={The British Journal of Radiology},
volume={92},
number={1093},
pages={20180691},
year={2019},
publisher={British Institute of Radiology},
doi={10.1259/bjr.20180691},
note={99% MLO, 96% CC agreement with radiologists}
}

<a name="ref7"></a>
**[7] Mahmood et al. (2024) - Hybrid CNN for Cancer Diagnosis**
```bibtex@article{mahmood2024harnessing,
title={Harnessing the power of radiomics and deep learning for improved breast cancer diagnosis with multiparametric breast mammography},
author={Mahmood, Tariq and Saba, Tanzila and Rehman, Amjad and Alamri, Faten S},
journal={Expert Systems with Applications},
volume={249},
pages={123747},
year={2024},
publisher={Elsevier},
doi={10.1016/j.eswa.2024.123747},
note={99% sensitivity, AUC = 0.99 for benign vs. malignant}
}

<a name="ref8"></a>
**[8] Abdel et al. (2016) - Deep Belief Networks**
```bibtex@article{abdel2016breast,
title={Breast cancer classification using deep belief networks},
author={Abdel-Zaher, Ahmed M and Eldeib, Ayman M},
journal={Expert Systems with Applications},
volume={46},
pages={139--144},
year={2016},
publisher={Elsevier},
doi={10.1016/j.eswa.2015.10.015},
note={99.68% accuracy for binary cancer classification}
}

<a name="ref9"></a>
**[9] Puttegowda et al. (2025) - Enhanced ML for Mammogram Classification**
```bibtex@article{puttegowda2025enhanced,
title={Enhanced machine learning models for accurate breast cancer mammogram classification},
author={Puttegowda, Kiran and Veeraprathap, V and Kumar, HS Ranjan and Sudheesh, KV and Prabhavathi, K and Vinayakumar, Ravi and Tabianan, Kayalvily},
journal={Global Transitions},
year={2025},
publisher={Elsevier},
doi={10.1016/j.glt.2025.04.007},
note={98.8% accuracy with YOLOv3, AUC = 0.99}
}

<a name="ref10"></a>
**[10] Suh et al. (2020) - DenseNet for Various Densities**
```bibtex@article{suh2020automated,
title={Automated breast cancer detection in digital mammograms of various densities via deep learning},
author={Suh, Yong Joon and Jung, Jaewon and Cho, Bum-Joo},
journal={Journal of Personalized Medicine},
volume={10},
number={4},
pages={211},
year={2020},
publisher={MDPI},
doi={10.3390/jpm10040211},
note={AUC = 0.952-0.954 across density categories}
}

<a name="ref11"></a>
**[11] Shimokawa et al. (2023) - Bilateral Asymmetry Detection**
```bibtex@article{shimokawa2023deep,
title={Deep learning model for breast cancer diagnosis based on bilateral asymmetrical detection (BilAD) in digital breast tomosynthesis images},
author={Shimokawa, Daiki and Takahashi, Kengo and Kurosawa, Daiya and Takaya, Eichi and Oba, Ken and Yagishita, Kazuyo and Fukuda, Toshinori and Tsunoda, Hiroko and Ueda, Takuya},
journal={Radiological Physics and Technology},
volume={16},
number={1},
pages={20--27},
year={2023},
publisher={Springer},
doi={10.1007/s12194-022-00686-y},
note={84% accuracy, 0.90 AUC using bilateral features}
}

---

### Key Papers on Multi-Class Classification

<a name="ref2"></a>
**[2] Kate et al. (2022) - GSA + InceptionV3 for 4-Class Density**
```bibtex@article{kate2022breast,
title={Breast tissue density classification based on gravitational search algorithm and deep learning: a novel approach},
author={Kate, Vandana and Shukla, Pragya},
journal={International Journal of Information Technology},
volume={14},
number={7},
pages={3481--3493},
year={2022},
publisher={Springer},
doi={10.1007/s41870-022-00930-z},
note={97.98% accuracy for BI-RADS A-D classification}
}

<a name="ref12"></a>
**[12] Busaleh et al. (2022) - Two-View Mammographic Classification**
```bibtex@article{busaleh2022twoviewdensitynet,
title={TwoViewDensityNet: Two-view mammographic breast density classification based on deep convolutional neural network},
author={Busaleh, Mariam and Hussain, Muhammad and Aboalsamh, Hatim A and Al Sultan, Sarah A},
journal={Mathematics},
volume={10},
number={23},
pages={4610},
year={2022},
publisher={MDPI},
doi={10.3390/math10234610},
note={96% accuracy on INbreast, AUC = 0.9951 on DDSM}
}

<a name="ref13"></a>
**[13] Chen et al. (2025) - InceptionV3 for Density Prediction**
```bibtex@article{chen2025deep,
title={Deep learning prediction of mammographic breast density using screening data},
author={Chen, Chen and Wang, Enyu and Wang, Vicky Yang and Chen, Xiayi and Feng, Bojian and Yan, Ruxuan and Zhu, Lingying and Xu, Dong},
journal={Scientific Reports},
volume={15},
number={1},
pages={11602},
year={2025},
publisher={Nature Portfolio},
doi={10.1038/s41598-025-95275-5},
note={94.6% accuracy for Category A, AP = 0.895-0.953}
}

<a name="ref14"></a>
**[14] Deng et al. (2020) - SE-Attention Neural Networks**
```bibtex@article{deng2020classification,
title={Classification of breast density categories based on SE-Attention neural networks},
author={Deng, Jian and Ma, Yanyun and Li, Deng-ao and Zhao, Jumin and Liu, Yi and Zhang, Hui},
journal={Computer Methods and Programs in Biomedicine},
volume={193},
pages={105489},
year={2020},
publisher={Elsevier},
doi={10.1016/j.cmpb.2020.105489},
note={92.17% accuracy with DenseNet + SE-Attention}
}

<a name="ref15"></a>
**[15] Pawar et al. (2022) - Multichannel DenseNet**
```bibtex@article{pawar2022multichannel,
title={Multichannel DenseNet architecture for classification of mammographic breast density for breast cancer detection},
author={Pawar, Shivaji D and Sharma, Kamal K and Sapate, Suhas G and Yadav, Geetanjali Y and Alroobaea, Roobaea and Alzahrani, Sabah M and Hedabou, Mustapha},
journal={Frontiers in Public Health},
volume={10},
pages={885212},
year={2022},
publisher={Frontiers},
doi={10.3389/fpubh.2022.885212},
note={90.06% accuracy, AUC = 0.9625 for BI-RADS I-IV}
}

---

### Key Papers on Segmentation

<a name="ref16"></a>
**[16] Saffari et al. (2020) - cGAN-UNet for Segmentation**
```bibtex@article{saffari2020fully,
title={Fully automated breast density segmentation and classification using deep learning},
author={Saffari, Nasibeh and Rashwan, Hatem A and Abdel-Nasser, Mohamed and Singh, Vivek Kumar and Arenas, Meritxell and Mangina, Eleni and Herrera, Blas and Puig, Domenec},
journal={Diagnostics},
volume={10},
number={11},
pages={988},
year={2020},
publisher={MDPI},
doi={10.3390/diagnostics10110988},
note={98% accuracy, Dice = 0.88, processing 7-8 images/second}
}

<a name="ref17"></a>
**[17] Ahn et al. (2017) - CNN for Density Estimation**
```bibtex@inproceedings{ahn2017novel,
title={A novel deep learning-based approach to high accuracy breast density estimation in digital mammography},
author={Ahn, Chul Kyun and Heo, Changyong and Jin, Heongmin and Kim, Jong Hyo},
booktitle={Medical Imaging 2017: Computer-Aided Diagnosis},
volume={10134},
pages={691--697},
year={2017},
organization={SPIE},
doi={10.1117/12.2254264},
note={Pearson's r = 0.96 with manual segmentations}
}

<a name="ref18"></a>
**[18] Li et al. (2018) - Supervised DCNN for Density**
```bibtex@article{li2018computer,
title={Computer-aided assessment of breast density: comparison of supervised deep learning and feature-based statistical learning},
author={Li, Songfeng and Wei, Jun and Chan, Heang-Ping and Helvie, Mark A and Roubidoux, Marilyn A and Lu, Yao and Zhou, Chuan and Hadjiiski, Lubomir M and Samala, Ravi K},
journal={Physics in Medicine & Biology},
volume={63},
number={2},
pages={025005},
year={2018},
publisher={IOP Publishing},
doi={10.1088/1361-6560/aa9f87},
note={Dice = 0.79±0.13 (CV), r = 0.97}
}

<a name="ref19"></a>
**[19] Lee et al. (2018) - Fully Convolutional Network**
```bibtex@article{lee2018automated,
title={Automated mammographic breast density estimation using a fully convolutional network},
author={Lee, Juhun and Nishikawa, Robert M},
journal={Medical Physics},
volume={45},
number={3},
pages={1178--1190},
year={2018},
publisher={Wiley},
doi={10.1002/mp.12763},
note={Strong correlation with BI-RADS (p = 0.85)}
}

---

### Additional Key References

<a name="ref4"></a>
**[4] Xu et al. (2018) - Residual Learning for Density**
```bibtex@article{xu2018classifying,
title={Classifying mammographic breast density by residual learning},
author={Xu, Jingxu and Li, Cheng and Zhou, Yongjin and Mou, Lisha and Zheng, Hairong and Wang, Shanshan},
journal={arXiv preprint arXiv:1809.10241},
year={2018},
note={96.8% accuracy for binary, 92.6% for multi-class}
}

<a name="ref5"></a>
**[5] Kriti et al. (2015) - PCA-PNN and PCA-SVM CAD**
```bibtex@incollection{kriti2015pca,
title={PCA-PNN and PCA-SVM based CAD systems for breast density classification},
author={Kriti and Virmani, Jitendra and Dey, Nilanjan and Kumar, Vinod},
booktitle={Applications of Intelligent Optimization in Biology and Medicine},
pages={159--180},
year={2015},
publisher={Springer},
doi={10.1007/978-3-319-21212-8_7},
note={94.4% accuracy (SVM), 92.5% (PNN)}
}

---

## 👥 Authors

<div align="center">

**Bootun Doorgeshwarea¹**, Mohamed Elsharkawy², Mohammed Ghaza³, Ashraf Khalil⁴, Sohail Contractor⁵, Guruprasad A. Giridharan², Mamodekhan Maleika¹, **Ayman El-Baz²***

</div>

### Affiliations

¹ **University of Mauritius**  
Department of Software and Information Systems  
Faculty of Information Communication and Digital Technologies  
Moka, Mauritius

² **University of Louisville**  
Department of Bioengineering  
Louisville, KY, USA

³ **Abu Dhabi University**  
Electrical, Computer and Biomedical Engineering Department  
Abu Dhabi, UAE

⁴ **Zayed University**  
College of Technological Innovation  
Abu Dhabi, UAE

⁵ **University of Louisville**  
Department of Radiology  
Louisville, KY, USA



---

*Last Updated: November 4, 2025*

**Made with ❤️ by the Bioengineering Imaging Lab at University of Louisville**

</div>
