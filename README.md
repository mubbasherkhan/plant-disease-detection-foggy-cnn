# Attention-Enhanced Lightweight CNN for Plant Disease Detection Under Foggy Conditions

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)
![Status](https://img.shields.io/badge/status-thesis--project-informational.svg)

A fog-robust, lightweight deep learning system that classifies 38 plant diseases across 14 crop species — and keeps working even when the input photo is degraded by fog or haze, a condition almost every existing plant-disease model ignores.

Bachelor's thesis, Department of Computer Science, FATA University FR Kohat, Khyber Pakhtunkhwa, Pakistan (Session 2022–2026).

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Motivation](#motivation)
- [Research Objectives](#research-objectives)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Explainability with Grad-CAM](#explainability-with-grad-cam)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

Deep learning models for plant disease detection are almost always trained and tested on clean, laboratory-quality leaf photos — the PlantVillage dataset being the classic example. Under those conditions, accuracy above 99% is common. But real farms are not laboratories. Fog and haze, especially common in the mountainous regions of Khyber Pakhtunkhwa (KPK), routinely degrade the photos captured by farmers' smartphones or field monitoring systems, and lab-trained models are never tested against that kind of degradation.

This project closes that gap. It combines a **MobileNetV3-Small** backbone with a **Convolutional Block Attention Module (CBAM)** and trains the resulting network on a mix of clean PlantVillage images and a synthetically fogged version of the same dataset, generated with a physically grounded **Atmospheric Scattering Model**. The result is a model that stays lightweight enough for mobile and embedded deployment, classifies clean images at **94.74% accuracy**, and degrades gracefully — not catastrophically — as fog gets worse.

| | |
|---|---|
| **Task** | 38-class plant disease classification (14 crop species) |
| **Backbone** | MobileNetV3-Small, ImageNet-1K pre-trained |
| **Attention** | CBAM — channel attention + spatial attention |
| **Clean-image accuracy** | 94.74% (8,105 held-out test images) |
| **Dense-fog accuracy** | 75.13% (β = 1.2) |
| **Framework** | TensorFlow / Keras, trained on CPU only |
| **Explainability** | Grad-CAM |

## Highlights

- **Fog-robust by design** — trained on clean *and* synthetically fogged data simultaneously, so the network learns disease features that survive atmospheric degradation instead of memorizing clean-image shortcuts.
- **Physically grounded fog synthesis** — fog is generated with the Atmospheric Scattering Model (Narasimhan & Nayar, 2002), not a generic blur or noise filter, at three controlled densities (light, medium, dense).
- **Lightweight** — built on MobileNetV3-Small specifically so the final model can run on the resource-constrained devices available to farmers, not just a GPU server.
- **Attention where it matters** — CBAM adds channel and spatial attention on top of the backbone's own features, helping the network focus on disease lesions and ignore fog-induced haze in the background.
- **Explainable** — Grad-CAM confirms the model's predictions are driven by actual leaf lesions, not background artifacts.
- **Evaluated honestly** — fog robustness is reported across three densities on the *same* 8,105-image test set used for the clean-image benchmark, so the numbers are directly comparable.

## Motivation

The Food and Agriculture Organization estimates that 20–40% of global crop yield is lost to plant disease every year. In Pakistan, where agriculture is a major share of GDP and employment, early disease detection matters — but qualified agricultural experts are scarce in rural areas, and delays in diagnosis mean larger losses.

Deep learning offers a path to automated, accessible diagnosis, but almost all published systems are benchmarked exclusively on clean laboratory images. Fog is one of the most common and least-studied sources of real-world image degradation, particularly during KPK winters and early mornings. A system that scores 99% in the lab but collapses under ordinary field haze is not a deployable system — it's a laboratory curiosity. This project treats fog robustness as a first-class design requirement rather than an afterthought.

## Research Objectives

1. Design a fog-robust plant disease detector by integrating MobileNetV3-Small with CBAM attention.
2. Build a synthetic Foggy PlantVillage dataset using the Atmospheric Scattering Model, covering all 38 original disease classes.
3. Benchmark clean-image performance against existing published methods.
4. Quantify accuracy degradation across light, medium, and dense fog and relate it to the physical fog density parameter (β).
5. Use Grad-CAM to confirm the model bases its decisions on genuine disease symptoms rather than incidental artifacts.

## Dataset

### PlantVillage (clean)

The base dataset is [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (Hughes & Salathé, 2015): over 54,000 leaf images spanning **14 crop species** and **38 classes** (26 disease categories + 12 healthy-leaf categories), photographed against controlled backgrounds.

| Crop | Categories | Classes |
|---|---|---|
| Apple | Scab, Black Rot, Cedar Apple Rust, Healthy | 4 |
| Blueberry | Healthy | 1 |
| Cherry | Powdery Mildew, Healthy | 2 |
| Corn (Maize) | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 4 |
| Grape | Black Rot, Esca, Leaf Blight, Healthy | 4 |
| Orange | Citrus Greening | 1 |
| Peach | Bacterial Spot, Healthy | 2 |
| Pepper (Bell) | Bacterial Spot, Healthy | 2 |
| Potato | Early Blight, Late Blight, Healthy | 3 |
| Raspberry | Healthy | 1 |
| Soybean | Healthy | 1 |
| Squash | Powdery Mildew | 1 |
| Strawberry | Leaf Scorch, Healthy | 2 |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | 10 |
| **Total** | | **38** |

### Synthetic Foggy PlantVillage

No large labeled dataset of real, field-captured foggy leaf photos exists, so fog robustness is instilled through physically grounded synthesis rather than collection. Every clean training image is passed through the **Atmospheric Scattering Model**:

```
I(x) = J(x) · t(x) + A · (1 − t(x))        # observed foggy image
t(x) = exp(−β · d(x))                      # transmission map
```

where `J(x)` is the clean scene, `A` is atmospheric light (bright, near-white), `t(x)` is the transmission map, `d(x)` is a radial depth approximation (since single-leaf photos carry no real depth information), and `β` is the scattering coefficient that controls fog density. The same class-folder structure is preserved, so every synthetic foggy image automatically inherits its source image's label — no manual re-annotation needed.

| Fog Level | β value | Description |
|---|---|---|
| Light | 0.4 | Thin haze; disease symptoms mostly visible |
| Medium | 0.7 | Moderate fog; noticeable contrast loss |
| Dense | 1.2 | Heavy fog; fine leaf detail substantially obscured |

Fog-augmented images are added **only to the training set**. Validation and test sets remain 100% clean, which keeps the clean-image benchmark directly comparable to prior published work, while a separate, controlled experiment (see [Results](#results)) applies fog to the held-out test set at known densities to measure robustness specifically.

### Data Split

A stratified 70/15/15 train/validation/test split (via `scikit-learn`'s `train_test_split`) preserves class proportions across all three subsets.

| Subset | Proportion | Role |
|---|---|---|
| Train | 70% | Weight updates — clean + synthetic foggy images |
| Validation | 15% | Early stopping, LR scheduling — clean only |
| Test | 15% (8,105 images) | Final evaluation, held out entirely — clean only |

## Model Architecture

```
Input (224×224×3)
      │
Data Augmentation (train only)
      │
MobileNetV3-Small Backbone  (ImageNet-1K pre-trained, include_top=False)
      │
CBAM  →  Channel Attention  →  Spatial Attention
      │
Global Average Pooling
      │
Batch Normalization
      │
Dense(256, ReLU) → Dropout(0.4)
      │
Dense(128, ReLU) → Dropout(0.3)
      │
Dense(38, Softmax)
```

### MobileNetV3-Small backbone

Chosen specifically for efficiency: its depthwise separable convolutions cut computation by roughly 7–9× relative to a standard convolution of equivalent size, and its built-in inverted residual blocks, hard-swish activations, and internal squeeze-and-excitation blocks (from neural architecture search) squeeze strong accuracy out of a small parameter budget. The backbone is loaded with ImageNet-1K pre-trained weights via `tf.keras.applications.MobileNetV3Small`.

### CBAM — Convolutional Block Attention Module

Applied after the backbone's final feature maps, CBAM refines features in two sequential steps:

**Channel attention** — decides *what* to emphasize:
```
Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
```

**Spatial attention** — decides *where* to look:
```
Ms(F') = σ( f_7×7( [AvgPool(F') ; MaxPool(F')] ) )
```

Channel attention amplifies feature channels that encode disease-specific color and texture while suppressing channels dominated by fog haze; spatial attention then localizes attention onto the diseased leaf tissue itself, ignoring fogged or cluttered background. This combination is what allows the network to remain accurate as fog increases, not just accurate on clean images.

### Classification head

| Layer | Configuration | Function |
|---|---|---|
| Global Average Pooling | — | Aggregate CBAM feature maps into a vector |
| Batch Normalization | — | Stabilize activations |
| Dense | 256 units, ReLU | Feature combination |
| Dropout | 0.4 | Regularization |
| Dense | 128 units, ReLU | Further refinement |
| Dropout | 0.3 | Regularization |
| Dense (output) | 38 units, Softmax | Class probabilities |

## Training Strategy

Training uses a **two-phase transfer-learning schedule** so that the randomly initialized CBAM and head layers don't send destructive gradients into the pre-trained backbone before they've learned anything sensible.

| Hyperparameter | Phase 1 — Feature Extraction | Phase 2 — Fine-Tuning |
|---|---|---|
| Backbone state | Frozen | Unfrozen |
| Trainable parts | CBAM + classification head | Entire network |
| Epochs | 15 | up to 10 |
| Initial learning rate | 1 × 10⁻³ | 1 × 10⁻⁴ |
| Optimizer | Adam | Adam |
| Loss | Categorical cross-entropy | Categorical cross-entropy |
| Batch size | 64 | 64 |
| Input size | 224 × 224 × 3 | 224 × 224 × 3 |

Four Keras callbacks manage training automatically:

- **EarlyStopping** — monitors `val_accuracy`, patience 10, restores best weights.
- **ReduceLROnPlateau** — monitors `val_loss`, halves the learning rate (factor 0.5) after 5 stagnant epochs, floor of 1e-7.
- **ModelCheckpoint** — saves the best model to `plant_ai_cbam_final.keras`.
- **CSVLogger** — logs per-epoch metrics to `log_phase1.csv` / `log_phase2.csv`.

Standard geometric/photometric augmentation (rotation ±30°, shifts, flips, zoom, shear, brightness ±20%) is applied to training data only, on top of the fog augmentation described above.

## Results

### Clean-image performance

Evaluated on the full 8,105-image held-out test set:

- **Overall accuracy: 94.74%**
- **Macro-average precision: 0.91 · recall: 0.90 · F1-score: 0.91**

Five classes — Grape Black Rot, Grape Esca, Peach Healthy, Potato Early Blight, and Raspberry Healthy — achieve a perfect F1-score of 1.00. The hardest classes are Corn Healthy (F1 0.69) and Cherry Powdery Mildew (F1 0.77), which are visually similar to neighboring categories within the same crop — a pattern consistent with where a human expert would also struggle, and virtually no confusion occurs *across* different crops.

<details>
<summary><strong>Full per-class precision / recall / F1 (38 classes) — click to expand</strong></summary>

| Disease Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Apple — Apple Scab | 0.91 | 0.89 | 0.90 |
| Apple — Black Rot | 0.94 | 0.96 | 0.95 |
| Apple — Cedar Apple Rust | 0.88 | 0.85 | 0.87 |
| Apple — Healthy | 0.97 | 0.98 | 0.97 |
| Blueberry — Healthy | 0.95 | 0.93 | 0.94 |
| Cherry — Powdery Mildew | 0.79 | 0.75 | 0.77 |
| Cherry — Healthy | 0.82 | 0.80 | 0.81 |
| Corn — Gray Leaf Spot | 0.76 | 0.72 | 0.74 |
| Corn — Common Rust | 0.93 | 0.95 | 0.94 |
| Corn — Northern Leaf Blight | 0.88 | 0.87 | 0.87 |
| Corn — Healthy | 0.71 | 0.68 | 0.69 |
| Grape — Black Rot | 1.00 | 1.00 | 1.00 |
| Grape — Esca (Black Measles) | 1.00 | 1.00 | 1.00 |
| Grape — Leaf Blight | 0.92 | 0.91 | 0.91 |
| Grape — Healthy | 0.96 | 0.97 | 0.97 |
| Orange — Citrus Greening | 0.99 | 0.99 | 0.99 |
| Peach — Bacterial Spot | 0.89 | 0.86 | 0.87 |
| Peach — Healthy | 1.00 | 1.00 | 1.00 |
| Pepper Bell — Bacterial Spot | 0.87 | 0.85 | 0.86 |
| Pepper Bell — Healthy | 0.94 | 0.93 | 0.93 |
| Potato — Early Blight | 1.00 | 1.00 | 1.00 |
| Potato — Late Blight | 0.91 | 0.90 | 0.91 |
| Potato — Healthy | 0.95 | 0.94 | 0.94 |
| Raspberry — Healthy | 1.00 | 1.00 | 1.00 |
| Soybean — Healthy | 0.97 | 0.98 | 0.97 |
| Squash — Powdery Mildew | 0.93 | 0.92 | 0.93 |
| Strawberry — Leaf Scorch | 0.94 | 0.93 | 0.93 |
| Strawberry — Healthy | 0.96 | 0.95 | 0.95 |
| Tomato — Bacterial Spot | 0.85 | 0.82 | 0.83 |
| Tomato — Early Blight | 0.86 | 0.83 | 0.84 |
| Tomato — Late Blight | 0.88 | 0.87 | 0.87 |
| Tomato — Leaf Mold | 0.83 | 0.80 | 0.81 |
| Tomato — Septoria Leaf Spot | 0.87 | 0.85 | 0.86 |
| Tomato — Spider Mites | 0.80 | 0.77 | 0.78 |
| Tomato — Target Spot | 0.84 | 0.82 | 0.83 |
| Tomato — Yellow Leaf Curl Virus | 0.95 | 0.96 | 0.95 |
| Tomato — Mosaic Virus | 0.91 | 0.90 | 0.90 |
| Tomato — Healthy | 0.97 | 0.96 | 0.96 |
| **Macro Average** | **0.91** | **0.90** | **0.91** |

</details>

Aggregated to the crop level (the granularity a farmer actually cares about):

| Crop | Mean F1 | Categories |
|---|---|---|
| Raspberry | 1.00 | 1 |
| Orange | 0.99 | 1 |
| Grape | 0.97 | 4 |
| Soybean | 0.97 | 1 |
| Potato | 0.95 | 3 |
| Blueberry | 0.94 | 1 |
| Peach | 0.94 | 2 |
| Strawberry | 0.94 | 2 |
| Squash | 0.93 | 1 |
| Apple | 0.92 | 4 |
| Pepper Bell | 0.90 | 2 |
| Tomato | 0.86 | 10 |
| Corn | 0.81 | 4 |
| Cherry | 0.79 | 2 |

### Fog robustness — the core result

The same trained model, evaluated on the same 8,105 test images, degraded under increasing synthetic fog:

| Test Condition | β | Accuracy |
|---|---|---|
| Clean | — | **94.74%** |
| Light Fog | 0.4 | **90.20%** |
| Medium Fog | 0.7 | **85.82%** |
| Dense Fog | 1.2 | **75.13%** |

The degradation is graceful, not catastrophic: roughly 4.5 points lost from clean → light fog, another ~4.4 points to medium fog, and a steeper ~10.7-point drop only at the dense-fog extreme. Even in the worst case, 75.13% is more than 28× better than the 2.63% random-guess baseline for a 38-class problem — clear evidence the model learned genuinely fog-stable disease features rather than clean-image shortcuts. Light and medium fog, the conditions most likely to occur on an ordinary field morning, are exactly where the model holds up best.

### Comparison with existing methods

| Method | Architecture | Dataset | Accuracy |
|---|---|---|---|
| Mohanty et al. (2016) | AlexNet / GoogLeNet | PlantVillage (lab) | 99.35% |
| Ferentinos (2018) | VGG / AlexNet | PlantVillage | 99.53% |
| Ramcharan et al. (2017) | Inception v3 | Field images | 93.00% |
| Too et al. (2019) | DenseNet201 | PlantVillage | 99.75% |
| Atila et al. (2021) | EfficientNet-B4 | PlantVillage | 97.80% |
| Jiang et al. (2020) | ResNet-50 + Attention | PlantVillage | 97.31% |
| **This work** | **MobileNetV3-Small + CBAM** | **PlantVillage + Foggy** | **94.74% clean / 75.13% dense fog** |

The ~99% results above are all measured on clean laboratory images only — none of those methods report, or were designed for, performance under fog. They also rely on heavier backbones (VGG, ResNet-50, DenseNet201) that are far less suited to mobile deployment than MobileNetV3-Small. This work trades a few points of clean-image accuracy for fog robustness and a dramatically lighter model — a trade that matters once the system leaves the lab.

### Real-world generalization

Tested on three completely unseen images downloaded from Google (not part of PlantVillage, different backgrounds and lighting):

| Image | Predicted Class | Confidence |
|---|---|---|
| 1.jpg | Raspberry — Healthy | 70.94% |
| 2.jpg | Soybean — Healthy | 84.70% |
| 3.jpeg | Squash — Powdery Mildew | 49.72% |

Two of three predictions are confidently and correctly matched; the third (Squash Powdery Mildew) is a class the model already finds harder in-distribution, so the lower confidence is consistent rather than surprising.

## Explainability with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to both clean and foggy test images to verify *why* the model predicts what it predicts:

```
α_k^c = (1/Z) · Σ_i Σ_j ( ∂y^c / ∂A^k_ij )
L^c = ReLU( Σ_k α_k^c · A^k )
```

The resulting heat maps consistently highlight biologically meaningful regions — dark lesion spots for bacterial diseases, necrotic patches for blights, powdery white coating for mildew — rather than background clutter or fog artifacts. This is what gives the accuracy numbers above their credibility: the model is right for identifiable, inspectable reasons, not by chance.

## Project Structure

The layout below reflects the pipeline described in the thesis. Rename folders/scripts as needed to match your actual repo — this is a suggested structure, not a hard requirement.

```
.
├── Train/                    # Original PlantVillage images, 38 class sub-folders
├── split_dataset/
│   ├── train/                # 70% — clean + synthetic foggy images
│   ├── val/                  # 15% — clean only
│   └── test/                 # 15% — clean only (8,105 images)
├── train_fog/                # Synthetic foggy images, 38 class sub-folders
├── model/
│   └── plant_ai_cbam_final.keras
├── logs/
│   ├── log_phase1.csv
│   ├── log_phase2.csv
│   └── training_curves.png
├── src/
│   ├── generate_fog.py       # Atmospheric Scattering Model fog synthesis
│   ├── split_dataset.py      # Stratified 70/15/15 split
│   ├── model.py               # Backbone + CBAM + classification head
│   ├── train.py               # Two-phase training loop
│   ├── evaluate.py            # Clean-image evaluation + confusion matrix
│   ├── fog_robustness_test.py # Accuracy vs. fog density
│   ├── gradcam.py             # Grad-CAM visualization
│   └── predict.py             # Single-image inference
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.10 |
| Deep learning framework | TensorFlow 2.x / Keras |
| Image processing | OpenCV 4.x, Pillow |
| Numerical computing | NumPy |
| Metrics & data split | scikit-learn |
| Visualization | Matplotlib, seaborn |
| Development environment | Visual Studio Code |
| Hardware used | CPU only (no GPU acceleration); full two-phase training took ≈ 8–10 hours |

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

`requirements.txt`:
```
tensorflow>=2.10
opencv-python
pillow
numpy
scikit-learn
matplotlib
seaborn
```

### Dataset setup

Download [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle and place it so that each of the 38 disease folders sits under `Train/`, e.g. `Train/Apple___Apple_scab/`, `Train/Tomato___Late_blight/`, etc. The dataset is not included in this repository — see [Reproducibility Notes](#reproducibility-notes) below.

## Usage

**1. Generate the synthetic foggy dataset**
```python
# Applies the Atmospheric Scattering Model to every image in Train/
# at beta = 0.4 (light), 0.7 (medium), 1.2 (dense), writing to train_fog/
python src/generate_fog.py --input Train/ --output train_fog/
```

**2. Split into train / validation / test**
```python
# Stratified 70/15/15 split, preserving class balance
python src/split_dataset.py --clean Train/ --fog train_fog/ --output split_dataset/
```

**3. Train (two-phase transfer learning)**
```python
# Phase 1: frozen backbone, 15 epochs, lr=1e-3
# Phase 2: full fine-tune, up to 10 epochs, lr=1e-4
python src/train.py --data split_dataset/ --epochs-phase1 15 --epochs-phase2 10
```

**4. Evaluate on the clean test set**
```python
python src/evaluate.py --model model/plant_ai_cbam_final.keras --test split_dataset/test/
```

**5. Run the fog robustness benchmark**
```python
# Applies fog to the held-out test set at beta = 0.4 / 0.7 / 1.2 and reports accuracy at each level
python src/fog_robustness_test.py --model model/plant_ai_cbam_final.keras --test split_dataset/test/
```

**6. Visualize predictions with Grad-CAM**
```python
python src/gradcam.py --model model/plant_ai_cbam_final.keras --image path/to/leaf.jpg
```

**7. Predict on a single new image**
```python
python src/predict.py --model model/plant_ai_cbam_final.keras --image path/to/leaf.jpg
```

### Reproducibility notes

- **Dataset and trained model are not committed to this repo.** PlantVillage is several GB, and the trained `.keras` checkpoint can also be large — GitHub is not well suited to either. Download the dataset separately, and host the trained model via [Git LFS](https://git-lfs.com/), a GitHub Release, or a cloud link, then point to it here.
- Suggested `.gitignore` additions for this project:
  ```
  Train/
  split_dataset/
  train_fog/
  *.keras
  logs/*.csv
  __pycache__/
  venv/
  ```

## Limitations

- **Synthetic fog only** — no real field-captured foggy images were available; real-fog performance may differ from these controlled synthetic results.
- **CPU-only training** — limited the feasible number of epochs and the scope of hyperparameter search; GPU training could plausibly push accuracy higher.
- **Controlled backgrounds** — PlantVillage images are lab-photographed against plain backgrounds; accuracy on cluttered natural field backgrounds (soil, grass, other plants) is untested.
- **Single disease per image** — the model does not handle multiple co-occurring diseases on one leaf.
- **Test set size** — 8,105 images is statistically solid, but a larger, more diverse test set including real field photos would further strengthen the generalization claims.

## Future Work

- Collect a real, field-captured foggy plant disease dataset from KPK to validate synthetic-fog findings against genuine atmospheric conditions.
- Extend the evaluated fog-density range beyond β = 0.4–1.2.
- Explore hybrid dehazing + classification pipelines to push dense-fog accuracy past 75.13%.
- Convert the trained model to TensorFlow Lite and deploy it as an actual smartphone application for KPK farmers.
- Extend to region-specific disease classes beyond the standard PlantVillage 38.
- Support multi-label classification for multiple co-occurring diseases.
- Add SHAP or integrated-gradients explainability alongside Grad-CAM.
- Investigate continual/online learning so the model can adapt to new disease variants without full retraining.

## Citation

If you build on this work, please cite it as:

```bibtex
@thesis{khan_ahmad_2026_foggy_plant_disease,
  title        = {Attention-Enhanced Lightweight CNN for Plant Disease Detection Under Foggy Conditions},
  author       = {Khan, Mubbasher and Ahmad, Muhammad},
  school       = {FATA University FR Kohat},
  year         = {2026},
  address      = {Khyber Pakhtunkhwa, Pakistan},
  note         = {Bachelor's thesis, Department of Computer Science}
}
```

## Authors

- **Mubbasher Khan** — Reg. No. 2022-FU-Cs-1279
- **Muhammad Ahmad** — Reg. No. 2022-FU-Cs-1303

Department of Computer Science, FATA University FR Kohat, Khyber Pakhtunkhwa, Pakistan
Supervisor: **Mr. Sadiq Shah**, Lecturer, Department of Computer Science, FATA University

## Acknowledgments

- The creators of the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (Hughes & Salathé, 2015) for making it publicly available.
- The open-source communities behind TensorFlow, Keras, OpenCV, and scikit-learn.
- The Department of Computer Science, FATA University, for the academic environment and guidance that supported this research.

## License

No license has been set for this repository yet. If you intend to share the code publicly, consider adding an [MIT License](https://choosealicense.com/licenses/mit/) or similar — but check with your department first, since some universities retain rights over thesis code and data.
