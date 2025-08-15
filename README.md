# 📸 Image Captioning with CNN–RNN Encoder–Decoder Architecture

---

## 📌 Abstract

This repository contains the complete implementation, experimental analysis, and evaluation of an **image captioning system** based on the **CNN–RNN Encoder–Decoder** framework. The project integrates **ResNet18** as the visual feature extractor and **LSTM** as the sequence generator, trained on the **Flickr8k** dataset. The model is evaluated using **BLEU-1 to BLEU-4** scores, comparing **Greedy Search** and **Beam Search** decoding strategies.

The work is conducted as part of the **Special Topics in Artificial Intelligence (Deep Learning)** course, **Spring 2025**, Shahid Bahonar University of Kerman, under the supervision of **Dr. Eftekhari**.

---

## 1. Introduction

Automatic image captioning aims to generate a **human-readable description** for an input image, combining the strengths of **computer vision** (for visual feature extraction) and **natural language processing** (for sequence generation).

### Problem Definition

Given an image `I`, the goal is to generate a textual sequence `S = {w₁, w₂, ..., wₙ}` that accurately describes the salient content of `I`. The task requires:

1. **Visual understanding**: Identifying objects, actions, and relationships in the image.
2. **Language modeling**: Producing syntactically correct and semantically coherent captions.

### Significance

Image captioning has wide applications, including:

* Accessibility for visually impaired users.
* Content-based image retrieval.
* Social media image tagging.
* Autonomous robotics interaction.

---

## 2. Related Work

### 2.1 Show and Tell (Vinyals et al., 2015)

Introduced the first end-to-end CNN–RNN captioning model. Achieved competitive results but suffered from repetitive captions and lack of focus on specific image regions.

### 2.2 Show, Attend and Tell (Xu et al., 2015)

Enhanced the encoder–decoder architecture with an **attention mechanism**, enabling the model to focus on relevant regions at each word-generation step. This significantly improved captions for complex images.

**Our model** follows the classical encoder–decoder paradigm without attention, making it efficient and easy to train but still prone to common CNN–RNN limitations (e.g., generic captions, lack of spatial focus).

---

## 3. Methodology

### 3.1 Dataset — Flickr8k

* **Size**: 8,000 images.
* **Captions**: 5 per image (40,000 total).
* **Variety**: Scenes include humans, animals, landscapes, and urban environments.

---

### 3.2 Data Preprocessing

* **Image transformations**: Resize, center crop, normalization (ImageNet mean/std).
* **Text processing**: Tokenization, lowercasing, punctuation removal.
* **Vocabulary construction**: Index mapping for each unique word; special tokens `<START>`, `<END>`, `<PAD>`.

---

### 3.3 Model Architecture

![Architecture Diagram](docs/architecture.png)

**Encoder**:

* **Backbone**: Pre-trained **ResNet18** (ImageNet).
* Final classification layers removed.
* Outputs a fixed-length feature vector.

**Decoder**:

* **Embedding size**: 256
* **LSTM hidden size**: 512
* **Number of layers**: 1–2
* **Dropout**: Applied between LSTM layers to reduce overfitting.

**Training strategy**:

* **Teacher Forcing** for faster convergence.
* **Loss**: CrossEntropyLoss.
* **Optimizer**: Adam (`lr=1e-3`).
* **Epochs**: 20, batch size = 64.

---

## 4. Implementation

### Directory Structure

```
image-captioning/
├── data/                      # Dataset scripts & loaders
├── models/                    # Encoder, Decoder, Combined model
├── utils/                     # Vocabulary, Metrics, Trainer
├── notebooks/                 # Step-by-step workflow notebooks
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

* `encoder.py`: CNN-based feature extractor.
* `decoder.py`: LSTM-based language generator.
* `caption_model.py`: Combines encoder & decoder into a single pipeline.
* `metrics.py`: BLEU score implementation.

---

## 5. Experimental Setup

### 5.1 Decoding Strategies

* **Greedy Search**: Selects the highest-probability word at each step.
* **Beam Search**: Explores top-k probable sequences to improve overall caption quality.

---

### 5.2 Evaluation Metrics

We use **BLEU-n** scores (Papineni et al., 2002), where higher scores indicate better alignment between generated and reference captions:

$$
\text{BLEU-n} = BP \cdot \exp \left( \frac{1}{n} \sum_{i=1}^n \log p_i \right)
$$

Where $p_i$ is the precision of i-grams, and $BP$ is the brevity penalty.

---

## 6. Results

### 6.1 BLEU Scores

**Greedy Search**

| Metric | Score  |
| ------ | ------ |
| BLEU-1 | 0.2536 |
| BLEU-2 | 0.0660 |
| BLEU-3 | 0.0189 |
| BLEU-4 | 0.0062 |

**Beam Search**

| Metric | Score  |
| ------ | ------ |
| BLEU-1 | 0.2607 |
| BLEU-2 | 0.0756 |
| BLEU-3 | 0.0261 |
| BLEU-4 | 0.0081 |

**Observation:**
Beam Search consistently outperforms Greedy Search, particularly at higher-order BLEU scores.

---

### 6.2 Qualitative Examples

| Input Image           | Greedy Output                  | Beam Search Output                   |
| --------------------- | ------------------------------ | ------------------------------------ |
| ![](docs/sample1.jpg) | "a man is standing"            | "a man is standing"                  |
| ![](docs/sample2.jpg) | "a dog running in the grass"   | "a brown dog runs through the grass" |
| ![](docs/sample3.jpg) | "a group of people on a beach" | "people walking along the beach"     |

---

### 6.3 Error Analysis

* **Repetitive captions** for different images.
* **Gender bias**: Many female subjects misclassified as male.
* **Weak performance on crowded/complex scenes**.

---

## 7. Discussion

* ResNet18 provides a good **trade-off** between feature richness and computational efficiency.
* Beam Search offers **slight but consistent improvements** over Greedy decoding.
* Without attention, the model lacks spatial awareness, leading to **loss of detail** in complex scenes.

---

## 8. Future Work

* Replace encoder with **ResNet50/EfficientNet** for richer features.
* Integrate **Attention Mechanism** to focus on important image regions.
* Train on **larger datasets** (e.g., MS-COCO) to improve generalization.
* Use **pre-trained embeddings** (GloVe, FastText).
* Evaluate with **CIDEr** and **METEOR** metrics.
* Explore **Transformer-based** architectures (e.g., BLIP, ViT).

---

## 9. Installation & Usage

```bash
git clone https://github.com/HivaAbolhadizade/image-captioning
cd image-captioning
pip install -r requirements.txt
```

Run the notebooks in order:

1. `1_Data_Exploration.ipynb`
2. `2_Feature_Extraction.ipynb`
3. `3_Model_Training.ipynb`
4. `4_Evaluation_Visualization.ipynb`

---

## 10. Citation

```
@project{abolhadizadeh2025imagecaptioning,
  title={Image Captioning with CNN–RNN Encoder–Decoder Architecture},
  author={Hiva Abolhadizadeh},
  year={2025},
  institution={Shahid Bahonar University of Kerman},
  note={Special Topics in AI (Deep Learning), Spring 2025}
}
```


If you want, I can now **design the actual figures** (architecture diagram, training flow, BLEU score bar charts, sample captions) so your GitHub README will look visually like a **research project page**.
Do you want me to prepare those visuals for you?
