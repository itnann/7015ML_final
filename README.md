# Analysis of Traditional Deep Learning and Large Vision-Language Models for Medical Visual Question Answering on VQA-RAD

This project presents a comparative study of two paradigms for **Medical Visual Question Answering (Med-VQA)** on the **VQA-RAD** dataset:

- a **traditional deep learning baseline** based on **ResNet50 + LSTM**
- a **large vision-language model** based on **LLaVA-v1.5-7b** fine-tuned with **LoRA**

The goal is to evaluate how classification-based and generative multimodal models behave on two different Med-VQA settings:

- **CLOSED questions**: binary yes/no clinical questions
- **OPEN questions**: short free-text answers such as anatomical locations, findings, or device descriptions

This repository highlights both the engineering workflow and the model comparison, showing when a classical baseline remains competitive and when a generative vision-language model provides clear advantages.

---

## Overview

Medical Visual Question Answering is a multimodal task that requires a model to understand:

- a medical image
- a natural language question about that image
- and produce a clinically relevant answer

Unlike standard medical image classification, Med-VQA better reflects real clinical reasoning, where image interpretation is often guided by a sequence of targeted questions.

In this project, I:

- implemented a reproducible **CNN + LSTM** baseline
- fine-tuned **LLaVA-v1.5-7b** with **LoRA** and **4-bit quantization**
- evaluated both models separately on **OPEN** and **CLOSED** question types
- used metrics aligned with the nature of each task
- analyzed performance trade-offs, reliability concerns, and practical deployment implications

---

## Dataset

This project uses the **VQA-RAD** dataset, which contains radiology images paired with clinically generated question-answer annotations.

### Data Characteristics
- **Dataset**: VQA-RAD
- **Domain**: Radiology / Medical Imaging
- **Task**: Medical Visual Question Answering
- **Input**: Medical image + question
- **Output**: Short clinical answer

### Splits Used
- `train.json`
- `test_closed.json`
- `test_open.json`

The dataset is relatively small compared with general VQA datasets, which makes overfitting and evaluation stability important considerations.

---

## Models

## 1. Baseline Model: CNN + LSTM

The baseline follows a standard VQA classification pipeline:

- **Image Encoder**: ResNet50
- **Question Encoder**: Embedding layer + LSTM
- **Fusion**: Late fusion of image and question representations
- **Output Layer**: Fully connected classifier over a fixed answer vocabulary

This design is effective when the answer space is constrained and repetitive, especially for **CLOSED** questions.

### Training Settings
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy
- **Epochs**: 40
- **Learning Rate**: 1e-4
- **Regularization**: dropout / weight decay

---

## 2. Vision-Language Model: LLaVA-v1.5-7b

The second approach uses **LLaVA-v1.5-7b**, a large open-source vision-language model capable of generating natural language answers.

### Fine-Tuning Strategy
To reduce computational cost, the model is fine-tuned using:

- **LoRA (Low-Rank Adaptation)**
- **4-bit quantization**

This allows parameter-efficient adaptation of a 7B model under limited compute resources while preserving generative capability.

---

## Evaluation Design

A key design choice in this project is to evaluate **OPEN** and **CLOSED** questions separately, because they represent different output spaces.

### CLOSED Questions
CLOSED questions are treated as binary classification:

- **yes**
- **no**

To make evaluation robust:
- any output starting with `"yes"` is mapped to **yes**
- all other outputs are mapped to **no**

### CLOSED Metrics
- Accuracy
- Precision
- Recall
- F1-score (with **yes** as the positive class)

### OPEN Questions
OPEN answers are short free-text responses, so exact-match evaluation can be too strict.

### OPEN Metrics
- **Token-F1**
- **Semantic Accuracy**

A prediction is counted as correct if:

- **Token-F1 ≥ 0.5**

This provides an “accuracy-like” metric for a generative model and makes comparison with the baseline more practical and reproducible.

---

## Results

## Baseline Results (CNN + LSTM)

### Overall
- **ALL Top-1 Accuracy**: 46.34%
- **ALL Top-5 Accuracy**: 74.28%

### By Answer Type
- **OPEN Top-1 Accuracy**: 18.04%
- **OPEN Top-5 Accuracy**: 47.42%
- **CLOSED Top-1 Accuracy**: 67.45%
- **CLOSED Top-5 Accuracy**: 94.51%

### Interpretation
The baseline performs reasonably well on **CLOSED** questions, where the output space is simple and constrained, but is much weaker on **OPEN** questions due to the limitations of a fixed answer vocabulary.

---

## LLaVA Results

### CLOSED Performance
- **Accuracy**: 72.69%
- **Precision (yes)**: 0.7442
- **Recall (yes)**: 0.5664
- **F1-score (yes)**: 0.6432

### OPEN Performance
- **Mean Token-F1**: 0.4495
- **Semantic Accuracy (Token-F1 ≥ 0.5)**: 45.55%

### Interpretation
LLaVA maintains competitive **CLOSED** performance while showing a clear improvement on **OPEN** questions, where semantic flexibility is more important than exact answer matching.

---

## Direct Comparison

### CLOSED Questions

| Model | Accuracy | F1-score |
|------|------:|------:|
| CNN + LSTM | 67.45% | 0.7436 |
| LLaVA-v1.5-7b + LoRA | 72.69% | 0.6432 |

### OPEN Questions

| Model | Accuracy |
|------|------:|
| CNN + LSTM | 18.04% |
| LLaVA-v1.5-7b + LoRA (Semantic Accuracy) | 45.55% |

### Key Insight
- The **baseline remains competitive on CLOSED questions**
- **LLaVA is substantially stronger on OPEN questions**

---

## Discussion

### Why the Baseline Still Works Well on CLOSED Questions
CLOSED questions are often simple binary decisions tied to common clinical findings. A classification-based model with a strong visual encoder can learn stable boundaries and benefit from a constrained output space.

### Why LLaVA Performs Better on OPEN Questions
OPEN questions require more flexible language generation, concept mapping, and semantic understanding. LLaVA can produce semantically correct paraphrases even when the exact wording differs from the ground truth.

### Why Evaluation Choices Matter
A direct comparison between classification and generative models is difficult if the metrics are not aligned:

- baseline outputs labels, so accuracy is natural
- LLaVA outputs text, so exact match is often too strict

Using **Token-F1** and threshold-based **Semantic Accuracy** provides a more practical comparison.

### Reliability and Clinical Risk
A major concern for generative models is **hallucination**: producing fluent but clinically incorrect answers. While LLaVA is stronger on open-ended reasoning, any real clinical deployment would require additional safeguards, calibration, and human oversight.

### Computational Cost vs Performance
- The baseline is lightweight and efficient
- LLaVA requires more compute and more careful engineering

In limited-resource settings, a classical baseline may still be preferred for constrained tasks.

---

## Repository Structure

```text
.
├── LLaVA.ipynb
├── analysis_LLavA.ipynb
├── cnn_Lstm.ipynb
└── README.md
```

### File Description

- **LLaVA.ipynb**  
  Main notebook for dataset preparation, formatting, fine-tuning, and experimentation with LLaVA.

- **analysis_LLavA.ipynb**  
  Evaluation notebook for inference, metric calculation, comparison, and result analysis.

- **cnn_Lstm.ipynb**  
  Baseline implementation of the ResNet50 + LSTM Med-VQA model.

---

## Project Workflow

```text
VQA-RAD Dataset
      ↓
Data preprocessing and dataset splitting
      ↓
CNN + LSTM baseline training
      ↓
LLaVA-v1.5-7b fine-tuning with LoRA
      ↓
Separate OPEN / CLOSED evaluation
      ↓
Quantitative comparison and qualitative analysis
```

---

## Tech Stack

- **Python**
- **Jupyter Notebook**
- **PyTorch**
- **Transformers / Hugging Face**
- **LoRA / PEFT**
- **LLaVA**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Torchvision**

---

## Key Features

- Medical VQA on a radiology dataset
- Comparative study between **traditional deep learning** and **large vision-language models**
- Parameter-efficient fine-tuning with **LoRA**
- Separate evaluation for **OPEN** and **CLOSED** questions
- Semantic-aware evaluation using **Token-F1** and **Semantic Accuracy**
- Strong portfolio project for **multimodal AI**, **LoRA fine-tuning**, and **research-oriented model comparison**

---

## Why This Project Matters

This project demonstrates hands-on experience in:

- **medical multimodal AI**
- **vision-language model fine-tuning**
- **LoRA-based adaptation**
- **baseline design and comparison**
- **evaluation methodology for generative models**
- **notebook-based research workflows**

It is especially relevant for roles related to:

- AI / ML engineering
- multimodal systems
- LLM applications
- model experimentation and evaluation
- research-oriented prototyping

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/itnann/7015ML_final.git
cd 7015ML_final
```

### 2. Install dependencies

```bash
pip install torch torchvision transformers datasets peft pandas numpy matplotlib
```

Install any additional notebook-specific packages if required by your environment.

### 3. Prepare the dataset

Download the **VQA-RAD** dataset and place it in the correct local path expected by the notebooks.

### 4. Prepare the model

Download or configure access to **LLaVA-v1.5-7b**.

### 5. Run the notebooks

Suggested order:

1. `LLaVA.ipynb`
2. `analysis_LLavA.ipynb`
3. `cnn_Lstm.ipynb`

---

## Reproducibility Notes

- Update local dataset paths before execution
- Update model and checkpoint paths according to your environment
- GPU is recommended for LLaVA fine-tuning and inference
- Some results may depend on local runtime settings and saved checkpoints

---

## Limitations

- The project is notebook-based rather than script-based
- Reproducibility depends on local file paths and environment configuration
- The VQA-RAD dataset is relatively small
- Open-ended medical VQA remains challenging
- Generative models may produce clinically plausible but incorrect answers

---

## Future Improvements

- Convert notebooks into structured training and evaluation scripts
- Add a `requirements.txt` file
- Include clearer result visualizations and summary tables
- Explore hybrid approaches combining baseline constraints with generative flexibility
- Add uncertainty estimation or calibration for safer medical usage
- Improve repository structure for reproducibility and extension

---

## Author Contribution

**Tian Zhennan** — 100%  
- dataset preprocessing and splitting
- baseline model implementation and training
- LLaVA fine-tuning setup and execution
- evaluation design for OPEN and CLOSED metrics
- analysis, visualization, and report writing

---

## Conclusion

This project compares a classical **CNN + LSTM** Med-VQA baseline with a fine-tuned **LLaVA-v1.5-7b** model on the **VQA-RAD** dataset.

The results suggest that:

- the baseline remains practical and reasonably strong for **CLOSED** yes/no questions
- LLaVA provides clear advantages on **OPEN** questions through generative flexibility and semantic reasoning

Overall, the study shows that **generative vision-language models are a strong direction for open-ended medical reasoning**, while **classical baselines remain useful for constrained decision tasks**.

---

## References

- Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., & Parikh, D. (2015). *VQA: Visual question answering.*
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.*
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models.*
- Lau, J. J., Gayen, S., Ben Abacha, A., & Demner-Fushman, D. (2018). *A dataset of clinically generated visual questions and answers about radiology images.*
- Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual instruction tuning.*
