# Medical Visual Question Answering with LoRA Fine-Tuning on LLaVA

This project explores **medical visual question answering (Medical VQA)** on the **VQA-RAD** dataset by fine-tuning **LLaVA-1.5-7B** with **LoRA**.  
A **ResNet50 + LSTM** model is also included as a baseline for comparison.

The main goal is to investigate how a vision-language model can answer radiology-related questions from medical images, and to compare its performance with a more traditional multimodal baseline.

---

## Overview

Medical Visual Question Answering is a multimodal task where a model must understand:

- a **medical image**
- a **natural language question** about that image
- and generate the correct **answer**

In this project, I:

- used the **VQA-RAD** dataset for medical VQA
- fine-tuned **LLaVA-1.5-7B** using **LoRA**
- implemented a **ResNet50 + LSTM** baseline
- organized the workflow in notebooks for training, evaluation, and analysis
- compared different approaches for answering radiology-related questions

---

## Dataset

This project uses the **VQA-RAD** dataset, a benchmark dataset for radiology visual question answering.

- **Dataset:** VQA-RAD
- **Task:** Medical Visual Question Answering
- **Data type:** Medical images + question-answer pairs

Dataset reference:  
[VQA-RAD on Hugging Face](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)

---

## Models

### Main Model
- **LLaVA-1.5-7B**
- Fine-tuned with **LoRA** for parameter-efficient adaptation

Model reference:  
[LLaVA-1.5-7B on Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b)

### Baseline Model
- **ResNet50 + LSTM**

The baseline is used to compare a traditional multimodal architecture with the fine-tuned vision-language model.

---

## Repository Structure

The repository currently contains three notebooks and a README. :contentReference[oaicite:1]{index=1}

```text
.
├── LLaVA.ipynb
├── analysis_LLavA.ipynb
├── cnn_Lstm.ipynb
└── README.md

File Description
LLaVA.ipynb
Main notebook for dataset preparation, formatting, and LoRA fine-tuning workflow for LLaVA.
analysis_LLavA.ipynb
Notebook for model inference, result analysis, and evaluation.
cnn_Lstm.ipynb
Baseline implementation using ResNet50 + LSTM.
Methodology
1. Data Preparation

The VQA-RAD dataset is prepared for multimodal training and evaluation.

Typical steps include:

loading image-question-answer samples
formatting the data for LLaVA-style instruction tuning
splitting the dataset into training and testing subsets
organizing samples for later evaluation and comparison
2. LoRA Fine-Tuning

Instead of full fine-tuning, this project uses LoRA (Low-Rank Adaptation) to efficiently adapt LLaVA-1.5-7B to the medical VQA domain.

Advantages of LoRA:

lower memory usage
fewer trainable parameters
faster experimentation
practical for limited compute settings
3. Baseline Modeling

A ResNet50 + LSTM architecture is used as a baseline:

ResNet50 extracts visual features from medical images
LSTM processes the text question
the combined representation is used to predict the answer
4. Evaluation and Analysis

The project includes a dedicated analysis notebook to:

run inference
inspect predictions
compare model outputs
analyze results qualitatively and/or quantitatively
Project Workflow
VQA-RAD Dataset
      ↓
Data preprocessing and formatting
      ↓
LLaVA + LoRA fine-tuning
      ↓
Model inference and evaluation
      ↓
Result analysis and comparison
      ↓
Baseline comparison with ResNet50 + LSTM
Tech Stack
Python
Jupyter Notebook
PyTorch
Transformers / Hugging Face
LoRA / PEFT
LLaVA
NumPy / Pandas
Matplotlib
Torchvision
Key Features
Medical VQA task on radiology data
LoRA-based fine-tuning for efficient adaptation
Vision-language modeling with LLaVA
Baseline comparison with ResNet50 + LSTM
Notebook-based experimentation and analysis
Practical multimodal AI project for research and portfolio presentation
Results

You can update this section with your actual experiment results.

Example Result Table
Model	Split	Metric	Score
LLaVA-1.5-7B + LoRA	Test	Accuracy	XX.XX
ResNet50 + LSTM	Test	Accuracy	XX.XX

You can also include:

sample predictions
failure cases
qualitative comparisons
visual examples from the dataset
Example Discussion
The LoRA-fine-tuned LLaVA model may show stronger multimodal reasoning ability.
The ResNet50 + LSTM baseline provides a useful reference point for performance comparison.
Results can highlight the benefits and limitations of large vision-language models in medical image understanding.
How to Run
1. Clone the repository
git clone https://github.com/itnann/7015ML_final.git
cd 7015ML_final
2. Install dependencies

You can install the required packages in a notebook or Python environment:

pip install torch torchvision transformers datasets peft pandas numpy matplotlib

If your notebook uses additional libraries, install them as needed.

3. Prepare the dataset

Download the VQA-RAD dataset and place it in your local working directory.

4. Prepare the base model

Download or access LLaVA-1.5-7B according to your environment and hardware setup.

5. Run the notebooks

Suggested order:

LLaVA.ipynb
analysis_LLavA.ipynb
cnn_Lstm.ipynb
Reproducibility Notes
Update local dataset paths before running the notebooks
Update model paths according to your machine or cloud environment
GPU is recommended for fine-tuning and inference with LLaVA
Some outputs may depend on local checkpoints and runtime settings
Limitations
The repository is currently notebook-based rather than script-based
Reproducibility depends on local file paths and environment setup
Medical VQA datasets are relatively small, so generalization may be limited
Open-ended medical question answering remains challenging
Future Improvements
Convert notebooks into structured training and evaluation scripts
Add a requirements.txt file
Add clearer result tables and visualizations
Add more evaluation metrics beyond simple accuracy
Compare additional multimodal baselines
Improve repository organization for research reproducibility
Why This Project Matters

This project demonstrates practical experience in:

multimodal AI
LoRA fine-tuning
vision-language models
medical VQA
baseline comparison
notebook-based research workflow

It is a strong portfolio project for roles related to:

AI / ML engineering
LLM applications
multimodal systems
research-oriented model experimentation
