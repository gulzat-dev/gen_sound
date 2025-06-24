# Classifying Environmental Sounds by Acoustic Salience and Semantic Urgency

This repository contains the code for the paper "Justifying and Classifying Environmental Sounds by Acoustic Salience and Semantic Urgency." The project introduces a novel alertness-based classification scheme for environmental sounds and compares the performance of classic machine learning models against an end-to-end Convolutional Neural Network (CNN).

## Key Finding

A Support Vector Machine (SVM) trained on engineered MFCC+Delta features achieved a test accuracy of **0.9294**, marginally outperforming a competitive CNN baseline (0.9214). This highlights the enduring value of principled feature engineering for semantically nuanced audio tasks.

## Setup and Installation

### 1. Prerequisites
- **Python**: 3.12.x
- **`numpy`**: `< 2.0` (handled by `requirements.txt`)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```
### 3.  Download Datasets
Manually download the following datasets and place them in a data/ directory at the project root:
UrbanSound8K
ESC-50


