# SMS Spam Detector

A machine learning project for SMS spam classification using Logistic Regression.

This project implements Logistic Regression from scratch with NumPy and compares it with the optimized implementation provided by Scikit-learn.

The goal of this project is not only to build a spam classifier, but also to understand the mathematical mechanism behind Logistic Regression and how it is applied in real machine learning pipelines.

---

# Project Overview

The complete pipeline:

```
SMS Dataset
     ↓
Text Preprocessing
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression Model
     ↓
Spam / Ham Classification
     ↓
Model Serialization
     ↓
New Message Prediction
```

Two Logistic Regression implementations are provided:

1. Logistic Regression implemented from scratch using NumPy
2. Logistic Regression implemented using Scikit-learn

---

# Features

## Implemented from Scratch

- Sigmoid function
- Forward propagation
- Binary Cross Entropy Loss
- Gradient Descent optimization
- Parameter update
- Prediction probability
- Model serialization


## Engineering Features

- Train/Test split
- TF-IDF vectorization
- Model saving with pickle
- Independent inference pipeline
- Manual test dataset evaluation


---

# Dataset

## SMS Spam Collection Dataset

The dataset contains:

- 5572 SMS messages
- Two classes:
    - ham (normal message)
    - spam (spam message)

Label encoding:
    ham  → 0
    spam → 1

## Manual Test Set

In addition to the original dataset, a manually created evaluation set containing 14 unseen messages was used to further compare the performance of the two Logistic Regression implementations.

The manual test set includes:

- 5 obvious spam messages
- 5 normal messages
- 4 challenging cases designed to test model robustness against ambiguous vocabulary and context

The test results are used to compare:

- Logistic Regression implemented from scratch
- Scikit-learn Logistic Regression implementation

The detailed comparison can be found in the **Scratch vs Scikit-learn Manual Test** section.

---

# Project Structure

```
SPAM_MAIL_DETECTOR/

│
├── data/
│   ├── SMSSpamCollection.txt          # Original SMS Spam Collection dataset
│   └── manual_test.txt                # Manually created unseen test samples
│
├── models/
│   ├── spam_classifier_scratch.pkl    # Saved model implemented from scratch
│   └── spam_classifier_sklearn.pkl    # Saved Scikit-learn Logistic Regression model
│
├── notebooks/
│   ├── 01_Logistic_Regression.ipynb           # Logistic Regression theory and experiments
│   ├── 02_TF_IDF.ipynb                        # Text feature extraction with TF-IDF
│   └── 03_LogisticRegression_FromScratch.ipynb # NumPy implementation experiments
│
├── src/
│   ├── spam_classifier_scratch.py      # Custom Logistic Regression implementation
│   └── __init__.py
│
├── training/
│   ├── train_scratch.py                # Train custom Logistic Regression model
│   ├── train_sklearn.py                # Train Scikit-learn Logistic Regression model
│   └── __init__.py
│
├── predict/
│   ├── predict_scratch.py              # Inference using custom model
│   ├── predict_sklearn.py              # Inference using Scikit-learn model
│   └── __init__.py
│
│
└── README.md                          # Project documentation
```

## Directory Description

- `data/`: Stores original dataset and manually created evaluation samples.
- `src/`: Contains model implementation code.
- `training/`: Contains scripts for training different models.
- `models/`: Stores serialized trained models.
- `predict/`: Contains inference scripts for making predictions on new messages.
- `notebooks/`: Contains experiments and mathematical exploration.

---

# Logistic Regression From Scratch

The model follows the mathematical formulation:

## Hypothesis

\[
h_\theta(x)=\sigma(\theta^Tx)
\]

where:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]


## Loss Function

Binary Cross Entropy:

\[
J(\theta)
=
-\frac1m
\sum
[
y\log(h_\theta(x))
+
(1-y)\log(1-h_\theta(x))
]
\]


## Gradient Descent

Parameter update:

\[
\theta_j
=
\theta_j-\alpha
\frac{\partial J(\theta)}{\partial\theta_j}
\]


The implementation uses NumPy to manually compute:

- prediction
- loss
- gradient
- parameter update


---

# Training Experiment

## 1. Different learning rates and iteration numbers were tested.

**Scratch Logistic Regression Performance**

| 实验 | 学习率 α | Irerations | 最终 Loss | Accuracy | F1-score |
|------|------|------|------|------|------|
| 1 | 0.01 | 1000 | 0.665 | 0.870 | 0.099 |
| 2 | 0.01 | 5000 | 0.576 | 0.872 | 0.134 |
| 3 | 0.01 | 10000 | 0.502 | 0.875 | 0.167 |
| 4 | 0.10 | 1000 | 0.502 | 0.875 | 0.167 |
| 5 | 0.10 | 5000 | 0.302 | 0.926 | 0.627 |
| 6 | 0.10 | 10000 | 0.224 | 0.952 | 0.791 |
| 7 | 0.10 | 15000 | 0.185 | 0.963 | 0.848 |
| 8 | 0.10 | 20000 | 0.160 | 0.971 | 0.885 |
| 9 | 1.00 | 5000 | 0.099 | 0.978 | 0.913 |
| 10 | 1.00 | 10000 | 0.065 | 0.982 | 0.932 |

Best Scratch Model:
Learning rate: 1.0
Iterations: 10000

Accuracy: 98.2%
F1-score: 93.2%



## 2. Scratch vs Scikit-learn

After training both models, a manually created test set was used.

The percentage represents that probability that the message is Spam


**Manual Evaluation Results**

| Case | Scratch | Sklearn | Correct Answer |
|------|------|------|------|
| 1 | Spam 62.95% | **Ham 49.39** | Spam |
| 2 | Spam 93.02% | Spam 77.04% | Spam |
| 3 | Spam 86.39% | Spam 73.06% | Spam |
| 4 | Spam 90.92% | Spam 59.38% | Spam |
| 5 | **Ham 27.30%** | **Ham 16.71%** | Spam |
| 6 | Ham 0.02% | Ham 0.88% | Ham |
| 7 | Ham 4.49% | Ham 8.35% | Ham |
| 8 | Ham 0.42% | Ham 2.67% | Ham |
| 9 | Ham 1.86% | Ham 6.45% | Ham |
| 10 | Ham 1.67% | Ham 3.71% | Ham |
| 11 | Spam 74.15% | Spam 51.99% | Spam |
| 12 | Ham 10.24% | Ham 10.68% | Ham |
| 13 | Ham 0.47% | Ham 6.56% | Ham |
| 14 | Spam 54.52% | **Ham 48.91%** | Spam |

## 3. Error Analysis 

Several difficult cases were observed.

**- False Negative Example**

Example:

```
Limited time offer! Get premium services at 90% discount.
```


Both models predicted Ham.

Possible reasons:

- The message lacks strong spam keywords such as:
    - free
    - win
    - prize
- TF-IDF representation cannot fully understand semantic meaning

**- Ambiguous Vocabulary Example**

Example:
```
I won the game yesterday!
```

The word: `won` frequently appears in spam messages.

However, the context is normal conversation.

This demonstrates a limitation of bag-of-words based models:

The model focuses on word frequency rather than deep semantic understanding.

---

# Model Comparison

## Scratch Logistic Regression

**Advantages:**

- Fully understands mathematical mechanism
- Gradient descent implemented manually
- Good educational value


**Disadvantages:**

- Requires manual optimization tuning
- Slower training


## Scikit-learn Logistic Regression

**Advantages:**

- Faster training
- Optimized solver
- More stable performance


**Disadvantages:**

- Internal optimization process is hidden


---

# How to Run

## Train Scratch Model
```
python -m training.train_scratch
```

## Train Sklearn Model
```
python -m training.train_sklearn
```

## Predict with Scratch Model
```
python -m predict.predict_scratch
```


## Predict with Sklearn Model
```
python -m predict.predict_sklearn
```


---

# Future Improvements

Possible improvements:

- Add L2 Regularization
- Try Naive Bayes classifier
- Try Support Vector Machine
- Use word embeddings instead of TF-IDF
- Build a web interface using Flask or Streamlit
- Deploy the classifier as an API

---

# Learning Outcomes

Through this project, I implemented a complete machine learning workflow:

- Mathematical understanding of Logistic Regression
- Numerical optimization with Gradient Descent
- Feature engineering for text classification
- Model evaluation
- Model serialization
- Inference pipeline construction

This project bridges the gap between machine learning theory and practical implementation.
