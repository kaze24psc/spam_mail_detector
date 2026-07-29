import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# ==========================
# Path
# ==========================


# 当前文件:
# spam_mail_detector/training/train_sklearn.py

ROOT = Path(__file__).resolve().parent.parent


DATA_PATH = (
    ROOT
    / "data"
    / "SMSSpamCollection.txt"
)


MODEL_PATH = (
    ROOT
    / "models"
    / "spam_classifier_sklearn.pkl"
)



# ==========================
# Load Dataset
# ==========================


df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=[
        "label",
        "message"
    ]
)


print("Dataset shape:")
print(df.shape)



# ==========================
# Label Encoding
# ==========================


X = df["message"]


y = df["label"].map(
    {
        "ham":0,
        "spam":1
    }
)



# ==========================
# Train Test Split
# ==========================


X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ==========================
# TF-IDF
# ==========================


vectorizer = TfidfVectorizer()


X_train = vectorizer.fit_transform(
    X_train_text
)


X_test = vectorizer.transform(
    X_test_text
)


print("Training shape:")
print(X_train.shape)


print("Testing shape:")
print(X_test.shape)



# ==========================
# Create Model
# ==========================


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)



# ==========================
# Training
# ==========================


print("\nStart Training...\n")


model.fit(
    X_train,
    y_train
)



# ==========================
# Evaluation
# ==========================


print("\nEvaluate Model...\n")


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)



print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall: {recall:.4f}"
)


print(
    f"F1-score: {f1:.4f}"
)



print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# ==========================
# Save Model
# ==========================


model_package = {

    "model": model,

    "vectorizer": vectorizer

}



with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model_package,
        f
    )



print("\nModel saved:")
print(MODEL_PATH)