import pickle
from pathlib import Path

# ==========================
# Path
# ==========================

# 当前文件:
# spam_mail_detector/predict/predict_sklearn.py

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "models"
    / "spam_classifier_sklearn.pkl"
)


# ==========================
# Load Model
# ==========================

with open(
    MODEL_PATH,
    "rb"
) as f:

    model_package = pickle.load(f)

model = model_package["model"]

vectorizer = model_package["vectorizer"]

print("Sklearn model loaded successfully!")


# ==========================
# Prediction Loop
# ==========================


while True:

    text = input(
        "\n请输入邮件内容（输入exit退出）:\n"
    )

    if text.lower() == "exit":
        break


    # 文本转换为TF-IDF

    X = vectorizer.transform(
        [text]
    )

    # 预测类别

    prediction = model.predict(
        X
    )[0]

    # 预测概率

    probability = model.predict_proba(
        X
    )[0][1]

    print(
        "\n========== Prediction =========="
    )

    if prediction == 1:

        print(
            "Result: Spam"
        )

    else:

        print(
            "Result: Ham"
        )

    print(
        f"Spam probability: {probability:.2%}"
    )

    print(
        "================================="
    )