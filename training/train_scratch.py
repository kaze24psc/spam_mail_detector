import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.spam_classifier_scratch import SpamClassifierScratch



# ==========================
# 路径设置
# ==========================

# 当前文件:
# spam_mail_detector/train/train.py

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "SMSSpamCollection.txt"

MODEL_PATH = ROOT / "models" / "spam_classifier.pkl"


# ==========================
# 读取数据
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
# 标签转换
# ==========================


X = df["message"]

y = df["label"].map(
    {
        "ham":0,
        "spam":1
    }
)



# ==========================
# 划分训练集测试集
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================
# 创建模型
# ==========================


model = SpamClassifierScratch(
    learning_rate=1.0,
    iterations=10000
)


# ==========================
# 训练
# ==========================

print("\nStart Training...\n")

model.fit(
    X_train,
    y_train.values
)

# ==========================
# 测试模型
# ==========================


print("\nEvaluate Model...\n")

y_pred = model.predict(
    X_test.tolist()
)

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

print(f"F1-score: {f1:.4f}")


# ==========================
# 保存模型
# ==========================


print("\nSaving Model...")

MODEL_PATH = ROOT / "models" / "spam_classifier_scratchAI.pkl"

with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

print("Model saved to:")

print(MODEL_PATH)