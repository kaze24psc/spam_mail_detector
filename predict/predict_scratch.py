import pickle
from pathlib import Path


# ==========================
# 找到项目根目录
# ==========================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT 
    / "models"
    / "spam_classifier_scratch.pkl"
)


# ==========================
# 加载模型
# ==========================

with open(
    MODEL_PATH,
    "rb"
) as f:

    model = pickle.load(f)


print("Model loaded successfully!")


# # ==========================
# # 用户输入
# # ==========================

# while True:

#     text = input(
#         "\n请输入邮件内容（输入exit退出）:\n"
#     )


#     if text == "exit":
#         break



#     # 预测

#     prediction = model.predict(
#         [text]
#     )


#     probability = model.predict_proba(
#         [text]
#     )


#     spam_probability = probability[0]



#     if prediction[0] == 1:

#         print("\nResult: Spam")

#     else:

#         print("\nResult: Ham")


#     print(
#         f"Spam probability: {spam_probability:.2%}"
#     )


email = input("请输入邮件: ")

result = model.predict_single(email)

result = model.predict_single(email)


print("\n========== Prediction ==========")
print(
    "Result:",
    result["label"]
)
print(
    f"Spam probability: {result['probability']:.2%}"
)
print("================================")