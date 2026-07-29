import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class SpamClassifierScratch:
    def __init__(
        self,
        learning_rate=1.0,
        iterations=10000
    ):

        self.learning_rate = learning_rate
        self.iterations = iterations

        # 文本 -> 数值矩阵
        self.vectorizer = TfidfVectorizer()

        # 参数 θ
        # 一开始不知道feature数量，所以先设置None
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = X @ self.theta
        h = self.sigmoid(z)
        return h

    def compute_loss(self, h, y):        # 计算损失函数 J(θ)
        epsilon = 1e-15
        h = np.clip(
            h,
            epsilon,
            1-epsilon
        )

        m = len(y)

        loss = -(1/m) * np.sum(
            y*np.log(h)
            +
            (1-y)*np.log(1-h)
        )

        return loss



    def fit(self, texts, y):        # 训练模型

        # Step 1:
        # 文本转换成 TF-IDF矩阵
        X = self.vectorizer.fit_transform(texts)
        # 转换成numpy array
        X = X.toarray()

        # Step 2:
        # 初始化theta
        n_features = X.shape[1]
        self.theta = np.zeros(n_features)

        # Step 3:
        # Gradient Descent
        m = X.shape[0]
        for i in range(self.iterations):
            h = self.forward(X)
            gradient = (
                1/m
                *
                X.T @ (h-y)
            )
            self.theta -= (
                self.learning_rate
                *
                gradient
            )

            if i % 1000 == 0:
                loss = self.compute_loss(h,y)
                print(f"Iteration {i}, Loss: {loss}")



    def predict_proba(self, texts):         # 返回spam的概率
        X = self.vectorizer.transform(texts)
        X = X.toarray()

        probability = self.forward(X)

        return probability



    def predict(self, texts):

        """
        分类预测

        返回:
        0 -> ham
        1 -> spam
        """

        probability = self.predict_proba(texts)
        prediction = (
            probability >= 0.5
        ).astype(int)

        return prediction



    def predict_single(self,text):

        """
        分类预测

        返回:
        0 -> ham
        1 -> spam
        """

        probability = self.predict_proba(
            [text]
        )[0]


        if probability >=0.5:
            return {
                "label":"Spam",
                "probability": float(probability)
            }

        else:
            return {
                "label":"Ham",
                "probability": float(probability)
            }