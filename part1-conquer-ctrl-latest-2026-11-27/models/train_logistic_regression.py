"""单独训练逻辑回归：直接运行这个文件。"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import run_model


def main():
    # 对标准化后的特征加权求和，再转换成胜率。
    # C 越小，正则化越强；无截距配合双方差值输入。
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, fit_intercept=False, max_iter=1500),
    )
    return run_model("logistic_regression", model)


if __name__ == "__main__":
    main()
