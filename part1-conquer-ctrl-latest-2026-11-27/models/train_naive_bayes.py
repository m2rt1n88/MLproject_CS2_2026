"""单独训练朴素贝叶斯：直接运行这个文件。"""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import run_model


def main():
    # 标准化统一量纲；在每种胜负类别中估计各特征的均值和方差。
    # var_smoothing 防止方差过小，参数保持与拆分前一致。
    model = make_pipeline(
        StandardScaler(),
        GaussianNB(var_smoothing=1.0),
    )
    return run_model("naive_bayes", model)


if __name__ == "__main__":
    main()
