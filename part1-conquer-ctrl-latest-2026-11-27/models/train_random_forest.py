"""单独训练随机森林：直接运行这个文件。"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import run_model


def main():
    # 128 棵树平均预测；限制叶子大小，减少过拟合。
    # 树通常不需要标准化，这里保留它使流程与拆分前完全一致。
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=128, min_samples_leaf=8,
            max_features="sqrt", random_state=2026, n_jobs=1,
        ),
    )
    return run_model("random_forest", model)


if __name__ == "__main__":
    main()
