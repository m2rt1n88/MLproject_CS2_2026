"""可选入口：一次运行三个模型。单独运行时打开对应的 train_*.py。"""
import pandas as pd

from data_utils import OUTPUT_DIR
from train_naive_bayes import main as run_nb
from train_logistic_regression import main as run_lr
from train_random_forest import main as run_rf


def main():
    results = []
    combined = None
    keys = ["day", "matchID", "team1", "team2", "map", "team1_win"]
    for run in (run_nb, run_lr, run_rf):
        result, predictions = run()
        results.append(result)
        if combined is None:
            combined = predictions
        else:
            combined = combined.merge(predictions, on=keys, validate="one_to_one")
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"Combined results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
