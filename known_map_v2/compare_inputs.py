"""Same M2 estimator and splits; change only the input columns."""
from .runtime import HOME, configure

configure()

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data_adapter import TARGET, TWO, load, pairs
from .evaluation import assert_split, outer_splits, score
from .models import encode, estimator, get_numbers


def main():
    rows, _ = load()
    stats = pairs(rows.columns)
    labels = rows[TARGET].to_numpy()
    differences = get_numbers(rows, stats, 1) - get_numbers(rows, stats, 2)
    i, j = (stats.index(n) for n in TWO)
    variants = {
        'rating_only': differences[:, [i]],
        'map_win_rate_only': differences[:, [j]],
        'two_features': differences[:, [i, j]],
        'all_differences': differences,
        'all_differences_plus_map_interactions': encode({'view': 'diff_interaction', 'stats': stats}, rows),
    }
    measurements = []
    with threadpool_limits(1):
        for tag, x in variants.items():
            for name, tr, te in outer_splits(rows):
                assert_split(rows, tr, te, name == 'date_holdout')
                m = estimator('m2_logistic', np.tile(rows.matchID.iloc[tr], 2))
                m.fit(np.vstack([x[tr], -x[tr]]), np.r_[labels[tr], 1-labels[tr]])
                p = m.predict_proba(x[te])[:, 1]
                measurements.append(dict(input=tag, split=name, n_features=x.shape[1], **score(labels[te], p)))
    table = pd.DataFrame(measurements)
    output = HOME / 'reports'
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'input_ablation_folds.csv', index=False)
    grouped = table[table.split.ne('date_holdout')].groupby('input', sort=False)
    main_metrics = ['accuracy', 'roc_auc', 'log_loss']
    mean = grouped[main_metrics].mean().add_prefix('grouped_')
    date = table[table.split.eq('date_holdout')].set_index('input')[['n_features', 'n_test', 'correct', *main_metrics]]
    summary = date.add_prefix('date_').join(mean)
    summary.to_csv(output / 'input_ablation.csv')
    print(summary.to_string())


if __name__ == '__main__':
    main()
