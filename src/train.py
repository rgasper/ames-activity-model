import json
import logging
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import typer
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

cli = typer.Typer()


def optimized_train(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, float]:
    param_distributions = {
        "max_depth": [None, 3, 5],
        "min_samples_split": randint(2, 11),
        "n_estimators": randint(50, 500),
    }
    estimator = RandomForestClassifier()
    optimization = RandomizedSearchCV(estimator, param_distributions, refit=True, n_iter=25, n_jobs=5, cv=10)
    optimization.fit(X, y)
    best_model: RandomForestClassifier = optimization.best_estimator_
    best_score = optimization.best_score_
    return best_model, best_score


@cli.command()
def load_split_train_test(
    targets_file: str,
    features_file: str,
    test_frac: float,
    model_save_file: str,
    metrics_save_file: str,
    num_samples: Optional[int] = None,
):
    tests = 25
    targets_df = pd.read_csv(targets_file)
    features_array = np.genfromtxt(features_file, delimiter=",")
    cv_scores = []
    test_scores = []
    for _ in range(tests):
        if num_samples is not None:
            targets_df = targets_df[:num_samples]  # samples already sorted randomly
            features_array = features_array[:num_samples]
        # regenerate random features
        # orsiloc work about 500 features - not exactly sure. https://www.nature.com/articles/s41598-021-90690-w
        features_array = np.random.rand(num_samples, 500)
        X_train, X_test, y_train, y_test = train_test_split(
            features_array, targets_df["ames"], test_size=test_frac, random_state=42
        )
        model, cv = optimized_train(X_train, y_train)
        test = model.score(X_test, y_test)
        metrics = {"cv": cv, "test": test}
        logging.critical(f"metrics: {metrics}")
        cv_scores.append(cv)
        test_scores.append(test)

    metrics = {
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "test_mean": np.mean(test_scores),
        "test_std": np.std(test_scores),
    }
    logging.critical(f"metrics: {metrics}")
    with open(metrics_save_file, "w") as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    cli()
