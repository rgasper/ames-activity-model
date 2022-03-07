import json
import logging

import joblib
import numpy as np
import pandas as pd
import typer
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

cli = typer.Typer()


def optimized_train(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    param_distributions = {
        "max_depth": [None, 3, 5],
        "min_samples_split": randint(2, 11),
        "n_estimators": randint(50, 500),
    }
    estimator = RandomForestClassifier()
    optimization = RandomizedSearchCV(estimator, param_distributions, refit=True, n_iter=25, n_jobs=5)
    optimization.fit(X, y)
    best_model: RandomForestClassifier = optimization.best_estimator_
    return best_model


@cli.command()
def load_split_train_test(
    targets_file: str, features_file: str, test_frac: float, model_save_file: str, metrics_save_file: str
):
    targets_df = pd.read_csv(targets_file)
    features_array = np.genfromtxt(features_file, delimiter=",")
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, targets_df["ames"], test_size=test_frac, random_state=42
    )
    model = optimized_train(X_train, y_train)
    metrics = model.score(X_test, y_test)
    logging.critical(f"test metrics: {metrics}")
    with open(metrics_save_file, "w") as f:
        f.write(json.dumps(metrics))
    joblib.dump(model, model_save_file)


if __name__ == "__main__":
    cli()
