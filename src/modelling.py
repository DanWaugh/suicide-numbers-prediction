"""Modelling pipeline for the suicide-numbers prediction project.

Pre-processing, model definitions and evaluation live here, out of the
notebook, so the analysis is reproducible from a single import and the
notebook stays a presentation of results rather than a script.

The task: predict the number of suicides (`suicides_no`) for a country,
year, sex and age group, from the WHO-compiled Kaggle dataset "Suicide
Rates Overview 1985 to 2016". Features are age, sex, country and
population. Seed fixed (42) throughout, so every number reproduces.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["age", "sex", "country", "population"]
TARGET = "suicides_no"
RANDOM_STATE = 42

# Figures from the 2021 version of this analysis (R, Kaggle InClass
# competition), kept here so the comparison table is computed, not hand-copied.
ORIGINAL_HOLDOUT = {
    "Linear regression": 239.08,
    "KNN-3 (weighted)": 268.85,
    "KNN-6 (weighted)": 241.17,
    "Decision tree (tuned)": 65.62,
}
ORIGINAL_CV10 = {
    "Linear regression": 242.63,
    "KNN-3 (weighted)": 254.85,
    "KNN-6 (weighted)": 241.95,
    "Decision tree (tuned)": 230.86,
}


def load_data(data_dir):
    """Load the competition train/test files. Returns (train, test)."""
    data_dir = Path(data_dir)
    return (
        pd.read_csv(data_dir / "train.csv"),
        pd.read_csv(data_dir / "test.csv"),
    )


def clean_data(train):
    """Pre-processing (same cuts as the 2021 version of this analysis).

    Drops the HDI column (69% missing), rows from 2016 (incomplete
    recording) and Japan (male suicide counts of 3,000-8,000 heavily
    skewed the target), and keeps only the four features used.
    Returns (X, y, cleaned_frame).
    """
    clean = train.drop(
        columns=[
            "id",
            "country-year",
            "HDI for year",
            " gdp_for_year ($) ",
            "gdp_per_capita ($)",
            "generation",
        ]
    )
    clean = clean[(clean["year"] != 2016) & (clean["country"] != "Japan")]
    return clean[FEATURES], clean[TARGET], clean


def build_models(X, y):
    """Build the five models. The tree's depth and leaf size are chosen by
    shuffled 5-fold CV grid search, so the choice is stated, not guessed.
    Returns a dict of name -> fitted-or-fittable model.
    """
    cat = ["age", "sex", "country"]
    num = ["population"]

    # One-hot encode the categoricals, scale the numeric. For the linear
    # model and KNN the encoded matrix is scaled as well.
    pre_scaled = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
        ]
    )
    pre_tree = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", "passthrough", num),
        ]
    )

    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear regression": Pipeline(
            [
                ("pre", pre_scaled),
                ("sc", StandardScaler()),
                ("m", LinearRegression()),
            ]
        ),
        "KNN-3 (weighted)": Pipeline(
            [
                ("pre", pre_scaled),
                ("sc", StandardScaler()),
                ("m", KNeighborsRegressor(n_neighbors=3, weights="distance")),
            ]
        ),
        "KNN-6 (weighted)": Pipeline(
            [
                ("pre", pre_scaled),
                ("sc", StandardScaler()),
                ("m", KNeighborsRegressor(n_neighbors=6, weights="distance")),
            ]
        ),
    }

    tree_grid = GridSearchCV(
        Pipeline(
            [("pre", pre_tree), ("m", DecisionTreeRegressor(random_state=RANDOM_STATE))]
        ),
        {
            "m__max_depth": [3, 5, 8, 10, 15, None],
            "m__min_samples_leaf": [1, 5, 20, 50],
        },
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    tree_grid.fit(X, y)
    models["Decision tree (tuned)"] = tree_grid.best_estimator_
    return models, tree_grid.best_params_, -tree_grid.best_score_


def _metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def holdout_table(models, X, y, test_size=0.30):
    """70/30 hold-out evaluation. Returns (table, split)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        rows.append(
            {"Model": name, **{k: round(v, 3) for k, v in _metrics(y_te, model.predict(X_te)).items()}}
        )
    return pd.DataFrame(rows).set_index("Model"), (X_tr, X_te, y_tr, y_te)


def cv_table(models, X, y, folds=(3, 5, 10)):
    """Shuffled k-fold CV (RMSE). The file is sorted by country, so
    unshuffled folds would be country blocks and the numbers would not be
    comparable across fold counts. Pipelines are re-fit inside each fold,
    so the encoding/scaling does not leak.
    """
    rows = []
    for name, model in models.items():
        row = {"Model": name}
        for k in folds:
            kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(
                model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            row[f"{k}-fold CV RMSE"] = round(-scores.mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def grouped_table(models, X, y, groups, n_splits=5):
    """Grouped CV: hold out whole countries. A random split lets a model
    lean on neighbouring rows from the same country in adjacent years,
    which flatters KNN in particular. Grouping by country asks the harder,
    honest question: how well does the model predict a country it has never
    seen?
    """
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=gkf,
            groups=groups,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rows.append({"Model": name, "Grouped 5-fold CV RMSE": round(-scores.mean(), 2)})
    return pd.DataFrame(rows).set_index("Model")


def comparison_table(holdout, cv, grouped):
    """2021 comparison, computed from the tables above (no hand-copied
    numbers in the results themselves; the 2021 figures are the recorded
    values from the original report, kept in ORIGINAL_HOLDOUT/ORIGINAL_CV10).
    """
    return pd.DataFrame(
        {
            "2021 hold-out RMSE": pd.Series(ORIGINAL_HOLDOUT),
            "Hold-out RMSE": holdout["RMSE"],
            "2021 10-fold CV RMSE": pd.Series(ORIGINAL_CV10),
            "10-fold CV RMSE": cv["10-fold CV RMSE"],
            "Grouped 5-fold CV RMSE": grouped["Grouped 5-fold CV RMSE"],
        }
    ).round(2)


def predict_test(models, cv, X, y, test, features=FEATURES):
    """Fit the best model by 10-fold CV on the full cleaned training set
    and predict the competition test set. Returns the submission frame.
    """
    best_name = cv["10-fold CV RMSE"].idxmin()
    best = models[best_name]
    best.fit(X, y)
    preds = np.clip(np.rint(best.predict(test[features])), 0, None).astype(int)
    submission = pd.DataFrame({"id": test["id"], "suicides_no": preds})
    return best_name, submission
